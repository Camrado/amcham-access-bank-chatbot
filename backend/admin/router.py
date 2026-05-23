import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import User, Conversation, Message
from auth.service import TokenClaims, require_admin
from admin.schemas import (
    UserSummary, MessageOut, AdminReplyRequest,
    ConversationDetail, CaseInConversation,
)

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger("admin")

AGENT_DB = os.environ.get("DB_PATH", "cases.db")
_CASE_RE = re.compile(r'CASE-[0-9A-F]{8}')


def _ensure_schema() -> None:
    """Add columns that postdate the original cases.db schema."""
    with sqlite3.connect(AGENT_DB) as conn:
        try:
            conn.execute("ALTER TABLE cases ADD COLUMN admin_reply TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


_ensure_schema()


# ─── cases.db helpers ─────────────────────────────────────────────────────────

def _cases_for_user(user_id: int) -> list[dict]:
    with sqlite3.connect(AGENT_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, department, status FROM cases WHERE user_id = ? ORDER BY created_at DESC",
            (str(user_id),),
        ).fetchall()
    return [dict(r) for r in rows]


def _get_case(case_id: str) -> Optional[dict]:
    with sqlite3.connect(AGENT_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    return dict(row) if row else None


def _resolve_case(case_id: str, admin_reply: str) -> None:
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(AGENT_DB) as conn:
        # Add column if the DB predates this feature
        try:
            conn.execute("ALTER TABLE cases ADD COLUMN admin_reply TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute(
            "UPDATE cases SET status = 'resolved', admin_reply = ?, updated_at = ? WHERE id = ?",
            (admin_reply, now, case_id),
        )
        conn.commit()


def _case_for_conv(conv_id: int, db: Session) -> Optional[dict]:
    """Find the case linked to a conversation by scanning its assistant messages."""
    msg = (
        db.query(Message)
        .filter(
            Message.conversation_id == conv_id,
            Message.role == "assistant",
            Message.content.like("%CASE-%"),
        )
        .first()
    )
    if not msg:
        return None
    m = _CASE_RE.search(msg.content)
    return _get_case(m.group(0)) if m else None


def _conv_for_case(case_id: str, db: Session) -> Optional[int]:
    """Return the conversation_id where this case was announced."""
    msg = (
        db.query(Message)
        .filter(
            Message.role == "assistant",
            Message.content.like(f"%{case_id}%"),
        )
        .first()
    )
    return msg.conversation_id if msg else None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/open-cases")
def list_open_cases(
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    with sqlite3.connect(AGENT_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, user_id, department, summary, status, created_at FROM cases "
            "WHERE status = 'open' ORDER BY created_at ASC",
        ).fetchall()

    result = []
    for row in rows:
        case = dict(row)
        try:
            user = db.query(User).filter(User.id == int(case["user_id"])).first()
        except (ValueError, TypeError):
            user = None
        case["user_name"] = user.username if user else f"User #{case['user_id']}"
        case["user_contact"] = user.email if user else ""
        result.append(case)

    logger.info("list_open_cases: admin_id=%d count=%d", claims.user_id, len(result))
    return result


@router.get("/users", response_model=list[UserSummary])
def list_users(
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    logger.info("list_users: admin_id=%d", claims.user_id)
    users = db.query(User).filter(User.is_admin == False).order_by(User.created_at.desc()).all()
    result = []
    for user in users:
        cases = _cases_for_user(user.id)
        result.append(UserSummary(
            id=user.id,
            username=user.username,
            email=user.email,
            open_cases=sum(1 for c in cases if c["status"] == "open"),
            total_cases=len(cases),
        ))
    return result


@router.get("/users/{user_id}/conversations", response_model=list[ConversationDetail])
def get_user_conversations(
    user_id: int,
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    logger.info("view_conversations: admin_id=%d target_user_id=%d", claims.user_id, user_id)
    user = db.query(User).filter(User.id == user_id, User.is_admin == False).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
        .all()
    )

    result = []
    for conv in conversations:
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conv.id, Message.role != "system")
            .order_by(Message.created_at.asc())
            .all()
        )

        raw_case = _case_for_conv(conv.id, db)
        case_out = None
        if raw_case:
            case_out = CaseInConversation(
                id=raw_case["id"],
                user_name=user.username,
                user_contact=user.email,
                issue_summary=raw_case["summary"],
                department=raw_case["department"],
                status=raw_case["status"],
                email_ref=raw_case.get("email_ref"),
                admin_reply=raw_case.get("admin_reply"),
            )

        result.append(ConversationDetail(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            messages=[MessageOut.model_validate(m) for m in messages],
            case=case_out,
        ))
    return result


@router.post("/cases/{case_id}/reply")
def reply_to_case(
    case_id: str,
    body: AdminReplyRequest,
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    case = _get_case(case_id)
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    if case["status"] == "resolved":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Case already resolved")

    # Add reply message to the user's conversation so they see it in the chat
    conv_id = _conv_for_case(case_id, db)
    if conv_id:
        db.add(Message(
            conversation_id=conv_id,
            role="assistant",
            content=f"Support team: {body.reply_text}",
        ))
        db.commit()

    _resolve_case(case_id, body.reply_text)
    logger.info("case_reply: admin_id=%d case_id=%s", claims.user_id, case_id)
    return {"ok": True}
