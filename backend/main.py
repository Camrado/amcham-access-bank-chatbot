from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from pathlib import Path
from typing import Optional, List
import time

# Configure logging before any chatbot imports so basicConfig in agent.py is a no-op
from logging_config import setup_logging
setup_logging()

from database import engine, get_db
import models
from models import Conversation, Message
from auth.router import router as auth_router
from auth.service import TokenClaims, get_claims
from conversations.router import router as conversations_router
from admin.router import router as admin_router

# ─── Agent imports ────────────────────────────────────────────────────────────
from chatbot.agent import Agent, get_case, update_case_status
from chatbot.anomaly import get_active_anomalies, resolve_anomaly, get_department_volume
from chatbot.case_similarity import find_similar_cases, format_similarity_hint
import sqlite3, json, os, logging
from datetime import datetime

logger = logging.getLogger("main")
frontend_logger = logging.getLogger("frontend")

AGENT_DB = os.environ.get("DB_PATH", "cases.db")

_agent = Agent(db_path=AGENT_DB)

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AccessBank Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(admin_router)


# ─── HTTP request / response logging ─────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s %d (%.0fms)", request.method, request.url.path, response.status_code, ms)
    return response


# ═════════════════════════════════════════════════════════════════════════════
# Schemas
# ═════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    case_id: Optional[str] = None
    department: Optional[str] = None
    flagged: bool = False
    language: Optional[str] = None
    sentiment: Optional[str] = None
    urgency: Optional[str] = None
    priority_boost: bool = False
    similar_cases: Optional[list] = None
    anomaly: Optional[dict] = None
    # Indicates an escalation email was dispatched to the relevant department
    email_routed: bool = False


class CaseStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(open|pending|resolved|closed)$")


class AdminReplyRequest(BaseModel):
    reply: str = Field(..., min_length=1)


class AnomalyResolveRequest(BaseModel):
    anomaly_id: str


class FrontendLogRequest(BaseModel):
    level: str = Field(..., max_length=10)
    message: str = Field(..., max_length=2000)
    url: Optional[str] = Field(None, max_length=300)
    user_agent: Optional[str] = Field(None, max_length=300)


# ═════════════════════════════════════════════════════════════════════════════
# POST /api/logs  — receives client-side ERROR reports from the browser
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/logs", status_code=204)
async def receive_frontend_log(body: FrontendLogRequest):
    frontend_logger.warning("client | url=%s | %s", body.url or "?", body.message)


# ═════════════════════════════════════════════════════════════════════════════
# Helper — load conversation history from accessbank.db messages table
# ═════════════════════════════════════════════════════════════════════════════

def _load_history(conversation_id: int, db: Session) -> list[dict]:
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .limit(20)
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in messages]


def _load_pending_state(conversation_id: int, db: Session) -> tuple[Optional[str], Optional[list]]:
    last_system = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.role == "system",
        )
        .order_by(Message.created_at.desc())
        .first()
    )
    if last_system:
        try:
            data = json.loads(last_system.content)
            if data.get("type") == "pending_state":
                return data.get("department"), data.get("missing_info")
        except (json.JSONDecodeError, AttributeError):
            pass
    return None, None


def _save_pending_state(
    conversation_id: int,
    department: Optional[str],
    missing_info: Optional[list],
    db: Session,
) -> None:
    db.query(Message).filter(
        Message.conversation_id == conversation_id,
        Message.role == "system",
    ).delete()

    if department is not None:
        payload = json.dumps({
            "type": "pending_state",
            "department": department,
            "missing_info": missing_info or [],
        })
        db.add(Message(
            conversation_id=conversation_id,
            role="system",
            content=payload,
        ))


# ═════════════════════════════════════════════════════════════════════════════
# POST /chat  — main customer-facing endpoint
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    claims: TokenClaims = Depends(get_claims),
    db: Session = Depends(get_db),
) -> ChatResponse:
    conv = db.query(Conversation).filter(
        Conversation.id == body.conversation_id,
        Conversation.user_id == claims.user_id,
    ).first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    is_first = not db.query(Message).filter(
        Message.conversation_id == conv.id,
        Message.role == "user",
    ).first()
    if is_first and conv.title.startswith("Chat · "):
        conv.title = body.message[:50] + ("..." if len(body.message) > 50 else "")
        db.add(conv)

    history = _load_history(body.conversation_id, db)
    pending_department, pending_missing_info = _load_pending_state(body.conversation_id, db)

    logger.info(
        "Chat request: conv=%s user=%s pending_dept=%s",
        body.conversation_id, claims.user_id, pending_department,
    )

    response = _agent.handle(
        user_id=str(claims.user_id),
        message=body.message,
        history=history,
        pending_department=pending_department,
        pending_missing_info=pending_missing_info,
    )

    db.add(Message(
        conversation_id=conv.id,
        role="user",
        content=body.message,
    ))
    db.add(Message(
        conversation_id=conv.id,
        role="assistant",
        content=response.text,
    ))

    if response.intent == "issue" and response.case_id is None and not response.flagged:
        _save_pending_state(conv.id, response.department, [], db)
    else:
        _save_pending_state(conv.id, None, None, db)

    db.commit()

    logger.info(
        "Chat response: conv=%s intent=%s case=%s flagged=%s lang=%s sentiment=%s email_routed=%s",
        body.conversation_id, response.intent, response.case_id,
        response.flagged, response.language, response.sentiment, response.email_routed,
    )

    return ChatResponse(
        reply=response.text,
        intent=response.intent,
        case_id=response.case_id,
        department=response.department,
        flagged=response.flagged,
        language=response.language,
        sentiment=response.sentiment,
        urgency=response.urgency,
        priority_boost=response.priority_boost,
        similar_cases=response.similar_cases,
        anomaly=response.anomaly,
        email_routed=response.email_routed,
    )


# ═════════════════════════════════════════════════════════════════════════════
# GET /conversations/{conversation_id}/messages
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: int,
    claims: TokenClaims = Depends(get_claims),
    db: Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == claims.user_id,
    ).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id,
            Message.role != "system",
        )
        .order_by(Message.created_at.asc())
        .all()
    )
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": str(m.created_at),
        }
        for m in messages
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Cases — admin endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/cases")
async def list_cases(
    status_filter: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 50,
):
    with sqlite3.connect(AGENT_DB) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT id, user_id, department, summary, status, created_at, updated_at FROM cases"
        conditions, params = [], []
        if status_filter:
            conditions.append("status = ?")
            params.append(status_filter)
        if department:
            conditions.append("department = ?")
            params.append(department)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/cases/{case_id}")
async def get_case_detail(case_id: str):
    case = get_case(case_id, AGENT_DB)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    similar = find_similar_cases(
        summary=case["summary"],
        db_path=AGENT_DB,
        top_k=3,
        min_score=0.55,
        exclude_case_id=case_id,
    )
    case["similar_cases"] = similar
    case["similar_cases_hint"] = format_similarity_hint(similar)
    return case


@app.patch("/cases/{case_id}/status")
async def update_status(case_id: str, body: CaseStatusUpdate):
    case = get_case(case_id, AGENT_DB)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    update_case_status(case_id, body.status, AGENT_DB)
    return {"case_id": case_id, "status": body.status, "updated": True}


# ═════════════════════════════════════════════════════════════════════════════
# Flagged conversations — admin queue
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/flagged")
async def list_flagged(resolved: bool = False, limit: int = 50):
    with sqlite3.connect(AGENT_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, user_id, flag_reason, admin_reply, resolved, created_at, updated_at
            FROM flagged_conversations
            WHERE resolved = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (1 if resolved else 0, limit),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/flagged/{flag_id}")
async def get_flagged_detail(flag_id: str):
    with sqlite3.connect(AGENT_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM flagged_conversations WHERE id = ?", (flag_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Flagged conversation not found")
    result = dict(row)
    result["history"] = json.loads(result["history"])
    return result


@app.post("/flagged/{flag_id}/reply")
async def admin_reply_to_flagged(flag_id: str, body: AdminReplyRequest):
    with sqlite3.connect(AGENT_DB) as conn:
        row = conn.execute(
            "SELECT id FROM flagged_conversations WHERE id = ?", (flag_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Flagged conversation not found")
        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            UPDATE flagged_conversations
            SET admin_reply = ?, resolved = 1, updated_at = ?
            WHERE id = ?
            """,
            (body.reply, now, flag_id),
        )
        conn.commit()
    logger.info("Admin replied to flagged conversation %s", flag_id)
    return {"flag_id": flag_id, "resolved": True, "reply": body.reply}


# ═════════════════════════════════════════════════════════════════════════════
# Anomaly alerts
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/anomalies")
async def list_anomalies():
    return get_active_anomalies(AGENT_DB)


@app.post("/anomalies/{anomaly_id}/resolve")
async def resolve_anomaly_endpoint(anomaly_id: str):
    resolve_anomaly(anomaly_id, AGENT_DB)
    return {"anomaly_id": anomaly_id, "resolved": True}


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard stats
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/stats")
async def dashboard_stats():
    with sqlite3.connect(AGENT_DB) as conn:
        conn.row_factory = sqlite3.Row

        status_rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM cases GROUP BY status"
        ).fetchall()
        by_status = {r["status"]: r["count"] for r in status_rows}

        total = conn.execute("SELECT COUNT(*) as c FROM cases").fetchone()["c"]

        today = datetime.utcnow().date().isoformat()
        today_count = conn.execute(
            "SELECT COUNT(*) as c FROM cases WHERE created_at >= ?", (today,)
        ).fetchone()["c"]

        flagged_pending = conn.execute(
            "SELECT COUNT(*) as c FROM flagged_conversations WHERE resolved = 0"
        ).fetchone()["c"]

    dept_volume = get_department_volume(AGENT_DB, window_minutes=60)
    anomalies = get_active_anomalies(AGENT_DB)

    return {
        "total_cases": total,
        "cases_today": today_count,
        "flagged_pending": flagged_pending,
        "by_status": by_status,
        "department_volume_60min": dept_volume,
        "active_anomalies": len(anomalies),
        "anomaly_alerts": anomalies,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Static frontend
# ═════════════════════════════════════════════════════════════════════════════

_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")