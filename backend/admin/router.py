from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import User, Conversation, Message, Case
from auth.service import TokenClaims, require_admin
from admin.schemas import (
    UserSummary, CaseDetail, MessageOut, AdminReplyRequest,
    ConversationDetail, CaseInConversation,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserSummary])
def list_users(
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    users = db.query(User).filter(User.is_admin == False).order_by(User.created_at.desc()).all()
    result = []
    for user in users:
        cases = db.query(Case).filter(Case.user_id == user.id).all()
        result.append(UserSummary(
            id=user.id,
            username=user.username,
            email=user.email,
            open_cases=sum(1 for c in cases if c.status == "open"),
            total_cases=len(cases),
        ))
    return result


@router.get("/users/{user_id}/conversations", response_model=list[ConversationDetail])
def get_user_conversations(
    user_id: int,
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id, User.is_admin == False).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.created_at.desc()).all()

    result = []
    for conv in conversations:
        messages = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(Message.created_at.asc()).all()

        case = db.query(Case).filter(Case.conversation_id == conv.id).first()

        result.append(ConversationDetail(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            messages=[MessageOut.model_validate(m) for m in messages],
            case=CaseInConversation.model_validate(case) if case else None,
        ))
    return result


@router.get("/users/{user_id}/cases", response_model=list[CaseDetail])
def get_user_cases(
    user_id: int,
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id, User.is_admin == False).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    cases = db.query(Case).filter(Case.user_id == user_id).order_by(Case.created_at.desc()).all()
    result = []
    for case in cases:
        messages = []
        if case.conversation_id:
            messages = db.query(Message).filter(
                Message.conversation_id == case.conversation_id
            ).order_by(Message.created_at.asc()).all()
        result.append(CaseDetail(
            id=case.id,
            user_name=case.user_name,
            user_contact=case.user_contact,
            issue_summary=case.issue_summary,
            department=case.department,
            status=case.status,
            email_ref=case.email_ref,
            admin_reply=case.admin_reply,
            conversation_id=case.conversation_id,
            created_at=case.created_at,
            messages=[MessageOut.model_validate(m) for m in messages],
        ))
    return result


@router.post("/cases/{case_id}/reply")
def reply_to_case(
    case_id: int,
    body: AdminReplyRequest,
    claims: TokenClaims = Depends(require_admin),
    db: Session = Depends(get_db),
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    if case.status == "resolved":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Case already resolved")

    if case.conversation_id:
        db.add(Message(
            conversation_id=case.conversation_id,
            role="assistant",
            content=f"Support team: {body.reply_text}",
        ))

    case.status = "resolved"
    case.admin_reply = body.reply_text
    db.commit()
    return {"ok": True}
