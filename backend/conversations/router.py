import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime
from database import get_db
from models import Conversation, Message
from auth.service import TokenClaims, get_claims
from conversations.schemas import ConversationOut, ConversationUpdate, MessageOut

router = APIRouter(prefix="/conversations", tags=["conversations"])
logger = logging.getLogger("conversations")


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def create_conversation(
    claims: TokenClaims = Depends(get_claims),
    db: Session = Depends(get_db),
):
    title = "Chat · " + datetime.now().strftime("%b %d, %H:%M")
    conv = Conversation(user_id=claims.user_id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    logger.info("created: user_id=%d conv_id=%d", claims.user_id, conv.id)
    return conv


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    claims: TokenClaims = Depends(get_claims),
    db: Session = Depends(get_db),
):
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == claims.user_id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    result = []
    for conv in convs:
        last_msg = (
            db.query(Message)
            .filter(
                Message.conversation_id == conv.id,
                Message.role.in_(["user", "assistant"]),
            )
            .order_by(Message.created_at.desc())
            .first()
        )
        result.append(ConversationOut(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            last_message_at=last_msg.created_at if last_msg else None,
            last_message_role=last_msg.role if last_msg else None,
        ))
    return result


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
def get_messages(
    conversation_id: int,
    claims: TokenClaims = Depends(get_claims),
    db: Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == claims.user_id,
    ).first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )


@router.patch("/{conversation_id}", response_model=ConversationOut)
def update_conversation(
    conversation_id: int,
    body: ConversationUpdate,
    claims: TokenClaims = Depends(get_claims),
    db: Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == claims.user_id,
    ).first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    conv.title = body.title
    db.commit()
    db.refresh(conv)
    return conv


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    claims: TokenClaims = Depends(get_claims),
    db: Session = Depends(get_db),
):
    conv = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == claims.user_id,
    ).first()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()
    db.delete(conv)
    db.commit()
    logger.info("deleted: user_id=%d conv_id=%d", claims.user_id, conversation_id)
