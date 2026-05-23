from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from pathlib import Path
from database import engine, get_db
import models
from models import Conversation, Message
from auth.router import router as auth_router
from auth.service import TokenClaims, get_claims
from conversations.router import router as conversations_router
from admin.router import router as admin_router

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AccessBank Support Agent")

app.include_router(auth_router)
app.include_router(conversations_router)
app.include_router(admin_router)


class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str


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

    # Update title from first user message
    is_first = not db.query(Message).filter(Message.conversation_id == conv.id).first()
    if is_first and conv.title.startswith("Chat · "):
        conv.title = body.message[:50] + ("..." if len(body.message) > 50 else "")
        db.add(conv)

    db.add(Message(conversation_id=conv.id, role="user", content=body.message))

    # TODO: replace with AI agent response
    reply = "Thank you for contacting AccessBank support. Our AI agent is being set up and will be ready soon."

    db.add(Message(conversation_id=conv.id, role="assistant", content=reply))
    db.commit()

    return ChatResponse(reply=reply)


_frontend = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
