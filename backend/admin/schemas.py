from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


class UserSummary(BaseModel):
    id: int
    username: str
    email: str
    open_cases: int
    total_cases: int


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class CaseInConversation(BaseModel):
    id: str                          # "CASE-XXXXXXXX" from cases.db
    user_name: str
    user_contact: str
    issue_summary: str
    department: str
    status: str
    email_ref: Optional[str] = None
    admin_reply: Optional[str] = None


class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: list[MessageOut] = []
    case: Optional[CaseInConversation] = None


class AdminReplyRequest(BaseModel):
    reply_text: str
