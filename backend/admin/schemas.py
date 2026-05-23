from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


class UserSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_name: str
    user_contact: str
    issue_summary: str
    department: str
    status: str
    email_ref: Optional[str]
    admin_reply: Optional[str]


class ConversationDetail(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: list[MessageOut] = []
    case: Optional[CaseInConversation] = None


class CaseDetail(BaseModel):
    id: int
    user_name: str
    user_contact: str
    issue_summary: str
    department: str
    status: str
    email_ref: Optional[str]
    admin_reply: Optional[str]
    conversation_id: Optional[int]
    created_at: datetime
    messages: list[MessageOut] = []


class AdminReplyRequest(BaseModel):
    reply_text: str
