from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime
