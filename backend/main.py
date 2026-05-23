from fastapi import FastAPI, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

app = FastAPI(title="AccessBank Support Agent")

bearer_scheme = HTTPBearer()


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message text.")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The agent's response text.")



@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> ChatResponse:
    pass
