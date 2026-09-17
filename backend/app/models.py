
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    source: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
