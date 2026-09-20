
from typing import Literal

from pydantic import BaseModel

EvidenceTier = Literal[
    "public",
    "attested",
    "self_claimed_with_evidence",
    "self_claimed_without_evidence",
]


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    source: str
    text: str
    evidence_tier: EvidenceTier


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
