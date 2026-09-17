import json
from pathlib import Path

import numpy as np

EVIDENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "evidence.json"


def load_evidence() -> list[dict]:
    if not EVIDENCE_PATH.exists():
        return []
    with open(EVIDENCE_PATH) as f:
        return json.load(f)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(question_embedding: list[float], evidence: list[dict], top_k: int = 4) -> list[tuple[dict, float]]:
    scored = [(chunk, cosine_similarity(question_embedding, chunk["embedding"])) for chunk in evidence]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
