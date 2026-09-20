import hashlib

from app.models import EvidenceTier


def chunk_text(
    text: str,
    source: str,
    evidence_tier: EvidenceTier,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)

        chunk_id = hashlib.sha1(f"{source}:{start}".encode()).hexdigest()[:12]
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_str,
                "source": source,
                "evidence_tier": evidence_tier,
            }
        )

        start += chunk_size - overlap

    return chunks
