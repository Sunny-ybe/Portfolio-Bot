"""
Chunk, tag, and embed a text file into the evidence store (data/evidence.json).

Usage:
    python scripts/ingest.py <file_path> <evidence_tier>

Example:
    python scripts/ingest.py resume.txt self_claimed_with_evidence
"""

import argparse
import json
import sys
from pathlib import Path
from typing import get_args

# scripts/ is a sibling of app/, not inside it — Python only auto-adds this
# script's own folder to its import search path, not backend/. Without this
# line, `from app.models import ...` below would fail with
# "ModuleNotFoundError: No module named 'app'".
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.models import EvidenceTier  # noqa: E402 (must come after sys.path.append)
from app.rag.chunking import chunk_text  # noqa: E402
from app.rag.embeddings import embed_texts  # noqa: E402

EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "evidence.json"

# Pulled from the EvidenceTier Literal in models.py rather than retyped here,
# so there's exactly one place the four valid tier strings are defined.
ALLOWED_TIERS = get_args(EvidenceTier)


def main(file_path: str, evidence_tier: str) -> None:
    source = Path(file_path).name  # e.g. "resume.txt" — tags every chunk and is the de-dup key below
    text = Path(file_path).read_text()

    chunks = chunk_text(text, source=source, evidence_tier=evidence_tier)

    # One batched API call for every chunk from this file, not one call per
    # chunk — cheaper and much faster than embedding chunks individually.
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    existing = json.loads(EVIDENCE_PATH.read_text()) if EVIDENCE_PATH.exists() else []

    # Upsert by source: drop any chunks from a previous ingestion of this same
    # file before adding the fresh ones. Without this, re-running ingestion
    # after editing resume.txt would leave stale duplicate chunks behind
    # instead of replacing them.
    existing = [chunk for chunk in existing if chunk["source"] != source]
    existing.extend(chunks)

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(existing, indent=2))

    print(
        f"Ingested {len(chunks)} chunks from {source} (tier: {evidence_tier}). "
        f"Total chunks in store: {len(existing)}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_path", help="Path to the text file to ingest")
    parser.add_argument("evidence_tier", choices=ALLOWED_TIERS, help="Trust tier for this source")
    args = parser.parse_args()

    main(args.file_path, args.evidence_tier)
