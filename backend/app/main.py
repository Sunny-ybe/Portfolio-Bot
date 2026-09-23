import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import ChatRequest, ChatResponse
from app.rag.embeddings import embed_query
from app.rag.generation import UNGROUNDED_ANSWER, answer_from_chunks
from app.rag.retrieval import load_evidence, retrieve

# logging (not print()) so every line gets a timestamp/severity level for free,
# and lands in the same log file we're already redirecting uvicorn's output to.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio Bot")

# Browsers block a frontend on one domain from calling an API on another domain
# unless the API explicitly allows it (CORS). This allow-list is what lets the
# Next.js frontend actually reach this backend from the browser. Without it,
# direct testing (curl, /docs) still works fine — only real browser requests fail.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level code runs once, at import time (i.e. once when the server starts),
# not per-request. Loading evidence here means every /chat request reuses the
# same in-memory list instead of re-reading evidence.json off disk each time.
evidence = load_evidence()


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(evidence)}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    logger.info("question: %r", request.question)

    question_embedding = embed_query(request.question)
    matches = retrieve(question_embedding, evidence, top_k=settings.top_k)

    best_score = matches[0][1] if matches else None
    logger.info("best retrieval score: %s (threshold: %s)", best_score, settings.grounding_threshold)

    # The actual anti-hallucination gate. `matches[0]` is the single best match
    # (a (chunk, score) tuple, since retrieve() returns them sorted best-first),
    # and `[1]` pulls out its similarity score. If even the best match isn't
    # similar enough, we never call Claude at all — no chance for it to improvise
    # an answer from irrelevant chunks, and no wasted API call either.
    if not matches or matches[0][1] < settings.grounding_threshold:
        logger.info("below threshold — skipping Claude call, returning ungrounded response")
        return ChatResponse(answer=UNGROUNDED_ANSWER, citations=[], grounded=False)

    result = answer_from_chunks(request.question, matches)
    # %r (not %s) escapes newlines as literal "\n" so a multi-line answer still
    # prints as exactly one line in the log file — keeps each request grep-able
    # as a single entry instead of spilling across several lines.
    logger.info("grounded=%s, citations=%d, answer=%r", result.grounded, len(result.citations), result.answer)
    return result
