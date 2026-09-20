from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import ChatRequest, ChatResponse
from app.rag.embeddings import embed_query
from app.rag.generation import UNGROUNDED_ANSWER, answer_from_chunks
from app.rag.retrieval import load_evidence, retrieve

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
    question_embedding = embed_query(request.question)
    matches = retrieve(question_embedding, evidence, top_k=settings.top_k)

    # The actual anti-hallucination gate. `matches[0]` is the single best match
    # (a (chunk, score) tuple, since retrieve() returns them sorted best-first),
    # and `[1]` pulls out its similarity score. If even the best match isn't
    # similar enough, we never call Claude at all — no chance for it to improvise
    # an answer from irrelevant chunks, and no wasted API call either.
    if not matches or matches[0][1] < settings.grounding_threshold:
        return ChatResponse(answer=UNGROUNDED_ANSWER, citations=[], grounded=False)

    return answer_from_chunks(request.question, matches)
