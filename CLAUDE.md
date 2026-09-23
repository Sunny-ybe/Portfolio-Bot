# Portfolio-Bot

## What this is

A personal portfolio website with an embedded chatbot, built as a flagship learning project. AI (Claude) does the implementation; the user owns the architecture and wants to understand every component well enough to defend, modify, and tune it themselves — this is not a "just make it work" build.

## Core product requirements

- Every chatbot answer must be **evidence-based**, citing specific ingested artifacts (resume, PDFs, videos, links). If no evidence supports an answer, the bot says so — **no hallucination, ever**. This is enforced in code (a similarity-threshold gate before Claude is even called, plus post-hoc citation verification), not just prompted.
- **Evidence tiers are core product, not a nice-to-have**: every citation carries a trust tier — `public` (independently verifiable), `attested` (a third party vouches), `self_claimed_with_evidence` (asserted + backed by an artifact), `self_claimed_without_evidence` (bare assertion). Set manually at ingestion, never LLM-inferred. "Stronger citation, stronger product" is the framing.
- **Tiered visitor access** (Phase 3, not built yet): visitors verified as a GitHub/Twitter follower, a recruiter, or "an interesting engineer" get more information and can trigger a live ping to the owner. Verification will be OAuth-based (real API checks), not self-reported claims.

## Architecture

Two separate services, not one app:
- **`backend/`** — FastAPI (Python). Owns all RAG logic: chunking, embeddings, retrieval, the grounding gate, generation with citation enforcement. Chosen because the user already knows FastAPI/Python well and wants to deeply tune RAG behavior themselves — better to build the hard conceptual part in a language they're not also learning.
- **Frontend** (not built yet) — Next.js/TypeScript, currently being learned. Thin client: portfolio pages + a chat widget that calls the FastAPI `/chat` endpoint. No RAG logic lives here.

v1 deliberately has **no database, no vector DB, no auth**: the evidence store is a flat JSON file (`backend/data/evidence.json`, gitignored — generated data, not source), retrieval is plain cosine similarity in Python. This was a deliberate choice to understand RAG from first principles before reaching for infrastructure that would hide it. Move to Postgres+pgvector only once actually outgrown (see NOTES.md for the concrete trigger: live ingestion from a deployed server with an ephemeral filesystem).

## Phased plan

1. **Phase 1 (current)** — portfolio site + evidence-based bot, no auth/tiers. Backend RAG pipeline is built; **not yet run end-to-end** (no real API keys or ingested content yet — see Current Status).
2. **Phase 2** — richer ingestion (videos via TwelveLabs transcription + existing chunk/embed pipeline, PDFs, links), **plus live ingestion from the deployed site itself**: a protected `/admin/ingest` route sharing logic with `scripts/ingest.py`, gated by simple owner-only auth (password + TOTP 2FA) — a different, simpler problem than visitor-tiering OAuth.
3. **Phase 3** — OAuth (GitHub/Twitter via Auth.js), visitor identity tiers, privileged content, ping-the-owner notifications.

A "live in-browser code editor with GitHub push" idea was explicitly considered and dropped as scope creep (github.dev already does this; the security surface isn't worth it for this project).

## Current status (as of 2026-09-22)

**Backend RAG pipeline is fully built, end to end, but has never been run.** Files that exist:
- `backend/app/config.py` — centralized settings from env vars (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, CORS origins, `RAG_TOP_K`, `RAG_GROUNDING_THRESHOLD`)
- `backend/app/models.py` — Pydantic schemas: `ChatRequest`, `Citation` (incl. `evidence_tier`), `ChatResponse`
- `backend/app/rag/chunking.py` — fixed-size word-window chunking (800 words, 100 overlap), requires an `evidence_tier` per source
- `backend/app/rag/embeddings.py` — Voyage AI client (`voyage-3`), asymmetric query/document embedding
- `backend/app/rag/retrieval.py` — loads `evidence.json`, cosine-similarity top-K retrieval (linear scan)
- `backend/app/rag/generation.py` — forced Claude tool-use for schema-guaranteed citations + independent citation verification; ungrounded answers get a hardcoded refusal (never the model's own unverified text)
- `backend/app/main.py` — FastAPI app, `/health`, `/chat` (embeds → retrieves → grounding-threshold gate → generate)
- `backend/scripts/ingest.py` — CLI: chunk + tag + embed a text file into `evidence.json`, upserts by source

**Pipeline verified working end-to-end (2026-09-22):** `.env` filled with real Anthropic + Voyage keys (both billed/credited), resume ingested from `backend/sources/resume.txt` (1 chunk, tier `self_claimed_without_evidence`), server runs locally, `/chat` returns real Claude-generated, correctly-cited, grounded answers. Tested via `http://localhost:8001/docs` (FastAPI's interactive Swagger UI) and curl. **Note: run the server on port 8001, not 8000** — the user has an unrelated pre-existing project (`ambi`) that occupies port 8000 locally; don't touch that process.

`RAG_GROUNDING_THRESHOLD` was recalibrated from the initial guess of `0.5` down to `0.2`, based on real similarity scores (relevant questions scored ~0.21–0.33, irrelevant ones ~0.02–0.19 against the single resume chunk). This is still a rough starting point — with only one chunk, every question is scored against the entire resume at once, so the threshold will need re-tuning once there's more/varied content to actually discriminate between.

**Not done yet:**
- Next.js frontend — not started at all
- Admin auth + 2FA — designed conceptually (see NOTES.md), not built
- Phase 2/3 features — not built
- Error handling around upstream API failures (Claude/Voyage billing, rate limits, network) — currently crashes `/chat` with a raw 500 instead of a graceful response (see NOTES.md)
- Re-embedding-everything-every-ingest and hybrid search (BM25) — both flagged as future optimizations in NOTES.md, not built

**Immediate next step**: no longer blocked on setup — the backend works. Natural next steps: start the Next.js frontend, or keep hardening the backend (error handling, more evidence, threshold re-tuning with more content).

## Working conventions for this project

- **Explain before creating or materially changing any file**: why it's needed, what the alternatives are, what breaks without it. Get a go-ahead before writing.
- **Inline code comments** for anything non-obvious — a design reason, a gotcha, a non-default library behavior. Chat explanations aren't enough; the code should be understandable standalone later.
- **Flag scope creep honestly** when a new idea comes up mid-build — does an existing tool already solve it, what's the real cost — and default to deferring/dropping rather than building everything suggested.
- **No `Co-Authored-By: Claude` trailer on commits in this repo.** Keep commit messages short.
- Detailed design rationale and tradeoffs for every file (chunking strategy alternatives, embedding model choice, hybrid search/BM25 plan, contextual retrieval, evidence-tier data flow, etc.) live in **`NOTES.md`** at the repo root — read that for the "why," this file is for orientation and status.
