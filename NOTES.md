# Portfolio-Bot — architecture & learning notes

Running log of design decisions and the reasoning behind them, so tradeoffs discussed while building don't get lost. Newest additions at the bottom of each section.

## Overall shape

- Two services: `backend/` (FastAPI, owns all RAG logic — chunking, embeddings, retrieval, grounding, generation) and a Next.js frontend (thin client, calls the FastAPI `/chat` endpoint). Chosen because the RAG logic is what should be deeply understood/tunable, and Python/FastAPI is already known well, vs. learning RAG concepts and TypeScript syntax simultaneously.
- v1 has no database, no vector DB, no auth. Evidence store is a flat JSON file (`backend/data/evidence.json`), retrieval is plain cosine similarity in Python. Move to Postgres+pgvector only once this is actually outgrown.
- Phase 2: richer ingestion pipeline (videos/PDFs/links). Phase 3: OAuth (GitHub/Twitter), identity tiers, privileged content, ping-the-owner notifications.

### Phase 2 addendum: live ingestion from the deployed website, not just local CLI

Ingesting from the site itself (not just running `scripts/ingest.py` by hand) means:
- A protected `/admin/ingest` API route, sharing the same core chunk/tag/embed logic as `scripts/ingest.py` (pulled into a shared function, e.g. `app/rag/ingestion.py`) rather than duplicating it.
- Its own, simpler owner-only auth (a single admin token/password), distinct from Phase 3's visitor-tiering OAuth — "is this me, the owner" is a different, easier problem than "who is this visitor."
- **Likely trigger to move off flat-file `evidence.json` sooner than chunk-count alone would suggest**: most hosting platforms (Railway, Fly.io) run an ephemeral filesystem — a write to local disk from the live server can be wiped on the next redeploy/restart unless a persistent volume is provisioned. Live ingestion from the deployed site is the point this actually bites; local-only CLI ingestion (current v1) never hits it since the disk is genuinely yours.

## `config.py`

- Centralizes all settings/secrets so the rest of the app never touches `os.environ` directly. Alternative (skip it, call `os.environ.get(...)` inline everywhere) works for tiny projects but scatters env var names and breaks silently on typos as the project grows.
- Required settings (`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`) use `os.environ["KEY"]` (bracket access) — fails loudly at startup if missing. Optional/tunable settings (`RAG_TOP_K`, `RAG_GROUNDING_THRESHOLD`) use `.get(key, default)`.
- Plain Python class instead of `pydantic-settings` — fewer concepts to learn right now; can swap in later, it's a contained change.
- `ALLOWED_ORIGINS` — CORS allow-list. Browsers block a frontend on one domain from calling an API on another domain unless the API explicitly allows it. Without this, direct API testing (curl/Postman/FastAPI's `/docs`) works fine but the actual browser-based chat widget silently fails with a CORS error. Using `["*"]` (allow anyone) is simpler but risks random sites burning your Anthropic/Voyage quota by calling your endpoint — restrict to your real frontend domain(s) instead.
- **Grounding threshold**: minimum cosine similarity (roughly 0–1) the best-matching chunk must hit before we trust we have relevant evidence. Below threshold → return "no evidence" *without calling Claude at all* (this is the actual anti-hallucination gate, enforced in code, not prompting). No principled default — 0.5 is a starting guess, meant to be tuned empirically once real questions are logged.

## `models.py`

- Uses Pydantic (`BaseModel`), the FastAPI-native choice: automatic request validation, auto-generated `/docs`, editor autocomplete. Plain dicts would skip all of that and fail with confusing errors on bad input instead of a clear 422 at the door.
- Extending later: add fields with a default (e.g. `confidence: float | None = None`) — fully backward compatible, nothing breaks. Adding a *required* field forces you to update every place that constructs the model, but Python/the editor will point at exactly those spots.

## `chunking.py` — fixed-size word windowing (v1 choice)

- v1: split text into N-word windows (800 words, 100 overlap ≈ 12.5%), slide forward by `N - overlap`. Zero dependencies, deterministic.
- Chunk ID = `sha1(f"{source}:{start}")[:12]` — deterministic, so re-chunking the same file with the same params reproduces the same IDs. Matters because citations reference chunk IDs; you don't want them to churn on every re-ingest. **Known gap**: `ingest.py` currently *appends* rather than upserting by source, so re-running ingestion on an edited file duplicates entries — fine for a single one-shot resume ingest, needs fixing before Phase 2's repeatable pipeline.
- Word count is a rough proxy for token count (~0.75 tokens/word for English) — imprecise, doesn't matter yet at resume scale, matters more once packing long video transcripts tightly into context.

### Chunking alternatives (for later)

- **RecursiveCharacterTextSplitter** (LangChain's technique, general concept): tries separators in priority order (`\n\n` → `\n` → `. ` → ` ` → raw char), recursively re-splitting any piece still over the size limit using the next separator down. Then merges adjacent small pieces back up to the size limit, carrying overlap forward. Cuts at the most "natural" available boundary instead of blindly at word N. Still purely structural — no understanding of meaning.
- **Proposition-based chunking**: LLM decomposes text into atomic, self-contained factual statements. High retrieval precision, one LLM call per document at ingestion, more total chunks.
- **Instructed/agentic chunking**: prompt an LLM directly to split a document into topic-coherent chunks (JSON array out). Understands content, not just punctuation.
- **Semantic-similarity chunking**: embed each sentence, cut where cosine distance between consecutive sentence embeddings spikes. Uses embeddings you're already paying for, no generation call needed.
- **Contextual retrieval** (Anthropic's technique): doesn't change chunk boundaries — prepends a short LLM-generated sentence describing where the chunk sits in the document before embedding it (e.g. "This chunk is from the Experience section, describing the Company X role"). Cheap, high-leverage accuracy lever, addable later without restructuring the chunker.

## Evidence tiers — core product feature, not a nice-to-have

Every chunk carries an `evidence_tier`, set manually at ingestion time (never inferred by an LLM, since it's the trust signal the whole product rests on):
- `public` — independently verifiable by anyone (public repo, published article/video)
- `attested` — a third party vouches for it (reference, recommendation, co-author confirmation)
- `self_claimed_with_evidence` — you assert it, backed by an attached artifact (certificate, screenshot, linked code)
- `self_claimed_without_evidence` — bare assertion, nothing backing it

Defined once as an `EvidenceTier` Literal type in `models.py` (the data-shapes file) and imported into `chunking.py`, rather than duplicating the four strings in both places — avoids drift if a tier is ever renamed/added. Flows purely as passthrough data: `chunking.py` stamps it on each chunk → `evidence.json` stores it → `retrieval.py` doesn't need to know about it (pure passthrough) → `Citation` model carries it → `generation.py` surfaces it per citation → frontend (later) renders a trust badge per citation. Also feeds the system prompt later so the model can hedge phrasing appropriately per tier — a prompting refinement layered on top of, not a replacement for, the structural metadata.

## `embeddings.py`

- Voyage AI chosen over local `sentence-transformers`: Anthropic's recommended embeddings partner (Claude has no embeddings API of its own), avoids pulling in ~500MB of local ML deps.
- `input_type="document"` vs `"query"` is not cosmetic — Voyage's models are trained asymmetrically, applying different internal instruction-prefixes depending on role, so queries and documents land correctly in a shared space. Mismatching these doesn't error, it silently degrades similarity scores.
- Client (`_client = voyageai.Client(...)`) is created once at module import time (server startup), reused for the life of the process — avoids repeating connection/auth setup on every request. See module-level-execution note above.
- `embed_texts` (batch) vs `embed_query` (single) split: ingestion embeds a whole document's chunks in one API call; a chat request only ever embeds one question. Voyage caps batch size/tokens per call — not an issue at resume scale, will need sub-batching once ingesting long transcripts.
- Model: `voyage-3` (general-purpose default). `voyage-3-lite` = cheaper/faster/lower quality. `voyage-3-large` = higher quality/cost. Domain models exist (`voyage-code-3`, etc.) — revisit only if corpus becomes heavily code-centric.

## `retrieval.py`

- `load_evidence()` reads `evidence.json` from disk; called once at server startup (kept in memory), not re-read per request.
- `cosine_similarity` uses `numpy` (dot product / product of norms) — measures angle between vectors, ignoring magnitude, since magnitude isn't semantically meaningful. Numpy chosen over a pure-Python loop for speed (C-implemented ops).
- `retrieve()` is a linear scan — scores every chunk, sorts, takes top-K. `O(n)` per query, fine at resume-scale (tens–hundreds of chunks). At real scale, swap in an ANN index (HNSW/IVF — what FAISS/pgvector/Pinecone use internally) for sub-linear lookup; same function signature, contained upgrade later.
- Returns `(chunk, score)` tuples rather than mutating chunk dicts in place — keeps evidence data untouched, avoids parallel lists drifting out of sync.

### Enterprise-scale / metadata-aware retrieval (beyond v1, good to know)

Three composable techniques, not one:

1. **Metadata as filters (pre-filtering)** — real vector stores (Pinecone, Weaviate, pgvector via SQL `WHERE`) combine vector similarity with structured filters: `WHERE source_type = 'slack' AND author = 'alice' AND timestamp > X`. Needed because raw semantic similarity across a huge heterogeneous corpus (Slack + Gmail + docs + website) gets noisy — a throwaway chat message and an authoritative doc on the same topic can embed nearly identically despite very different trust/recency value.
2. **Metadata folded into the embedded text itself** — generalized contextual retrieval: prepend a synthesized string ("Slack message from Alice in #eng-infra, 2024-03-01, replying to a thread about deploy failures:") before embedding. Shifts the vector itself to encode source/time/social context, helping even when a query doesn't explicitly specify filters. Complementary to (1), not a replacement.
3. **Recency decay + re-ranking as separate scoring stages** — standard production shape: vector search pulls a broad cheap candidate set (e.g. top 50) → a decay function down-weights stale content (`score = cosine_sim * exp(-λ · age_days)`, tunable per source) → a re-ranker (cross-encoder that scores query+candidate pairs jointly — Voyage has `voyage-rerank-2`) re-scores down to the true top-K sent to the LLM. Access control (does this user have permission to see this chunk's source) gets enforced as another filter at this stage, before the LLM ever sees the content.

For v1 (single-person, mostly resume text): none of this needed yet, but `evidence.json`'s chunk schema should eventually carry a `metadata: {source_type, timestamp, author, url}` dict per chunk so Phase 2 (multiple source types) doesn't require a rewrite — currently deferred, not yet added.

## `generation.py`

- **Structured output via forced tool use, not prompted JSON.** Prompting for JSON in free text is unreliable (markdown fences, preambles can break `json.loads`). Forcing a tool call (`tool_choice={"type": "tool", "name": "submit_answer"}`) constrains generation itself — Anthropic's serving infrastructure masks out any token that would violate the schema at each generation step (constrained/grammar decoding), so the model is mechanically incapable of responding outside the schema shape. **Enforcement happens server-side, on Anthropic's infrastructure** — your code only defines the schema and requests it, then receives an already-parsed dict back.
- This only guarantees *shape*, not *truth* — the model could still put a fabricated `chunk_id` inside a well-formed `citations` list. Hence the second, independent layer: filtering `parsed["citations"]` down to only IDs present in `chunk_by_id` (the chunks actually sent). Two separate defenses: tool-forcing solves validity, the filter solves truthfulness.
- **Ungrounded-answer policy: option (b) chosen.** If zero citations survive verification, the model's own `answer` text is discarded entirely and replaced with a fixed `UNGROUNDED_ANSWER` string — never trust free-text claims in a state that couldn't be verified, even if the model's own prose happens to say the honest thing. This doesn't rely on the model reliably self-reporting "no evidence" — it's enforced by code regardless of what text comes back.
- Evidence block passed to the model includes each chunk's `evidence_tier` inline (`tier: {chunk['evidence_tier']}`) so the model can hedge phrasing per tier later — a prompting refinement on top of the structural metadata already carried through `Citation`.

## Request/response logging

Added `logging` (not `print()`) to `main.py`'s `/chat` handler — logs the incoming question, the best retrieval score vs. the grounding threshold, whether the threshold gate passed, and (when it does call Claude) the final `grounded`/citation count. Lands in the same log file uvicorn's own output is already redirected to (`/tmp/portfolio-bot-uvicorn.log` locally). `logging` chosen over `print()` for free timestamps/severity levels and because it's the standard idiomatic choice, not a one-off script habit. This gives ongoing visibility into exactly the kind of thing the manual threshold-calibration script dug up earlier — as automatic, always-on output instead of an ad hoc check.

Also running the dev server with `uvicorn app.main:app --port 8001 --reload` now (not just `--port 8001`) — `--reload` watches source files and restarts automatically on save, which is what caught us out earlier (a stale, pre-`--reload` server process kept serving an old `generation.py` for 15 minutes after edits, which looked like "instructions are being ignored" but was actually "the edits were never loaded at all").

## `main.py`

- The actual anti-hallucination gate lives here, not in `generation.py`: `/chat` always embeds + retrieves first (cheap), then checks `matches[0][1]` (best match's similarity score) against `settings.grounding_threshold` *before* calling `answer_from_chunks` (which calls Claude). Below threshold → short-circuit to `UNGROUNDED_ANSWER` without ever calling Claude — saves an API call and removes any window where an irrelevant chunk could reach the model.
- `evidence = load_evidence()` at module level — loaded once at server startup, reused across every request, not re-read from disk per request.
- `/health` — standard liveness endpoint for deployment platforms/load balancers; also useful to manually confirm `evidence.json` loaded correctly (`chunks_loaded` count).
- **From this file onward, inline comments explain non-obvious logic directly in the code**, not just here in NOTES.md — the user asked for this as a standing rule (2026-09-17) since the goal is being able to open any file later and understand it without recalling the chat.

### Re-embedding on every ingest — future optimization, not built yet

`ingest.py` currently re-chunks and re-embeds a file's *entire* content on every run, with no check for "have I already embedded this exact content." Fine at resume-scale (cheap, instant); becomes real waste once ingesting long video transcripts or re-running ingestion frequently during iteration.

Fix: hash file content before chunking, compare against the hash from that source's last ingestion (stored alongside chunks or in a small manifest), skip re-embedding if unchanged. A chunk-level (not whole-file) version of this would need `chunk_id` to become content-based (`sha1(chunk_text)`) rather than position-based (`sha1(f"{source}:{start}")`) as it is now — position-based IDs shift for every chunk after an edit even when most content is unchanged, so they can't cleanly detect "this specific chunk is unchanged."

### Error handling for upstream API failures — future work, not built yet

`main.py`'s `/chat` currently has no try/except around the Claude or Voyage calls — a billing issue, rate limit, or network blip on either upstream API crashes the request with a raw 500 error instead of a graceful degraded response. Fine for local testing; worth wrapping before this is live for real visitors, so an upstream outage shows a sensible "temporarily unavailable" message instead of a stack trace.

### Hybrid search (BM25 + dense) — future extension, not built yet

- BM25 = lexical/sparse retrieval (TF-IDF's successor): scores chunks by exact keyword overlap, weighted by term rarity, normalized for length. Complementary to dense embeddings — embeddings catch semantic/paraphrase matches but can under-weight exact tokens that matter (proper nouns, company names, acronyms, numbers). A query like "when did you work at Acme Corp" is where BM25 nails the literal "Acme Corp" token even if the embedding model doesn't weight that proper noun strongly.
- Where it plugs in: additive, not a rewrite of `retrieval.py`. New module (e.g. `rag/lexical.py`) with a `bm25_score(query, evidence)` function (likely via `rank_bm25`), index built once at startup — same "compute once, reuse" pattern as the Voyage client and `load_evidence()`.
- Combine dense + sparse rankings via **Reciprocal Rank Fusion (RRF)**: for each chunk, sum `1 / (k + rank)` across its rank position in both separately-sorted lists (k ≈ 60 is a common constant), sort by that combined score. RRF avoids needing to normalize two incomparable score scales (cosine similarity and BM25 scores aren't on the same range).
- `cosine_similarity` and `retrieve()` stay unchanged when this is added later — hybrid search bolts on alongside them via a new `hybrid_retrieve()`, not a rewrite.
