# Portfolio-Bot — architecture & learning notes

Running log of design decisions and the reasoning behind them, so tradeoffs discussed while building don't get lost. Newest additions at the bottom of each section.

## Overall shape

- Two services: `backend/` (FastAPI, owns all RAG logic — chunking, embeddings, retrieval, grounding, generation) and a Next.js frontend (thin client, calls the FastAPI `/chat` endpoint). Chosen because the RAG logic is what should be deeply understood/tunable, and Python/FastAPI is already known well, vs. learning RAG concepts and TypeScript syntax simultaneously.
- v1 has no database, no vector DB, no auth. Evidence store is a flat JSON file (`backend/data/evidence.json`), retrieval is plain cosine similarity in Python. Move to Postgres+pgvector only once this is actually outgrown.
- Phase 2: richer ingestion pipeline (videos/PDFs/links). Phase 3: OAuth (GitHub/Twitter), identity tiers, privileged content, ping-the-owner notifications.

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
