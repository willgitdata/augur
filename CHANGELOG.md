# Changelog

All notable changes to Augur are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/) once we hit 1.0;
prior to 1.0 (`0.x.y`) breaking changes can land on minor bumps and will
be called out under **BREAKING** when they do.

## [Unreleased]

_Nothing yet._

## [0.1.0] - 2026-05-10

Initial public release. Two publishable packages: `@augur-rag/core` (the
SDK) and `@augur-rag/server` (an optional Fastify HTTP wrapper). MIT
licensed. Node ≥ 20. Zero runtime dependencies in core.

### SDK surface

- `Augur` orchestrator with constructor-injected components and a
  required `embedder`. Optional: `adapter` (defaults to
  `InMemoryAdapter`), `chunker` (defaults to `SentenceChunker`),
  `router` (defaults to `HeuristicRouter`), `reranker` (default `null`
  — bare retrieval; pass an explicit reranker to enable the
  cross-encoder stage), `traceStore`, `autoIndexAdHocDocuments`,
  `adHocCacheSize`, `autoLanguageFilter`.
- `augr.index(documents)` — chunks, embeds, upserts. Returns a timing
  breakdown.
- `augr.search(request)` — runs the routing + retrieval pipeline and
  returns `{ results, trace }`. Per-call options: `topK`,
  `forceStrategy`, `latencyBudgetMs`, `filter`, `context`,
  `documents` (ad-hoc inline corpus, LRU-cached by content
  fingerprint), `minScore` (confidence floor — drops results below
  the threshold for "no answer" downstream).

### Routing

- `Router` interface (`name`, `decide(req, caps, hasReranker?)`) so
  third-party routers slot in without changes to the orchestrator.
- `HeuristicRouter` — rule-based per-query strategy selection across
  vector / keyword / hybrid / rerank, driven by query signals and
  adapter capabilities. Records human-readable `reasons` for every
  decision. `alwaysRerank: true` by default; pass
  `new HeuristicRouter({ alwaysRerank: false })` for the latency-
  conscious BM25 fast path.
- `computeSignals(query)` — pure feature extraction (word count,
  question type, named entities, code-like syntax, dates/versions,
  negation, BCP-47 language code, ambiguity heuristic).
- BCP-47-style language detection from Unicode-script analysis: `en`,
  `ja`, `zh`, `ko`, `ru`, `ar`, `hi`, `th`, `he`, `el`. Drives the
  router's non-English branch and the optional auto-language filter.

### Retrieval pipeline

- Multi-stage gather → fuse → rerank — the production pattern from
  Turbopuffer / Vespa / Cohere Rerank. Pulls 50 from each retriever in
  parallel, RRF-fuses with adaptive weights (query-signal prior shifted
  by observed retrieval confidence), hands top 30 to the reranker.
- Adaptive weighted RRF in `fusion.ts` — exposed as pure helpers
  (`pickVectorWeight`, `weightedRrfFuse`, `adaptWeightByConfidence`)
  for testability.
- Per-chunk `metadata.lang` auto-tagging at index time, plus an
  optional `autoLanguageFilter` that pins non-English queries to
  same-language chunks with a soft fallback if the filter would empty
  the candidate pool.

### Chunkers

- `Chunker` (sync) and `AsyncChunker` (async) interfaces — separate
  contracts so async dependencies are caught at compile time, not
  runtime. `chunkDocument()` is the polymorphic entry point.
- `FixedSizeChunker`, `SentenceChunker` — sync, zero deps.
- `SemanticChunker` — embedding-driven sentence grouping (async).
- `MetadataChunker` — wraps a base chunker and prepends
  `[doc-id | topic | title]` so structured signals are searchable.
  "Doc2Query lite."
- `Doc2QueryChunker` — generates synthetic questions per chunk via a
  small T5 model (`Xenova/LaMini-T5-61M` by default) and appends them
  to chunk content before embedding.
- `ContextualChunker` — implementation of Anthropic's
  [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
  pattern. Pluggable `ContextProvider` interface (one method) wires any
  LLM provider. Includes the published prompt template
  (`ANTHROPIC_CONTEXTUAL_PROMPT`) and a content-hash-keyed
  `MemoryContextCache` so re-indexing unchanged content is free.

### Embedders

- `Embedder` interface — three required methods (`name`, `dimension`,
  `embed`), optional `fit` / `embedDocuments` / `embedQuery` for
  embedders that distinguish doc vs. query roles (Cohere v3, Gemini,
  BGE/E5 prefix-based).
- `LocalEmbedder` — on-device sentence-transformers via
  `@huggingface/transformers` (ONNX Runtime). Default
  `Xenova/all-MiniLM-L6-v2` (~22 MB, 384d). Configurable model,
  per-task prefixes, batch size, ONNX quantization (`fp32`/`fp16`/
  `q8`/`q4`), and device (`wasm`/`webgpu`/`cpu`).
- `tokenize` / `tokenizeAdvanced` (Porter stemming + stopword
  filtering) / `STOPWORDS` exported for use in custom embedders and
  rerankers.

### Rerankers

- `Reranker` interface — one method (`rerank`).
- `LocalReranker` — on-device cross-encoder via
  `@huggingface/transformers`. Default
  `Xenova/ms-marco-MiniLM-L-6-v2` (~22 MB). Calibrated [0, 1] sigmoid
  scores by default.
- `HeuristicReranker` — token-overlap + proximity baseline. Useful for
  smoke tests; pair with `LocalReranker` (or any hosted cross-encoder)
  for production.
- `MMRReranker` — Maximal Marginal Relevance for diverse top-K.
- `CascadedReranker` — chains stages (cheap-broad → expensive-narrow).

### Adapters

- `VectorAdapter` interface with explicit `capabilities` block
  (`{ vector, keyword, hybrid, computesEmbeddings, filtering }`) so the
  router can degrade gracefully against backends with partial support.
- `BaseAdapter` — provides default RRF-based `searchHybrid` over your
  `searchVector` and `searchKeyword` implementations.
- `InMemoryAdapter` — zero-dep, BM25 + brute-force vector + RRF
  hybrid. Optional Porter stemming + stopword filtering. Phrase-
  substring boost for quoted phrases and short identifier-like
  queries.
- `PineconeAdapter` — vector-only via Pinecone's REST API. SDK-free
  fetch implementation.
- `TurbopufferAdapter` — native vector + BM25 + hybrid via
  Turbopuffer's REST API. SDK-free.
- `PgVectorAdapter` — Postgres + `vector` extension. Vector via
  `<=>` cosine, keyword via `tsvector` + `plainto_tsquery`, hybrid
  via inherited RRF. Strict identifier validation on the table name
  and on every metadata filter key (defends against SQL injection
  via the JSON-path operands that Postgres can't parameter-bind).
- Custom adapters are five methods — `examples/custom-adapter` is a
  ~90-line worked example.

### Observability

- `Tracer` — first-class trace as API output, not a fire-and-forget
  side effect. Every `search()` returns its `SearchTrace` in the
  response.
- `SearchTrace` — typed fields for `id`, `query`, `decision`
  (strategy + reasons + signals + reranked), `spans`, `candidates`,
  `adapter`, `embeddingModel`, plus `adHoc`, `adHocCacheHit`,
  `autoLanguageFilter`, `autoLanguageFilterDropped` for the optional
  features above.
- `TraceStore` — bounded ring buffer (default 2000) for the HTTP
  server's `/traces` endpoint and trace-explorer integrations.

### `@augur-rag/server`

- Fastify wrapper around `@augur-rag/core` with `POST /search`,
  `POST /index`, `GET /traces`, `GET /traces/:id`, `DELETE /traces`,
  `GET /health`, `GET /openapi.json`, `GET /docs` (Swagger UI),
  `GET /admin/stats`, `POST /admin/clear`.
- Hand-rolled OpenAPI 3.1 spec at `/openapi.json`.
- Optional shared-secret auth via `AUGUR_API_KEY`.
- CLI entry `augur-server` reads adapter / model / API-key config
  from environment variables.

### Performance

CI enforces `NDCG@10 > 0.65` on a 16-doc / 12-query synthetic corpus
(`packages/core/src/eval-smoke.test.ts`) — a structural regression net
that catches pipeline breakage on every PR.

The recommended local stack is **44 MB on-device total**
(`Xenova/all-MiniLM-L6-v2` 22 MB embedder + `Xenova/ms-marco-MiniLM-L-6-v2`
22 MB cross-encoder), no network at query time, no API keys.

**BEIR — measured by the `Eval matrix` workflow** (Actions tab,
`workflow_dispatch` with `target=beir-only`, `auto_stack=default`,
`run_fiqa=true`). Auto stack = MiniLM-L6 + ms-marco cross-encoder +
MetadataChunker + stemmed BM25, no per-corpus tuning:

| Dataset    |    Auto |  BM25 | BM25+rerank | Contriever | ColBERTv2 | BGE-large (1.3GB) | E5-large (1.3GB) |
| ---------- | ------: | ----: | ----------: | ---------: | --------: | ----------------: | ---------------: |
| SciFact    |   0.707 | 0.665 |       0.688 |      0.677 |     0.694 |             0.745 |            0.736 |
| FiQA       |   0.345 | 0.236 |       0.347 |      0.329 |     0.356 |             0.450 |            0.424 |
| NFCorpus   |   0.324 | 0.325 |       0.350 |      0.328 |     0.339 |             0.380 |            0.371 |

Auto numbers are this workflow's measurements; baseline columns are
the published numbers from the BEIR (Thakur et al. 2021), BGE
(Xiao et al. 2023), E5 (Wang et al. 2022), and ColBERTv2 papers
(static, not re-measured here).

A 504-query / 182-doc development eval drove the routing constants
and fusion weights during development. That harness lives at git
commit `feffc73^` and is slated for republish as a standalone
`augur-eval` sister repo. The `Eval matrix` workflow restores it on
demand to produce the table above.

[Unreleased]: https://github.com/willgitdata/augur/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/willgitdata/augur/releases/tag/v0.1.0
