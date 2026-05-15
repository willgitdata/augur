# Changelog

## [Unreleased]

## [0.2.0] - 2026-05-14

- New adapters: `QdrantAdapter` (vector + sparse-dense hybrid via Qdrant's Query API), `ChromaAdapter`, `SqliteVecAdapter` (with `SqliteVecAdapter.migrate()` for local-first / edge / desktop).
- `PineconeAdapter` gains an optional `sparseEncoder` slot — flips `capabilities.hybrid` to true and overrides `searchHybrid` to call Pinecone's native sparse-dense Query API.
- `BM25SparseEncoder` — pluggable BM25-weighted `SparseEncoder` for any `{ indices, values }` backend.
- `PgVectorAdapter.migrate(client, opts)` — idempotent schema setup; supports `vectorIndex: "ivfflat" | "hnsw"` and `ftsLanguage`.
- Framework integrations under `@augur-rag/core/integrations/{langchain,llamaindex,vercel-ai}` — three small adapter functions that emit each framework's expected shape without taking a hard dep on the framework itself.
- Dual ESM + CJS build via `tsup`, plus a browser / edge bundle at `dist/browser/index.js` (refactored `node:crypto` to Web Crypto).
- `LocalEmbedder` + `LocalReranker`: new `onProgress` callback for HuggingFace download events; typed `MissingTransformersError` when the optional peer is missing; `LocalReranker` gains `dtype` + `device` parity with `LocalEmbedder` (recommend `"fp16"` in JSDoc).
- Eval harness restored in-tree as `@augur-rag/evaluations` (private workspace package); metric unit tests now run in every PR's CI.
- README: install line next to hero snippet; CONTRIBUTING gains a good-first-issue list and the `gh` commands for repo-side settings.
- Bug nits: `InMemoryAdapter` IDF docstring corrected, `import("pg")` cast dropped from server CLI, `signals.ts` regexes bounded to close CodeQL `js/polynomial-redos` patterns.

## [0.1.1] - 2026-05-11

Republish of 0.1.0. The 0.1.0 tarball shipped with an empty `dist/` because of a `prepublishOnly` cache issue and was deprecated on npm. 0.1.1 contains the same feature set, packaged correctly.

## [0.1.0] - 2026-05-10

First public release.

- `Augur` orchestrator with per-query routing across vector / keyword / hybrid / rerank
- `HeuristicRouter` driven by query signals (quoted phrase, code-like syntax, named entity, question type, language)
- Adapters: `InMemoryAdapter`, `PgVectorAdapter`, `PineconeAdapter`, `TurbopufferAdapter`
- Chunkers: `FixedSizeChunker`, `SentenceChunker`, `SemanticChunker`, `MetadataChunker`, `Doc2QueryChunker`, `ContextualChunker`
- Embedders: `Embedder` interface + `LocalEmbedder` (on-device ONNX, `Xenova/all-MiniLM-L6-v2` default)
- Rerankers: `LocalReranker`, `HeuristicReranker`, `MMRReranker`, `CascadedReranker`
- `Tracer` + `TraceStore` (bounded ring buffer); every `search()` returns a typed `SearchTrace`
- `@augur-rag/server`: optional Fastify wrapper, OpenAPI 3.1 spec, Swagger UI at `/docs`

[Unreleased]: https://github.com/willgitdata/augur/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/willgitdata/augur/releases/tag/v0.2.0
[0.1.1]: https://github.com/willgitdata/augur/releases/tag/v0.1.1
[0.1.0]: https://github.com/willgitdata/augur/releases/tag/v0.1.0
