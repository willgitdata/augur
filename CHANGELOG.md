# Changelog

## [Unreleased]

## [0.2.0] - 2026-05-14

Adoption + extensibility pass. No breaking changes to existing public API; new packages, new options, new adapters all additive.

### New packages

- `@augur-rag/langchain` — `searchAsLangchainDocs(augur, query)` returns LangChain-shaped `{ pageContent, metadata }`. Zero hard dep on `@langchain/core` (the shape is the contract).
- `@augur-rag/llamaindex` — `searchAsLlamaIndexNodes(augur, query)` returns `NodeWithScore`-shaped results.
- `@augur-rag/vercel-ai` — `augurToolDescriptor(augur, opts)` returns a Vercel AI SDK tool descriptor a language model can call.
- `@augur-rag/evaluations` — eval harness restored to in-tree (was previously a `git archive` step in the workflow). 504-query / 182-doc bundled corpus + metric unit tests now run as part of every PR's CI.

### New adapters

- `QdrantAdapter` — vector + optional sparse-dense hybrid via Qdrant's Query API (`prefetch` + `{ fusion: "rrf" }`). Named-vector schema; sparse encoder slot.
- `ChromaAdapter` — vector-only against Chroma's v2 REST API. Default tenant/database, bearer auth, paged `clear()`.
- `SqliteVecAdapter` — vector-only against SQLite + sqlite-vec for local-first / edge / desktop use. Includes `SqliteVecAdapter.migrate()`.

### Adapter improvements

- `PineconeAdapter` gains optional `sparseEncoder` — when supplied, capabilities flip to include hybrid and the adapter overrides `searchHybrid` to use Pinecone's native sparse-dense Query API (no client-side RRF).
- `PgVectorAdapter.migrate(client, opts)` — idempotent schema setup (extension + table + 3 indexes). Supports `vectorIndex: "ivfflat" | "hnsw"` and `ftsLanguage`.
- `BM25SparseEncoder` — pluggable BM25-weighted `SparseEncoder` for any backend that accepts `{ indices, values }` sparse vectors (Pinecone, Qdrant, Vespa, OpenSearch).

### Build + runtime

- Dual ESM + CJS build via `tsup`. `require("@augur-rag/core")` now works from CJS consumers; the existing ESM path is unchanged.
- Browser / edge bundle at `dist/browser/index.js`. `node:crypto` calls in `tracer.ts` and `contextual-chunker.ts` were refactored to Web Crypto so the bundle runs in browsers, Cloudflare Workers, Vercel Edge, Deno, and Bun.

### Embedder / reranker

- `LocalEmbedder` + `LocalReranker` gain an `onProgress` callback option that streams `@huggingface/transformers` download events. Replaces the silent ~5-10 second first-call wait with progress feedback when wired.
- `MissingTransformersError` — typed error thrown when the optional peer dep isn't installed. Names the install command in the message.
- `LocalReranker` reaches parity with `LocalEmbedder` on `dtype` + `device` constructor options. JSDoc on both classes now actively recommends `"fp16"` for production.

### Docs / contributor experience

- README hero now leads with the verb and pairs `npm install` with the hero snippet so first-run friction drops.
- "How Augur compares" table contrasting LangChain.js retrievers, LlamaIndex.ts retrievers, and raw vector-DB SDKs.
- CONTRIBUTING.md gains a 17-item good-first-issue list (adapters, bindings, chunkers, rerankers, observability bridges) plus a maintainer-only block with the `gh` CLI commands to enable Discussions and set repo topics.

### Bug-fix nits

- `InMemoryAdapter` docstring clarifies the IDF rebuild semantics (invalidate-on-write, rebuild on next read, average doc length recomputed eagerly).
- Server CLI's dynamic `import("pg")` no longer carries an `as Promise<any>` cast.
- `stripPunct` in `signals.ts` refactored to two anchored replaces (closes a `js/polynomial-redos` CodeQL pattern).

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
