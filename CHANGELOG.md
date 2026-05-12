# Changelog

## [Unreleased]

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

[Unreleased]: https://github.com/willgitdata/augur/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/willgitdata/augur/releases/tag/v0.1.1
[0.1.0]: https://github.com/willgitdata/augur/releases/tag/v0.1.0
