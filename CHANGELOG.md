# Changelog

All notable changes to Augur are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/) once we hit 1.0;
prior to 1.0 (`0.x.y`) breaking changes can land on minor bumps and are
called out under **BREAKING** in the entry.

## [Unreleased]

### Added

- `LocalEmbedder` `dtype` and `device` options for ONNX quantization (fp32 / fp16 / q8 / q4) and runtime selection.
- `ContextualChunker` — implementation of Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) pattern, including the published prompt template (`ANTHROPIC_CONTEXTUAL_PROMPT`).
- `Augur` constructor option `autoLanguageFilter` (default `false`) — when enabled, non-English queries are filtered to chunks tagged with the matching language, with a soft fallback if the filter empties the candidate pool.
- Granular language detection: `signals.language` is now a BCP-47-style code (`en`, `ja`, `zh`, `ko`, `ru`, `ar`, `hi`, `th`, `he`, `el`) instead of binary `"en" | "non-en"`.
- Per-chunk `metadata.lang` auto-tagging at index time.
- `SearchRequest.minScore` — drops results below a confidence floor (useful for "no answer" signals to downstream LLMs).

### Changed

- **BREAKING** `signals.tokens` → `signals.wordCount`, `signals.avgTokenLen` → `signals.avgWordLen`. The previous names were misleading — these are whitespace-split words, not embedding subword tokens.
- **BREAKING** `signals.language` type widened from `"en" | "non-en"` to `string` (specific language code).
- **BREAKING** `HeuristicRouter` defaults `alwaysRerank: true`. The cross-encoder votes on every query out of the box for best NDCG@10. Latency-sensitive callers opt out via `new HeuristicRouter({ alwaysRerank: false })`.
- **BREAKING** `Augur` constructor requires `embedder`. The placeholder `HashEmbedder` / `TfIdfEmbedder` are removed — they produced near-random vectors that surfaced as product bugs. Use `LocalEmbedder` (zero-config, on-device ONNX) or implement the 3-method `Embedder` interface.
- **BREAKING** `Augur` no longer defaults to `HeuristicReranker`. Default is now `null` — bare retrieval if `reranker` is omitted. The previous default did almost nothing and gave fake "yes I rerank" comfort in traces; pass `new LocalReranker()` (or any provider's reranker) explicitly to keep cross-encoder voting on. See [MIGRATING.md](MIGRATING.md).
- Ad-hoc search with inline `documents` now caches the scratch adapter (LRU, fingerprint-keyed by id+content). Repeat searches over the same documents skip re-chunking + re-embedding. Tunable via `adHocCacheSize` (default 8; set 0 to disable).
- `gatherCandidatePool` now uses adaptive weighted RRF (query-signal prior + retrieval-confidence shift) instead of symmetric RRF.
- Multi-stage retrieval pipeline (gather → fuse → rerank) — production pattern from Turbopuffer / Vespa / Cohere Rerank.
- HeuristicRouter rule split: code-like syntax stays on keyword; bare identifier / numeric / date-version queries route to hybrid for topical recall.
- `InMemoryAdapter.searchKeyword` — phrase-substring boost for quoted phrases and short identifier-like queries.

### Removed

- `HashEmbedder`, `TfIdfEmbedder` — placeholder embedders no longer ship; `LocalEmbedder` is the only built-in.
- Hosted-provider embedder/reranker classes from core (`GeminiEmbedder`, etc.) — moved to user-implemented `Embedder` / `Reranker` instances per the EXAMPLES.md §5 pattern.
- `apps/dashboard` (Next.js trace explorer) and `evaluations/` (BEIR runner + bundled 182-doc corpus) — both removed from the main repo so it stays focused on the SDK. The dashboard was a development tool; the eval harness was for benchmarking the maintainers' own routing changes. Neither was a runtime dependency. Both remain in git history and may be re-published as standalone sister repos in the future. Performance numbers in this CHANGELOG and the README still apply — they were measured with the published `@augur/core` + `@augur/server` packages against the eval corpus that's now in git history.
- `AUGUR_SEED_DEMO` env var — required the bundled corpus that's now removed.

### Performance

- Bundled 504-query eval (LocalEmbedder + LocalReranker + MetadataChunker + stemmed BM25 + multi-stage retrieval): **NDCG@10 = 0.920**, MRR = 0.918, Recall@10 = 0.962.
- BEIR (auto-routing, default 22 MB MiniLM-L6 + 22 MB cross-encoder): SciFact 0.707, FiQA 0.338, NFCorpus 0.324.
- BEIR with BGE-large embedder: SciFact 0.742 (matches published vector-only baseline), NFCorpus 0.315.
- End-to-end p50 ~25 ms, p95 ~35 ms, ~40 QPS single-threaded with cross-encoder reranking on every query.

[Unreleased]: https://github.com/willgitdata/augur/commits/main
