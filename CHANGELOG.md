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
- BEIR runner (`evaluations/beir.ts`) — runs the auto-routing pipeline against any BEIR dataset, with `--model` / `--query-prefix` / `--dtype` / `--device` / `--fast-keyword` flags.
- Server demo seed (`AUGUR_SEED_DEMO=1`) — server seeds the bundled 182-doc eval corpus on boot for dev / dashboard demos.

### Changed

- **BREAKING** `signals.tokens` → `signals.wordCount`, `signals.avgTokenLen` → `signals.avgWordLen`. The previous names were misleading — these are whitespace-split words, not embedding subword tokens.
- **BREAKING** `signals.language` type widened from `"en" | "non-en"` to `string` (specific language code).
- **BREAKING** `HeuristicRouter` defaults `alwaysRerank: true`. The cross-encoder votes on every query out of the box for best NDCG@10. Latency-sensitive callers opt out via `new HeuristicRouter({ alwaysRerank: false })`.
- **BREAKING** `Augur` constructor requires `embedder`. The placeholder `HashEmbedder` / `TfIdfEmbedder` are removed — they produced near-random vectors that surfaced as product bugs. Use `LocalEmbedder` (zero-config, on-device ONNX) or implement the 3-method `Embedder` interface.
- `gatherCandidatePool` now uses adaptive weighted RRF (query-signal prior + retrieval-confidence shift) instead of symmetric RRF.
- Multi-stage retrieval pipeline (gather → fuse → rerank) — production pattern from Turbopuffer / Vespa / Cohere Rerank.
- HeuristicRouter rule split: code-like syntax stays on keyword; bare identifier / numeric / date-version queries route to hybrid for topical recall.
- `InMemoryAdapter.searchKeyword` — phrase-substring boost for quoted phrases and short identifier-like queries.

### Removed

- `HashEmbedder`, `TfIdfEmbedder` — placeholder embedders no longer ship; `LocalEmbedder` is the only built-in.
- Hosted-provider embedder/reranker classes from core (`GeminiEmbedder`, etc.) — moved to user-implemented `Embedder` / `Reranker` instances per the EXAMPLES.md §5 pattern.

### Performance

- Bundled 504-query eval (LocalEmbedder + LocalReranker + MetadataChunker + stemmed BM25 + multi-stage retrieval): **NDCG@10 = 0.920**, MRR = 0.918, Recall@10 = 0.962.
- BEIR (auto-routing, default 22 MB MiniLM-L6 + 22 MB cross-encoder): SciFact 0.707, FiQA 0.338, NFCorpus 0.324.
- BEIR with BGE-large embedder: SciFact 0.742 (matches published vector-only baseline), NFCorpus 0.315.
- End-to-end p50 ~25 ms, p95 ~35 ms, ~40 QPS single-threaded with cross-encoder reranking on every query.

[Unreleased]: https://github.com/willgitdata/augur/commits/main
