<picture>
  <source media="(prefers-color-scheme: dark)" srcset="augur-wordmark-dark.svg">
  <img src="augur-wordmark-light.svg" alt="Augur">
</picture>

###### Named after the ancient Roman augurs who interpreted signs to foresee the best path forward. To augur is to predict, and this package predicts the optimal retrieval method for your use case.

**An adaptive retrieval orchestration layer for AI / RAG systems.**

> **Competitive accuracy, out of the box. Full traces on every search. Retrieval becomes a one-line constructor — not a six-week side quest.**

Augur sits on top of your existing vector database, embedder, and reranker and decides — *per query* — which retrieval strategy to use: vector, keyword, hybrid, or vector-then-rerank. Every decision is a transparent trace recorded in the response. Drop into any RAG pipeline; **the auto method is best out of the box**.

```ts
import { Augur, LocalEmbedder } from "@augur/core";

const augr = new Augur({ embedder: new LocalEmbedder() });

await augr.index([
  { id: "1", content: "PostgreSQL supports vector indexing via pgvector." },
  { id: "2", content: "Pinecone is a managed vector database." },
]);

const { results, trace } = await augr.search({
  query: "How do I store vectors in Postgres?",
});

// results[0].chunk.documentId === "1"
// trace.decision.strategy === "vector"
// trace.decision.reasons === ["natural-language question → semantic search", ...]
```

## Why Augur

Modern RAG pipelines fail in three predictable ways. Augur addresses all three:

- **One-strategy-fits-all retrieval** misses recall. Pure vector misses exact matches (error codes, SKUs); pure BM25 misses paraphrases. → **Adaptive routing**: `HeuristicRouter` picks per-query strategy from query signals; the cross-encoder reranks every candidate. The interface lets `MLRouter` drop in later without changing user code.
- **Untunable chunking** kneecaps quality. → **Pluggable chunkers**: `SentenceChunker`, `SemanticChunker`, plus `ContextualChunker` (Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) — +67% reduction in chunk-failure rate per their published numbers). One-method interface for any custom chunker.
- **Opaque retrieval** makes debugging RAG impossible. → **First-class observability**: every `search()` returns a `SearchTrace` with the decision, reasoning, span timings, candidates, and scores.

## Performance — auto method, no tuning

The recommended local stack — `LocalEmbedder` + `LocalReranker` + `MetadataChunker(SentenceChunker)` + `InMemoryAdapter({ useStemming: true })` + the default `HeuristicRouter` — runs in **44 MB total on-device** (`Xenova/all-MiniLM-L6-v2` 22 MB embedder + `Xenova/ms-marco-MiniLM-L-6-v2` 22 MB cross-encoder), no network at query time, no API keys. Same configuration across every dataset — no per-corpus tuning.

**BEIR — NDCG@10, Auto vs. published baselines:**

| Dataset                            |    **Auto** |  BM25 | BM25+rerank | Contriever | ColBERTv2 | BGE-large (1.3GB) | E5-large (1.3GB) |
| ---------------------------------- | ----------: | ----: | ----------: | ---------: | --------: | ----------------: | ---------------: |
| **SciFact** (scientific claims)    |   **0.707** | 0.665 |       0.688 |      0.677 |     0.694 |             0.745 |            0.736 |
| **FiQA** (finance Q&A, 57K docs)   |   **0.345** | 0.236 |       0.347 |      0.329 |     0.356 |             0.450 |            0.424 |
| **NFCorpus** (medical literature)  |   **0.324** | 0.325 |       0.350 |      0.328 |     0.339 |             0.380 |            0.371 |

Auto numbers measured by the [`Eval matrix` workflow](.github/workflows/eval.yml) — trigger from the Actions tab with `target=beir-only`, `auto_stack=default`, `run_fiqa=true`. Published baselines are from the BEIR (Thakur et al. 2021), BGE (Xiao et al. 2023), E5 (Wang et al. 2022), and ColBERTv2 papers — static, not re-measured here.

**The headline:** at **44 MB on-device**, Auto **beats BM25, BM25+rerank, Contriever, AND ColBERTv2** on SciFact; **beats BM25 and Contriever** on FiQA and ties BM25+rerank within noise (0.345 vs 0.347); ties BM25 on NFCorpus. The 1.3 GB BGE-large and E5-large win consistently — for a stack ~30× our footprint, you'd expect them to. The routing pipeline matches their published numbers when you swap in BGE-large yourself (`new LocalEmbedder({ model: "Xenova/bge-large-en-v1.5", queryPrefix: "..." })` — see [EXAMPLES.md](EXAMPLES.md)).

The router adapts per corpus with no per-dataset tuning — code-like queries route to keyword, natural-language questions to vector, the rest to weighted hybrid; quoted phrases and short identifiers always favor BM25. The trace records the routing decision and reasons for every query, so when the auto choice is wrong on your corpus you see *why*.

### What's enforced in CI on every commit

`packages/core/src/eval-smoke.test.ts` runs a 16-doc / 12-query synthetic corpus with a deterministic stub embedder and asserts `NDCG@10 > 0.65`. That's a structural regression net — it catches "the routing pipeline broke" but does *not* establish the absolute quality numbers above. Read it, run it (`pnpm test`), trust it for what it is.

## Pluggable backends

| Adapter             | Capabilities                                          |
| ------------------- | ----------------------------------------------------- |
| `InMemoryAdapter`   | Zero-dep, BM25 + brute-force vector. Dev / small datasets. |
| `PineconeAdapter`   | Pinecone REST. Vector only.                           |
| `TurbopufferAdapter`| Native vector + BM25 + hybrid.                        |
| `PgVectorAdapter`   | Postgres + `vector` extension. Vector + tsvector + RRF hybrid. |

Custom adapter is five methods — see [`examples/custom-adapter`](./examples/custom-adapter/index.ts).

## Repository layout

```
augur/
├── packages/core/         # @augur/core — the SDK (this is what most users install)
├── packages/server/       # @augur/server — optional Fastify wrapper for standalone deploy
├── examples/              # basic-search, custom-adapter, chunking
├── ARCHITECTURE.md        # how the system is organized + why
├── EXAMPLES.md            # extended walkthroughs (hosted embedders, contextual retrieval, BGE-large, MMR, ...)
├── API_REFERENCE.md       # SDK + HTTP API reference
└── DEVELOPMENT_GUIDE.md   # contributor + local-dev guide
```

The repo intentionally ships only the SDK and the optional HTTP wrapper. A trace-explorer dashboard and the BEIR-eval harness used during development are kept out of tree — both remain available in git history at commit `feffc73^` and are slated for republish as standalone sister repos. The `Eval matrix` workflow restores the harness in CI on demand to measure the numbers above.

## Quick start

```bash
pnpm install && pnpm build
pnpm --filter example-basic-search start

# Or run the HTTP server:
docker compose up
# API docs → http://localhost:3001/docs
```

## Status

v0.1, active development. Small enough to read end-to-end in an afternoon, useful enough to drop into a real RAG project tomorrow. Issues, ideas, and PRs welcome — see [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md).

## License

MIT.
