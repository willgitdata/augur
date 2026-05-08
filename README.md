<picture>
  <source media="(prefers-color-scheme: dark)" srcset="augur-wordmark-dark.svg">
  <img src="augur-wordmark-light.svg" alt="Augur">
</picture>

###### Named after the ancient Roman augurs who interpreted signs to foresee the best path forward. To augur is to predict, and this package predicts the optimal retrieval method for your use case.

**An adaptive retrieval orchestration layer for AI / RAG systems.**

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

The numbers below all use **`new Augur({ embedder, reranker, chunker, adapter })` with default routing** — same configuration across every dataset.

**Bundled eval** (504 queries, 182 docs, 12 query archetypes, 12 languages):

| Metric    |    Auto |
| --------- | ------: |
| NDCG@10   | **0.920** |
| MRR       |   0.918 |
| Recall@10 |   0.962 |

**Public BEIR benchmarks** (apples-to-apples vs. baselines reported in BEIR / BGE / E5 / ColBERTv2 papers):

| Dataset                            |    **Auto** |  BM25 | BM25+rerank | Contriever | ColBERTv2 | BGE-large (1.3GB) | E5-large (1.3GB) |
| ---------------------------------- | ----------: | ----: | ----------: | ---------: | --------: | ----------------: | ---------------: |
| **SciFact** (scientific claims)    |   **0.707** | 0.665 |       0.688 |      0.677 |     0.694 |             0.745 |            0.736 |
| **FiQA** (finance Q&A, 57K docs)   |   **0.338** | 0.236 |       0.347 |      0.329 |     0.356 |             0.450 |            0.424 |
| **NFCorpus** (medical literature)  |   **0.324** | 0.325 |       0.350 |      0.328 |     0.339 |             0.380 |            0.371 |

The auto stack is **44 MB total on-device** (`Xenova/all-MiniLM-L6-v2` 22 MB + `Xenova/ms-marco-MiniLM-L-6-v2` reranker 22 MB), no network at query time. On SciFact it beats BM25+rerank, Contriever, and ColBERTv2 — using a 22 MB embedder vs. their 1.3 GB. The trailing gap on the largest dense models is purely embedder size; the routing pipeline matches their published numbers when you swap in BGE-large (`new LocalEmbedder({ model: "Xenova/bge-large-en-v1.5", queryPrefix: "..." })` — see [EXAMPLES.md](EXAMPLES.md)).

The router adapts per corpus with **no per-dataset tuning**: 76% keyword on NFCorpus (medical terms reward exact match), 98% hybrid on SciFact (claims need both signals), 72% vector on FiQA (natural-language questions), 45% hybrid on the bundled eval. Same code, same configuration.

End-to-end latency, single-threaded: **p50 ~25 ms, p95 ~35 ms, ~40 QPS** with the cross-encoder voting on every query. For RAG where the LLM call dominates, this is free. Latency-conscious deployments can opt out with `new HeuristicRouter({ alwaysRerank: false })` and get the BM25 fast path back.

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
├── packages/core/         # @augur/core — the SDK
├── packages/server/       # @augur/server — Fastify HTTP API + OpenAPI
├── apps/dashboard/        # Next.js trace explorer + query playground
├── examples/              # basic-search, custom-adapter, chunking
├── ARCHITECTURE.md        # how the system is organized + why
├── EXAMPLES.md            # extended walkthroughs (hosted embedders, contextual retrieval, BGE-large, MMR, ...)
├── API_REFERENCE.md       # SDK + HTTP API reference
└── DEVELOPMENT_GUIDE.md   # contributor + local-dev guide
```

## Quick start

```bash
pnpm install && pnpm build
pnpm --filter example-basic-search start

# Or the full stack (API + dashboard):
docker compose up
# dashboard → http://localhost:3000
# API docs  → http://localhost:3001/docs
```

## Reproduce the numbers

```bash
# Bundled eval
pnpm eval -- --reranker local --metadata-chunker --bm25-stem

# BEIR — example: SciFact
mkdir -p /tmp/beir && cd /tmp/beir
curl -sLo scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip -q scifact.zip
cd - && pnpm exec tsx evaluations/beir.ts /tmp/beir/scifact
```

## Status

v0.1, active development. Small enough to read end-to-end in an afternoon, useful enough to drop into a real RAG project tomorrow. Issues, ideas, and PRs welcome — see [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md).

## License

MIT.
