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

The recommended local stack — `LocalEmbedder` + `LocalReranker` +
`MetadataChunker(SentenceChunker)` + `InMemoryAdapter({ useStemming: true })`
+ the default `HeuristicRouter` — runs in **44 MB total on-device**
(`Xenova/all-MiniLM-L6-v2` 22 MB embedder + `Xenova/ms-marco-MiniLM-L-6-v2`
22 MB cross-encoder), no network at query time, no API keys. The cross-
encoder votes on every query by default (`alwaysRerank: true`); latency-
conscious deployments opt out with `new HeuristicRouter({ alwaysRerank: false })`.

The router adapts per corpus with no per-dataset tuning — code-like
queries route to keyword, natural-language questions to vector, the rest
to weighted hybrid; quoted phrases and short identifiers always favor
BM25. The trace records the routing decision and reasons for every
query, so when the auto choice is wrong on your corpus you see *why*.

### What's enforced in CI today

`packages/core/src/eval-smoke.test.ts` runs a 16-doc / 12-query synthetic
corpus with a deterministic stub embedder on every PR and asserts
`NDCG@10 > 0.65`. That's a structural regression net — it catches "the
routing pipeline broke" but does *not* establish absolute quality
numbers. Read it, run it (`pnpm test`), trust it for what it is.

### Quality benchmarks (out of tree)

The earlier development tree included a 504-query / 182-doc internal
eval and a BEIR runner (SciFact / FiQA / NFCorpus). On that harness the
auto stack measured competitively against published BM25, Contriever,
and ColBERTv2 baselines using a 22 MB embedder vs. their 100s of MB —
but **the harness is no longer in this repo** (preserved at commit
`feffc73^` and slated for republish as a standalone `augur-eval` sister
repo). Until then, anyone wanting to verify quality numbers should
either re-run from that commit or wait for the dedicated repo. We're not
quoting specific numbers here we can't reproduce on demand.

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

The repo intentionally ships only the SDK and the optional HTTP wrapper. A trace-explorer dashboard and the BEIR-eval harness used during development are kept out of tree — both remain available in git history at commit `feffc73^` and are slated for republish as standalone sister repos.

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
