<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/augur-wordmark-dark.svg">
  <img src="assets/augur-wordmark-light.svg" alt="Augur">
</picture>

###### Named after the ancient Roman augurs who interpreted signs to foresee the best path forward. To augur is to predict, and this package predicts the optimal retrieval method for your use case.

Adaptive retrieval orchestration for RAG and semantic search in TypeScript. Augur sits on top of your existing vector database (pgvector, Pinecone, Turbopuffer, or in-memory), embedder, and reranker, and decides per query which strategy to use: vector / semantic search, BM25 keyword, weighted hybrid, or vector-then-cross-encoder rerank. Every decision is recorded in a trace returned with the search response.

```ts
import { Augur, LocalEmbedder } from "@augur-rag/core";

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

## Use cases

- Add semantic search to an existing app. Wrap your Postgres / pgvector, Pinecone, or Turbopuffer store with Augur and you get per-query strategy routing and cross-encoder reranking on top, in roughly 50 lines of integration code.
- Combine vector and keyword search. RRF fusion comes free via `BaseAdapter`; implement vector + keyword paths and you get hybrid retrieval and the routing engine for free.
- Cross-encoder reranking without standing up extra infrastructure. `LocalReranker` runs MS-MARCO MiniLM on-device (22 MB ONNX, no API key). Plug in a hosted reranker (Cohere, Voyage, Jina) through a one-method interface.
- On-device embeddings. `LocalEmbedder` runs sentence-transformers (`all-MiniLM-L6-v2` by default, swap to BGE / E5 / nomic) via `@huggingface/transformers`. No OpenAI bill, no network at query time.
- Anthropic-style contextual retrieval. `ContextualChunker` ships with the published prompt template and a pluggable LLM provider. Works with Anthropic, OpenAI, Gemini, or anything you wire up.
- Debuggable retrieval. Every `search()` returns a `SearchTrace` with the routing decision, reasons, span timings, candidates, and raw scores, so when the auto choice is wrong on your corpus you can see why.

## Why this exists

RAG pipelines fail in three predictable ways, and Augur addresses each:

- One-strategy-fits-all retrieval misses recall. Pure vector misses exact matches (error codes, SKUs, named entities); pure BM25 misses paraphrases. `HeuristicRouter` picks per-query strategy from query signals, and the cross-encoder reranks every candidate. The `Router` interface lets an ML-based router drop in later without changes to user code.
- Untunable chunking caps quality. `SentenceChunker`, `SemanticChunker`, and `ContextualChunker` (implementing [Anthropic's contextual retrieval](https://www.anthropic.com/news/contextual-retrieval), reported 67% reduction in chunk-failure rate) are all interchangeable through a one-method interface.
- Opaque retrieval makes debugging hard. Every `search()` returns a `SearchTrace` with the decision, reasons, span timings, candidates, and scores.

## Performance

The numbers below come from running the auto method on a 44 MB on-device stack (22 MB MiniLM embedder + 22 MB MS-MARCO cross-encoder reranker). No network at query time, no API keys, no per-corpus tuning. That's roughly 30x smaller than the 1.3 GB BGE-large and E5-large baselines.

BEIR NDCG@10, auto vs. published baselines:

| Dataset                          |    Auto |  BM25 | BM25+rerank | Contriever | ColBERTv2 | BGE-large (1.3GB) | E5-large (1.3GB) |
| -------------------------------- | ------: | ----: | ----------: | ---------: | --------: | ----------------: | ---------------: |
| SciFact (scientific claims)      |   0.707 | 0.665 |       0.688 |      0.677 |     0.694 |             0.745 |            0.736 |
| FiQA (finance Q&A, 57K docs)     |   0.345 | 0.236 |       0.347 |      0.329 |     0.356 |             0.450 |            0.424 |
| NFCorpus (medical literature)    |   0.324 | 0.325 |       0.350 |      0.328 |     0.339 |             0.380 |            0.371 |

Auto numbers measured by the [`Eval matrix` workflow](.github/workflows/eval.yml); trigger from the Actions tab with `target=beir-only`, `auto_stack=default`, `run_fiqa=true`. Baseline columns are the published numbers from the BEIR (Thakur et al. 2021), BGE (Xiao et al. 2023), E5 (Wang et al. 2022), and ColBERTv2 papers; static, not re-measured here.

At 44 MB on-device, the auto stack beats BM25, BM25+rerank, Contriever, and ColBERTv2 on SciFact. It beats BM25 and Contriever on FiQA and is within noise of BM25+rerank (0.345 vs 0.347). It ties BM25 on NFCorpus. The 1.3 GB BGE-large and E5-large win consistently. For a stack about 30x our footprint, you would expect them to. The routing pipeline matches their published numbers if you swap in BGE-large yourself; see [docs/examples.md](docs/examples.md) for the constructor.

The router adapts per corpus with no per-dataset tuning. Code-like queries route to keyword, natural-language questions to vector, the rest to weighted hybrid; quoted phrases and short identifiers favor BM25. The trace records the routing decision and reasons for every query.

### What CI enforces on every commit

`packages/core/src/eval-smoke.test.ts` runs a 16-doc / 12-query synthetic corpus with a deterministic stub embedder and asserts `NDCG@10 > 0.65`. That is a structural regression net: it catches "the routing pipeline broke" but does not establish the absolute quality numbers above. Read it, run it (`pnpm test`), and trust it for what it is.

## Pluggable backends

| Adapter             | Capabilities                                          |
| ------------------- | ----------------------------------------------------- |
| `InMemoryAdapter`   | Zero-dep, BM25 + brute-force vector. Dev / small datasets. |
| `PineconeAdapter`   | Pinecone REST. Vector only.                           |
| `TurbopufferAdapter`| Native vector + BM25 + hybrid.                        |
| `PgVectorAdapter`   | Postgres + `pgvector` extension. Vector + tsvector + RRF hybrid. |

A custom adapter is five methods. See [`examples/custom-adapter`](./examples/custom-adapter/index.ts).

## Repository layout

```
augur/
├── packages/core/         # @augur-rag/core, the SDK (this is what most users install)
├── packages/server/       # @augur-rag/server, optional Fastify wrapper for standalone deploy
├── examples/              # basic-search, custom-adapter, chunking
├── docs/
│   ├── architecture.md    # how the system is organized + why
│   ├── examples.md        # extended walkthroughs (hosted embedders, contextual retrieval, BGE-large, MMR, ...)
│   ├── api-reference.md   # SDK + HTTP API reference
│   ├── development.md     # contributor + local-dev guide
│   ├── releasing.md       # publish runbook
│   └── migrating.md       # version-to-version upgrade notes
└── assets/                # wordmark SVGs
```

The repo ships only the SDK and the optional HTTP wrapper. A trace-explorer dashboard and the BEIR-eval harness used during development live in git history at commit `feffc73^` and are slated for republish as standalone sister repos. The `Eval matrix` workflow restores the harness in CI on demand to produce the table above.

## Quick start

```bash
pnpm install && pnpm build
pnpm --filter example-basic-search start

# Or run the HTTP server:
docker compose up
# API docs → http://localhost:3001/docs
```

## Status

v0.1, active development. Issues, ideas, and PRs welcome. See [docs/development.md](./docs/development.md).

## License

MIT.
