<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/augur-wordmark-dark.svg">
  <img src="assets/augur-wordmark-light.svg" alt="Augur">
</picture>

**Pick the best retrieval strategy per query — automatically.** Adaptive retrieval orchestration for RAG and semantic search in TypeScript. Augur sits on top of your vector DB (pgvector, Pinecone, Turbopuffer, or in-memory) and picks per query which strategy to run: vector, BM25 keyword, weighted hybrid, or vector-then-cross-encoder rerank. The routing decision and timings come back in every search response.

```bash
npm install @augur-rag/core @huggingface/transformers
```

```ts
import {
  Augur,
  InMemoryAdapter,
  LocalEmbedder,
  LocalReranker,
  MetadataChunker,
  SentenceChunker,
} from "@augur-rag/core";

// Recommended stack — matches the BEIR numbers in the Performance table below.
// 44 MB on-device total, no API keys, no network at query time.
const augr = new Augur({
  embedder: new LocalEmbedder(),
  reranker: new LocalReranker(),
  chunker: new MetadataChunker({ base: new SentenceChunker() }),
  adapter: new InMemoryAdapter({ useStemming: true }),
});

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

The bare `new Augur({ embedder: new LocalEmbedder() })` shape also works — it gives single-strategy retrieval without the cross-encoder rerank stage. The headline performance numbers in the table below require the recommended stack above; pass `new LocalReranker()` to turn the rerank stage on.

## Why

Most RAG pipelines pick one retrieval strategy and run it for everything. Pure vector misses exact matches (error codes, SKUs, named entities). Pure BM25 misses paraphrases. Hybrid is better but a fixed mix is wrong for whichever side the current query needs less of. Augur routes per query from cheap heuristics on query signals, with the cross-encoder reranker as the final precision stage. When the auto choice is wrong, the trace shows you why.

## How Augur compares

|                                  | LangChain.js retriever | LlamaIndex.ts retriever | Raw vector-DB SDK | **Augur** |
| -------------------------------- | :--------------------: | :---------------------: | :---------------: | :-------: |
| Per-query strategy routing       |           —            |            —            |         —         |     ✓     |
| First-class search trace on every call |       —            |            —            |         —         |     ✓     |
| Hybrid (BM25 + vector) RRF fusion |        partial         |         partial         |         —         |     ✓     |
| Cross-encoder rerank step        |         manual         |         manual          |      manual       |     ✓     |
| On-device default (zero API key) |           —            |            —            |         —         |     ✓     |
| Drop-in adapter contract         |           ✓            |            ✓            |       n/a         |     ✓     |
| One-line `npm install` to working retrieval |  —          |            —            |         —         |     ✓     |

Augur is intentionally smaller in scope than LangChain or LlamaIndex — no agents, no chains, no LLM-call wrapper. It does one thing (route + retrieve + rerank, with a trace) and stays out of the way of the rest of your stack.

## Performance

On-device stack: `Xenova/all-MiniLM-L6-v2` (22 MB embedder) + `Xenova/ms-marco-MiniLM-L-6-v2` (22 MB cross-encoder). 44 MB total. No network at query time, no API keys, no per-corpus tuning.

BEIR NDCG@10:

| Dataset                          |    Auto |  BM25 | BM25+rerank | Contriever | ColBERTv2 | BGE-large (1.3GB) |
| -------------------------------- | ------: | ----: | ----------: | ---------: | --------: | ----------------: |
| SciFact (scientific claims)      |   0.707 | 0.665 |       0.688 |      0.677 |     0.694 |             0.745 |
| FiQA (finance Q&A, 57K docs)     |   0.345 | 0.236 |       0.347 |      0.329 |     0.356 |             0.450 |
| NFCorpus (medical literature)    |   0.324 | 0.325 |       0.350 |      0.328 |     0.339 |             0.380 |

Auto numbers measured by the [`Eval matrix`](.github/workflows/eval.yml) workflow. Baseline columns are the published numbers from the BEIR, BGE, E5, and ColBERTv2 papers. Same router across all three corpora, no per-dataset tuning. Swap in BGE-large as the embedder if you want to match the 1.3 GB column.

## Adapters

| Adapter             | Capabilities                                          |
| ------------------- | ----------------------------------------------------- |
| `InMemoryAdapter`   | Zero-dep, BM25 + brute-force vector. Dev / small datasets. |
| `PgVectorAdapter`   | Postgres + `pgvector`. Vector + tsvector + RRF hybrid. |
| `PineconeAdapter`   | Pinecone REST. Vector only.                           |
| `TurbopufferAdapter`| Native vector + BM25 + hybrid.                        |

A custom adapter is five methods. See [`examples/custom-adapter`](./examples/custom-adapter/index.ts).

## Install

```bash
npm install @augur-rag/core @huggingface/transformers
```

`@huggingface/transformers` is an optional peer needed for `LocalEmbedder` and `LocalReranker`. Skip it if you're wiring a hosted embedder via the one-method `Embedder` interface.

## Quick start

```bash
pnpm install && pnpm build
pnpm --filter example-basic-search start

# Or the HTTP server:
docker compose up
# http://localhost:3001/docs   (Swagger UI)
```

## Docs

- [docs/architecture.md](./docs/architecture.md) — how the pieces fit together
- [docs/examples.md](./docs/examples.md) — hosted embedders, contextual retrieval, pgvector, MMR, trace inspection
- [CHANGELOG.md](./CHANGELOG.md)

## License

MIT.
