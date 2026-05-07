# Augur

**An adaptive retrieval orchestration layer for AI/RAG systems.**

Augur sits on top of your existing vector database, embedder, and reranker and decides — per query — *which retrieval strategy to use*. Vector? Keyword? Hybrid? Vector-then-rerank? It picks based on signals from the query itself, with a transparent, explainable decision recorded in every response.

It is **not** a vector database. It is a thin, composable orchestration layer designed to drop into existing RAG stacks.

```ts
import { Augur } from "@augur/core";

const qb = new Augur();

await qb.index([
  { id: "1", content: "PostgreSQL supports vector indexing via pgvector." },
  { id: "2", content: "Pinecone is a managed vector database." },
]);

const { results, trace } = await qb.search({
  query: "How do I store vectors in Postgres?",
});

// results[0].chunk.documentId === "1"
// trace.decision.strategy === "vector"
// trace.decision.reasons === ["natural-language question → semantic search", ...]
```

## Why Augur

Modern RAG pipelines fail in three predictable ways:

1. **One-strategy-fits-all retrieval.** Pure vector search misses exact-match queries (error codes, SKUs, names). Pure BM25 misses paraphrased questions. Most teams pick one and ship known-bad recall.
2. **Untunable chunking.** Chunking is the highest-leverage knob in RAG, yet most stacks hardcode 512-token windows and never revisit it.
3. **Opaque retrieval.** When a query returns the wrong result, you can't tell *why*. Was the embedding bad? Did the reranker drop it? Did the user just not use the right keywords?

Augur addresses all three:

- **Adaptive routing**: `HeuristicRouter` (today) decides between vector / keyword / hybrid / rerank based on query signals. The interface is built so an `MLRouter` can drop in later without changing user code.
- **Pluggable chunking**: `FixedSizeChunker`, `SentenceChunker`, `SemanticChunker` ship in core. Anything else is a one-method interface.
- **First-class observability**: every search returns a `SearchTrace` with the decision, the reasoning, the spans, the candidates, and the scores. The dashboard is just a UI on top of that data.

## Product principles

- **Drop-in, not rip-and-replace.** Your existing Pinecone/pgvector/OpenAI stack is exactly the input. Augur wraps it.
- **Composable like Stripe.** Every component (router, chunker, adapter, reranker, embedder) is constructor-injected and replaceable.
- **Observable like Datadog.** The trace is a first-class API output, not a side effect.
- **Simple like Vercel.** `npm install @augur/core`, `new Augur()`, done.

## Repository layout

```
augur/
├── packages/
│   ├── core/              # @augur/core — the SDK
│   └── server/            # @augur/server — Fastify HTTP API + OpenAPI
├── apps/
│   └── dashboard/         # Next.js trace explorer + query playground
├── examples/
│   ├── basic-search/      # 30-line "hello world"
│   ├── custom-adapter/    # write your own VectorAdapter
│   └── chunking/          # compare chunking strategies
├── README.md              # ← you are here
├── ARCHITECTURE.md        # how the system is organized + why
├── DEVELOPMENT_GUIDE.md   # contributor + local-dev guide
├── API_REFERENCE.md       # SDK + HTTP API reference
├── EXAMPLES.md            # extended walkthroughs
└── docker-compose.yml     # one-command local stack
```

## Quick start (local, no API keys)

```bash
# 1. Install
pnpm install

# 2. Build the core + server packages
pnpm build

# 3. Run the example
pnpm --filter example-basic-search start
```

For the full stack (API + dashboard):

```bash
docker compose up
# dashboard → http://localhost:3000
# API docs  → http://localhost:3001/docs
```

## Pluggable backends

Augur ships adapters for:

- `InMemoryAdapter` — zero-dep, BM25 + brute-force vector. Good for dev and small datasets.
- `PineconeAdapter` — Pinecone REST. Vector only (Pinecone has no native BM25).
- `TurbopufferAdapter` — Turbopuffer REST. Native vector + BM25 + hybrid.
- `PgVectorAdapter` — Postgres + `vector` extension. Vector + tsvector keyword + RRF hybrid.

Writing a new adapter is implementing five methods. See [`examples/custom-adapter`](./examples/custom-adapter/index.ts).

## What's in the box vs. what to bring

| You bring                         | Augur provides                              |
|-----------------------------------|--------------------------------------------------|
| Documents                         | Chunking (3 strategies)                          |
| (optional) An embedder + API key  | A default `HashEmbedder` that runs offline       |
| (optional) A vector DB            | A default `InMemoryAdapter`                      |
| (optional) A reranker             | A default `HeuristicReranker` + `CohereReranker` |
| Nothing                           | Routing, hybrid fusion, traces, dashboard, HTTP API |

## Status

This is a v0.1 MVP under active development. It is small enough to read end-to-end in an afternoon and useful enough to point at a real RAG project tomorrow. Issues, ideas, and PRs welcome — see [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md).

## License

MIT.
