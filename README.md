# Augur (Named after the ancient Roman augurs who interpreted signs {query vectors} to foresee outcomes {optimized retrieval architecture})


**An adaptive retrieval orchestration layer for AI/RAG systems.**

Augur sits on top of your existing vector database, embedder, and reranker and decides — per query — *which retrieval strategy to use*. Vector? Keyword? Hybrid? Vector-then-rerank? It picks based on signals from the query itself, with a transparent, explainable decision recorded in every response.

It is **not** a vector database. It is a thin, composable orchestration layer designed to drop into existing RAG stacks.

```ts
import { Augur } from "@augur/core";

const augr = new Augur();

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
| Documents                         | Chunking (3 strategies + `MetadataChunker` wrapper) |
| (optional) An embedder + API key  | Offline: `HashEmbedder`, `TfIdfEmbedder`, `LocalEmbedder` (ONNX, no network). Hosted: `OpenAIEmbedder`, `GeminiEmbedder` |
| (optional) A vector DB            | A default `InMemoryAdapter` (BM25 + brute-force vector + RRF hybrid) |
| (optional) A reranker             | Offline: `HeuristicReranker`, `LocalReranker` (cross-encoder ONNX). Hosted: `CohereReranker`, `JinaReranker`, `HttpCrossEncoderReranker`. Plus `CascadedReranker` for staged pipelines. |
| Nothing                           | Routing, hybrid fusion, traces, dashboard, HTTP API |

## Evaluation

Augur ships a built-in eval harness (**182 docs, 504 labeled queries**
across 12 archetypes — factoid, procedural, definitional, code,
error_code, quoted, short_kw, named_entity, negation, non_english,
ambiguous, internal). The corpus covers Postgres, Kubernetes, Redis,
networking, ML/AI, security/compliance, code snippets, company-internal
runbooks/policies, and 12 foreign languages (es, ja, fr, de, zh, ko, pt,
ru, ar, hi, it, vi). Metrics: NDCG@10, MRR, Recall@10 — overall, per
category, per router-chosen strategy.

```bash
pnpm eval                                                        # default config
pnpm eval -- --verbose                                           # per-query lines
pnpm eval -- --save baseline.json                                # snapshot metrics
pnpm eval -- --compare baseline.json                             # diff vs snapshot
pnpm eval -- --embedder tfidf                                    # swap to TfIdfEmbedder (offline, no deps)
pnpm eval -- --embedder local                                    # offline ONNX (Xenova/all-MiniLM-L6-v2, ~22MB)
pnpm eval -- --embedder local --reranker local                   # + cross-encoder reranker (~22MB)
pnpm eval -- --embedder local --reranker local --metadata-chunker # full local stack
pnpm eval -- --embedder local --reranker local --metadata-chunker --bm25-stem  # best (0.910 NDCG@10)
pnpm eval -- --embedder local --reranker local --mmr --mmr-lambda 0.7          # diversity-aware top-K
pnpm eval -- --embedder gemini --gemini-cache-dir .cache/gemini  # Gemini API w/ disk cache
```

### Reference numbers (no API keys, no network)

Measured on the bundled 504-query / 182-doc eval. **All numbers below are
real, locally reproducible runs** — no remote APIs touched.

| Config                                                                                          | NDCG@10 | MRR    | Recall@10 |
| ----------------------------------------------------------------------------------------------- | ------: | -----: | --------: |
| `HashEmbedder` (default placeholder, not semantic)                                              | 0.786   | 0.782  | 0.857     |
| `TfIdfEmbedder`                                                                                 | 0.825   | 0.816  | 0.906     |
| `TfIdfEmbedder` + `MetadataChunker`                                                             | 0.848   | 0.839  | 0.923     |
| `LocalEmbedder` (Xenova/all-MiniLM-L6-v2)                                                       | 0.845   | 0.835  | 0.924     |
| `LocalEmbedder` + `LocalReranker` (ms-marco-MiniLM cross-encoder)                               | 0.877   | 0.871  | 0.932     |
| `LocalEmbedder` + `LocalReranker` + `MetadataChunker`                                           | 0.899   | 0.896  | 0.943     |
| `LocalEmbedder` + `LocalReranker` + `MetadataChunker` + stemmed BM25 (`useStemming`)            | **0.910** | **0.907** | **0.956** |

The best row uses ~44MB of on-device ONNX models, no network at query
time, and beats the HashEmbedder default by **+12.4% NDCG@10**, with
vector-strategy NDCG going from **0.638 → 0.922 (+28.4%)** and keyword
from 0.874 → 0.925 (+5.1%, almost entirely from Porter stemming).

Hosted production embedders (Cohere v3, OpenAI text-embedding-3, Voyage)
typically lift another 5-10% on top of all-MiniLM-L6-v2. The harness is
a pure function of the `Augur` instance, so swap the embedder, adapter,
router, or reranker between runs to measure the impact of any change.

### MMR for diverse top-K (opt-in)

`MMRReranker` implements Maximal Marginal Relevance — useful when queries
have multiple distinct relevant docs and you want the top-K to span them
rather than concentrate on near-duplicates. **Not on by default**: on the
bundled QA-style eval where most queries have 1 relevant doc, MMR pushes
hits out of top-10 in favor of diversity (NDCG drops ~0.04). Reach for it
on multi-aspect queries, recommendation feeds, and RAG pipelines where
the LLM benefits from non-redundant context. See [EXAMPLES §5](EXAMPLES.md#5-switching-to-openai--pinecone) for wiring.

## Status

This is a v0.1 MVP under active development. It is small enough to read end-to-end in an afternoon and useful enough to point at a real RAG project tomorrow. Issues, ideas, and PRs welcome — see [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md).

## License

MIT.
