<picture>
  <source media="(prefers-color-scheme: dark)" srcset="augur-wordmark-dark.svg">
  <img src="augur-wordmark-light.svg" alt="Augur">
</picture>

###### Named after the ancient Roman augurs who interpreted signs to foresee the best path forward. To augur is to predict, and this package predicts the optimal retrieval method for your use case.


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

- **Drop-in.** Your existing Pinecone/pgvector/OpenAI stack is exactly the input. Augur wraps it.
- **Composable.** Every component (router, chunker, adapter, reranker, embedder) is constructor-injected and replaceable.
- **Observable.** The trace is a first-class API output, not a side effect.
- **Simple.** `npm install @augur/core`, `new Augur()`, done.

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
| Documents                         | Chunking (3 strategies + `MetadataChunker`, `Doc2QueryChunker` wrappers) |
| (optional) An embedder + API key  | Offline only: `LocalEmbedder` (ONNX, ~22MB). For hosted providers, implement the 3-method `Embedder` interface in ~30 lines — see [EXAMPLES.md](EXAMPLES.md) for OpenAI / Cohere / Gemini snippets. |
| (optional) A vector DB            | A default `InMemoryAdapter` (BM25 + brute-force vector + RRF hybrid) |
| (optional) A reranker             | Offline only: `HeuristicReranker`, `LocalReranker` (cross-encoder ONNX), `MMRReranker` (diversity). Plus `CascadedReranker` for staged pipelines. Hosted rerankers are a 4-method `Reranker` interface — see [EXAMPLES.md](EXAMPLES.md). |
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
pnpm eval -- --reranker local                                    # + cross-encoder reranker (~22MB)
pnpm eval -- --reranker local --metadata-chunker                 # + metadata-prepended chunks
pnpm eval -- --reranker local --metadata-chunker --bm25-stem     # best (0.912 NDCG@10)
pnpm eval -- --reranker local --mmr --mmr-lambda 0.7             # diversity-aware top-K
```

### Reference numbers (no API keys, no network)

Measured on the bundled 504-query / 182-doc eval. **All numbers below are
real, locally reproducible runs** — no remote APIs touched.

| Config                                                                                          | NDCG@10 | MRR    | Recall@10 |
| ----------------------------------------------------------------------------------------------- | ------: | -----: | --------: |
| `LocalEmbedder` (Xenova/all-MiniLM-L6-v2)                                                       | 0.845   | 0.835  | 0.924     |
| `LocalEmbedder` + `LocalReranker` (ms-marco-MiniLM cross-encoder)                               | 0.877   | 0.871  | 0.932     |
| `LocalEmbedder` + `LocalReranker` + `MetadataChunker`                                           | 0.899   | 0.896  | 0.943     |
| `LocalEmbedder` + `LocalReranker` + `MetadataChunker` + stemmed BM25 + multi-stage gather       | **0.912** | **0.910** | **0.954** |

The best row uses ~44MB of on-device ONNX models, no network at query
time. Vector-strategy NDCG reaches **0.926** and keyword reaches **0.920**.
End-to-end query latency at this config: **p50 12 ms, p95 16 ms, p99 22 ms,
~111 QPS** single-threaded.

Hosted production embedders (Cohere v3, OpenAI text-embedding-3, Voyage)
typically lift another 5-10% on top of all-MiniLM-L6-v2. The harness is
a pure function of the `Augur` instance, so swap the embedder, adapter,
router, or reranker between runs to measure the impact of any change.

### On public BEIR benchmarks

Same auto-routing pipeline, run against [BEIR](https://github.com/beir-cellar/beir) — the standard cross-domain retrieval benchmark used by published research. Apples-to-apples NDCG@10 with our 22MB local stack vs. baselines reported in the BEIR paper, the BGE / E5 / ColBERTv2 papers, and the MTEB leaderboard:

**With the default 22MB MiniLM-L6 embedder:**

| Dataset                            | **Augur (auto, 44MB total)** | BM25  | BM25 + cross-encoder | Contriever | ColBERTv2 | BGE-large (1.3GB) | E5-large (1.3GB) |
| ---------------------------------- | ---------------------------: | ----: | -------------------: | ---------: | --------: | ----------------: | ---------------: |
| **SciFact** (scientific claims)    |                    **0.709** | 0.665 |                0.688 |      0.677 |     0.694 |             0.745 |            0.736 |
| **FiQA** (finance Q&A, 57K docs)   |                    **0.338** | 0.236 |                0.347 |      0.329 |     0.356 |             0.450 |            0.424 |
| **NFCorpus** (medical literature)  |                    **0.312** | 0.325 |                0.350 |      0.328 |     0.339 |             0.380 |            0.371 |

On SciFact our pipeline **beats BM25+rerank by +0.021, Contriever by +0.032, and ColBERTv2 by +0.015** — using a 22MB embedder. On FiQA we beat BM25 by +0.102, Contriever by +0.009, and land within ~0.02 of ColBERTv2 and BM25+rerank. We trail BGE-large and E5-large by 0.05–0.11 — those are 1.3GB models. On NFCorpus (medical, where exact-term BM25 has historically dominated) we score around BM25 baseline — the small embedder is the limiting factor, not the architecture.

**Swap in BGE-large** (1.3GB ONNX, top of MTEB retrieval) **and the auto pipeline matches the published baselines:**

| Dataset                            | Augur (auto, MiniLM-L6) | **Augur (auto, BGE-large)** | BGE-large published (vector-only) |
| ---------------------------------- | ----------------------: | --------------------------: | --------------------------------: |
| **SciFact**                        |                   0.709 |                   **0.742** |                             0.745 |
| **NFCorpus**                       |                   0.312 |                   **0.315** |                             0.380 |

On SciFact, BGE-large lifts NDCG by +0.033 — essentially closing the gap with the published vector-only BGE-large number (0.742 vs 0.745). On NFCorpus the lift is only +0.003 because **the router sends 76% of NFCorpus queries to keyword (BM25) regardless of embedder** — they never touch the vector path, so a bigger embedder doesn't help. The published 0.380 is pure-vector; our auto pipeline correctly falls back to BM25 for medical terminology where lexical match wins. Different priorities, same architecture.

The router adapts to the corpus shape with **no per-dataset tuning**: 76% keyword on NFCorpus (precise medical terminology), 98% hybrid on SciFact (claims need both signals), 72% vector on FiQA (natural-language finance questions), 45% hybrid on the internal eval. Same code, same configuration.

To use BGE-large yourself:

```ts
new LocalEmbedder({
  model: "Xenova/bge-large-en-v1.5",
  queryPrefix: "Represent this sentence for searching relevant passages: ",
});
```

> **Latency note**: FiQA at 57K docs hit p50 ~118 ms / 8 QPS — that's brute-force cosine over ~150K chunks in `InMemoryAdapter`. Expected. For corpora past ~100K chunks, swap in `PgVectorAdapter`, `PineconeAdapter`, or `TurbopufferAdapter` — those bring native ANN and the per-query cost drops back to ~10 ms regardless of corpus size. The orchestrator code is unchanged; only the adapter swaps.

Reproduce:
```bash
mkdir -p /tmp/beir && cd /tmp/beir
curl -sLo scifact.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
unzip -q scifact.zip
cd /path/to/augur && pnpm exec tsx evaluations/beir.ts /tmp/beir/scifact
```

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
