# Examples

Hands-on walkthroughs. Each example lives in `examples/<name>/` and is runnable with `pnpm --filter example-<name> start`.

---

## 1. Hello, Augur

The 30-line "hello world".

```ts
import { Augur } from "@augur/core";

const augr = new Augur();

await augr.index([
  { id: "pg",    content: "PostgreSQL supports vector search via pgvector." },
  { id: "redis", content: "Redis is an in-memory key-value store." },
  { id: "k8s",   content: "Kubernetes manages containers across hosts." },
]);

const { results, trace } = await augr.search({ query: "best vector database for production" });

console.log(trace.decision.strategy); // → "hybrid"
console.log(results[0].chunk.documentId);
```

Run: `pnpm --filter example-basic-search start`. The full example exercises four query archetypes (natural-language question, error code, quoted phrase, single keyword) and prints the routing decision for each.

---

## 2. The router in action

Same documents, four queries, four different strategies:

```ts
const queries = [
  "How do I configure connection pooling in Postgres?",  // → vector
  "ERR_CONNECTION_REFUSED 4101",                         // → keyword
  '"liveness probes"',                                   // → keyword (quoted)
  "kubernetes",                                          // → keyword (very short)
];
```

What you'll see in the trace:

```
=== Query: How do I configure connection pooling in Postgres?
Strategy: vector (+rerank)  · 14.3 ms · 30 candidates
Reasons:
  - natural-language question → semantic search
  - reranking enabled (latency budget allows)
```

```
=== Query: ERR_CONNECTION_REFUSED 4101
Strategy: keyword  · 1.8 ms · 1 candidates
Reasons:
  - short query with specific identifiers/codes
  - reranking skipped (latency budget)
```

The point: the user never asked about strategies. The router did.

---

## 3. Custom adapter

Implement `VectorAdapter`, get the whole orchestrator for free.

```ts
import { BaseAdapter } from "@augur/core";

class JsonFileAdapter extends BaseAdapter {
  readonly name = "json-file";
  readonly capabilities = {
    vector: true, keyword: true, hybrid: true,
    computesEmbeddings: false, filtering: false,
  };
  // ... upsert / searchVector / searchKeyword / delete / count / clear ...
}

const augr = new Augur({ adapter: new JsonFileAdapter("./store.json") });
```

The full implementation is `examples/custom-adapter/index.ts`. About 90 lines, including the BM25-ish keyword search.

The takeaway: hybrid retrieval (RRF) and the routing engine come for free as soon as you implement vector + keyword. You don't write the fusion code yourself.

---

## 4. Comparing chunkers

```ts
import { FixedSizeChunker, SentenceChunker, SemanticChunker, HashEmbedder } from "@augur/core";

const fixed    = new FixedSizeChunker({ size: 200, overlap: 30 });
const sentence = new SentenceChunker({ targetSize: 200 });
const semantic = new SemanticChunker({ embedder: new HashEmbedder(), threshold: 0.5 });
```

Run on the same document, you'll see:

- `fixed-size` → predictable count, mid-sentence cuts
- `sentence` → grammatical units, slight count variance
- `semantic` → topic-aligned cuts, fewer chunks for cohesive text

Pick based on content type, not on intuition. When in doubt, `SentenceChunker` is the default for a reason.

---

## 5. Switching to a hosted embedder + reranker

`@augur/core` ships only offline embedders and rerankers (HashEmbedder,
TfIdfEmbedder, LocalEmbedder, HeuristicReranker, LocalReranker, MMRReranker).
Hosted providers are intentionally not in core — the `Embedder` interface is
three methods, the `Reranker` is one. Implement against your provider's
official SDK and pass it in:

```ts
import { Augur, PineconeAdapter, type Embedder, type Reranker } from "@augur/core";
import OpenAI from "openai";
import { CohereClient } from "cohere-ai";

class OpenAIEmbedder implements Embedder {
  readonly name = "openai:text-embedding-3-small";
  readonly dimension = 1536;
  private client = new OpenAI();
  async embed(texts: string[]): Promise<number[][]> {
    const r = await this.client.embeddings.create({
      model: "text-embedding-3-small",
      input: texts,
    });
    return r.data.map((d) => d.embedding);
  }
}

class CohereReranker implements Reranker {
  readonly name = "cohere:rerank-english-v3.0";
  private client = new CohereClient({ token: process.env.COHERE_API_KEY! });
  async rerank(query: string, results, topK) {
    const r = await this.client.rerank({
      model: "rerank-english-v3.0",
      query,
      documents: results.map((x) => x.chunk.content),
      topN: topK,
    });
    return r.results.map((x) => ({ ...results[x.index]!, score: x.relevanceScore }));
  }
}

const augr = new Augur({
  adapter: new PineconeAdapter({
    indexHost: process.env.PINECONE_INDEX_HOST!,
    apiKey: process.env.PINECONE_API_KEY!,
    namespace: "prod",
  }),
  embedder: new OpenAIEmbedder(),
  reranker: new CohereReranker(),
});
```

The router adapts to Pinecone's keyword-incapable status automatically (stops
picking keyword, lets the reranker carry precision).

### Picking a better default embedder (no API key needed)

The default `HashEmbedder` is a feature-hashed bag-of-tokens — useful as
a deterministic placeholder, but its vectors are not semantically
meaningful. Three offline upgrade paths:

**TF-IDF (no extra deps).** Feature-hashed TF-IDF with Porter stemming and
stopword removal — the classical IR baseline:

```ts
import { Augur, TfIdfEmbedder, MetadataChunker, SentenceChunker } from "@augur/core";

const augr = new Augur({
  embedder: new TfIdfEmbedder(),
  chunker: new MetadataChunker({ base: new SentenceChunker() }),
});
```

**Local sentence-transformer (recommended for production-grade local).**
`LocalEmbedder` runs a real sentence-transformer model entirely on-device
via `@huggingface/transformers` (ONNX Runtime). Default model is
`Xenova/all-MiniLM-L6-v2` (~22MB, 384d). First run downloads the model
to `~/.cache/huggingface/hub`; subsequent runs are instant.

```ts
import {
  Augur,
  LocalEmbedder,
  LocalReranker,
  MetadataChunker,
  SentenceChunker,
} from "@augur/core";

const augr = new Augur({
  embedder: new LocalEmbedder(),                                  // 22MB, 384d
  reranker: new LocalReranker(),                                  // 22MB cross-encoder
  chunker: new MetadataChunker({ base: new SentenceChunker() }),
});
```

You'll need to install the optional peer dep:

```bash
pnpm add @huggingface/transformers
```

For higher accuracy at a slightly larger size, swap the model and supply
the model's required prefixes:

```ts
// BGE-small: top of MTEB at this size; query prefix required.
new LocalEmbedder({
  model: "Xenova/bge-small-en-v1.5",
  queryPrefix: "Represent this sentence for searching relevant passages: ",
});

// E5-small: balanced; both prefixes required.
new LocalEmbedder({
  model: "Xenova/e5-small-v2",
  queryPrefix: "query: ",
  docPrefix: "passage: ",
});

// nomic-embed-text-v1.5: 768d, 137MB; instruction-tuned.
new LocalEmbedder({
  model: "nomic-ai/nomic-embed-text-v1.5",
  dimension: 768,
  queryPrefix: "search_query: ",
  docPrefix: "search_document: ",
});
```

`MetadataChunker` wraps any base chunker and prepends
`[doc-id | topic | title]` to each chunk before embedding — the
"Doc2Query lite" pattern.

**Measured impact on the bundled 504-query eval (no API keys):**

```
HashEmbedder (default)                                                NDCG@10 = 0.786
TfIdfEmbedder                                                         NDCG@10 = 0.825 (+0.039)
TfIdfEmbedder + MetadataChunker                                       NDCG@10 = 0.848 (+0.062)
LocalEmbedder (all-MiniLM-L6-v2)                                      NDCG@10 = 0.845 (+0.059)
LocalEmbedder + LocalReranker                                         NDCG@10 = 0.877 (+0.091)
LocalEmbedder + LocalReranker + MetadataChunker                       NDCG@10 = 0.899 (+0.113)
LocalEmbedder + LocalReranker + MetadataChunker + stemmed BM25        NDCG@10 = 0.910 (+0.124)
```

Vector-strategy NDCG goes from 0.638 (HashEmbedder) to **0.922** with the
full local stack — the kind of jump you typically need a hosted API for.

### Doc2Query — synthetic-question expansion at index time

For each chunk, generate N questions the chunk could answer using a small
T5 model (`Xenova/LaMini-T5-61M`, ~24MB), then append them to the chunk's
content before embedding and BM25 indexing. Cost is paid once at index;
**zero query-time latency**. Works particularly well on conversational
queries against reference-style content (and on non-English chunks
indexed alongside English questions).

```ts
import { Augur, Doc2QueryChunker, SentenceChunker, MetadataChunker } from "@augur/core";

const augr = new Augur({
  chunker: new Doc2QueryChunker({
    base: new MetadataChunker({ base: new SentenceChunker() }),
    numQueries: 3,             // questions per chunk; more = better recall, longer index
    model: "Xenova/LaMini-T5-61M",
  }),
});
```

Requires `@huggingface/transformers` (same dep as LocalEmbedder).

### Query-aware hybrid weights

`Augur.search()` now picks the BM25-vs-vector mix per query from the
router's signals — quoted phrases / specific tokens / very short queries
lean BM25 (0.3-0.4 vector weight), long natural-language questions lean
vector (0.7), default is 0.5. Production hybrid systems all do some
version of this; a fixed 0.5/0.5 mix under-weights whichever side is
wrong for the current query shape.

No configuration needed — it's automatic when strategy = "hybrid".

### Stemmed BM25 (`InMemoryAdapter({ useStemming: true })`)

Turns on Porter stemming + English stopword filtering for the keyword
path. Same pipeline Lucene/Elasticsearch use by default. On the bundled
eval this single flag adds ~+0.022 NDCG@10 to *any* config and is the
biggest cheap win on quoted / named-entity / short keyword queries
(running ↔ runs, connection ↔ connections all collapse to one stem).

```ts
import { Augur, InMemoryAdapter, LocalEmbedder, LocalReranker } from "@augur/core";

const augr = new Augur({
  adapter: new InMemoryAdapter({ useStemming: true }),
  embedder: new LocalEmbedder(),
  reranker: new LocalReranker(),
});
```

### MMR (Maximal Marginal Relevance) for diverse top-K

For ambiguous queries with multiple relevant docs, pure-relevance reranking
concentrates on near-duplicates. `MMRReranker` rebalances toward novelty:

```ts
import { CascadedReranker, LocalReranker, MMRReranker, Augur } from "@augur/core";

// Cross-encoder narrows by relevance; MMR diversifies the survivors.
const reranker = new CascadedReranker([
  [new LocalReranker(), 50],
  [new MMRReranker({ lambda: 0.7 }), 10],
]);

const augr = new Augur({ reranker, /* ... */ });
```

`λ = 1.0` is pure relevance; `λ = 0.7` is the standard "relevance with
diversity boost". Not enabled by default — on QA-style queries (one
relevant doc) MMR pushes hits out in favor of variety, which hurts. Reach
for it on multi-aspect queries, search-results pages, and RAG pipelines
where the LLM benefits from non-redundant context.

### Picking a reranker

`@augur/core` ships four offline rerankers:

```ts
import {
  HeuristicReranker,    // zero-dep, sub-ms; weak baseline (token overlap + proximity)
  LocalReranker,        // local ONNX cross-encoder (~22MB, ms-marco-MiniLM-L-6-v2)
  MMRReranker,          // diversity-aware reranking (no model)
  CascadedReranker,     // chain rerankers: cheap-broad → expensive-narrow
} from "@augur/core";

// Cascaded rerank — heuristic narrows 100 → 50, cross-encoder narrows 50 → 10:
const cascade = new CascadedReranker([
  [new HeuristicReranker(), 50],
  [new LocalReranker(), 10],
]);
```

For Cohere, Jina, Voyage, or any hosted cross-encoder, implement the
four-method `Reranker` interface directly against the provider's SDK —
see §5 above for a Cohere snippet. `HeuristicReranker` is fine for a
smoke test; cross-encoder rerankers are typically the single biggest
accuracy lever once embeddings are decent.

---

## 6. Postgres with pgvector

The recommended adapter for most teams — you probably already have Postgres.

```sql
CREATE EXTENSION vector;
CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  content TEXT NOT NULL,
  index INT NOT NULL,
  metadata JSONB,
  embedding VECTOR(1536),
  content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (content_tsv);
CREATE INDEX ON chunks (document_id);
```

```ts
import pg from "pg";
import { Augur, PgVectorAdapter, LocalEmbedder } from "@augur/core";

const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
await client.connect();

const augr = new Augur({
  adapter: new PgVectorAdapter({
    client: { query: (sql, params) => client.query(sql, params).then((r) => ({ rows: r.rows })) },
    table: "chunks",
    dimension: 384,
  }),
  embedder: new LocalEmbedder(),  // or your hosted embedder per §5
});
```

Vector + keyword + hybrid all in one place, no extra services.

---

## 7. Trace introspection

The whole point of the system. Every search produces a trace; every trace is enough to explain (or debug) the result.

```ts
const { trace } = await augr.search({ query: "how to deploy with zero downtime" });

console.log(trace.decision);
// {
//   strategy: "vector",
//   reasons: [
//     "natural-language question → semantic search",
//     "reranking enabled (latency budget allows)"
//   ],
//   reranked: true,
//   signals: { tokens: 6, avgTokenLen: 4.5, hasQuotedPhrase: false, ... }
// }

console.log(trace.spans.map(s => `${s.name}: ${s.durationMs.toFixed(1)}ms`));
// ["embed:query: 1.4ms", "search:vector: 0.8ms", "rerank: 0.3ms"]
```

Save the trace ID; if a user reports a bad result, you can look up exactly what happened.

---

## 8. The HTTP API

```bash
curl -X POST localhost:3001/index \
  -H 'Content-Type: application/json' \
  -d '{"documents":[{"id":"1","content":"hello world"}]}'

curl -X POST localhost:3001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello","topK":3}'
```

The trace is in the response. Open `http://localhost:3000` (the dashboard) to see it visually.

---

## 9. Wiring into LangChain (sketch)

```ts
import { BaseRetriever } from "@langchain/core/retrievers";
import { Augur } from "@augur/core";

class AugurRetriever extends BaseRetriever {
  augr = new Augur({ /* ... */ });
  lc_namespace = ["custom", "augur"];
  async _getRelevantDocuments(query: string) {
    const { results } = await this.augr.search({ query });
    return results.map((r) => ({
      pageContent: r.chunk.content,
      metadata: { ...r.chunk.metadata, score: r.score, traceId: r.chunk.id },
    }));
  }
}
```

20 lines. The same pattern works for LlamaIndex; we'll publish official bindings as `@augur/langchain` and `@augur/llamaindex` once the SDK API stabilizes.

---

## 10. A/B testing two routers

```ts
const a = new Augur({ router: new HeuristicRouter() });
const b = new Augur({ router: new MyMLRouter() });

const which = userId.charCodeAt(0) % 2 === 0 ? a : b;
const result = await which.search({ query, context: { ab: which === a ? "A" : "B" } });
```

The `context.ab` propagates into the trace, so every search is analyzable by bucket.
