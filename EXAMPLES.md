# Examples

Hands-on walkthroughs. Each example lives in `examples/<name>/` and is runnable with `pnpm --filter example-<name> start`.

---

## 1. Hello, Augur

The 30-line "hello world".

```ts
import { Augur } from "@augur/core";

const qb = new Augur();

await qb.index([
  { id: "pg",    content: "PostgreSQL supports vector search via pgvector." },
  { id: "redis", content: "Redis is an in-memory key-value store." },
  { id: "k8s",   content: "Kubernetes manages containers across hosts." },
]);

const { results, trace } = await qb.search({ query: "vector database" });

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

const qb = new Augur({ adapter: new JsonFileAdapter("./store.json") });
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

## 5. Switching to OpenAI + Pinecone

When you outgrow the in-memory adapter:

```ts
import {
  Augur,
  OpenAIEmbedder,
  PineconeAdapter,
  CohereReranker,
} from "@augur/core";

const qb = new Augur({
  adapter: new PineconeAdapter({
    indexHost: process.env.PINECONE_INDEX_HOST!,
    apiKey: process.env.PINECONE_API_KEY!,
    namespace: "prod",
  }),
  embedder: new OpenAIEmbedder({ model: "text-embedding-3-small" }),
  reranker: new CohereReranker({ model: "rerank-english-v3.0" }),
});
```

Three constructor args. No code changes elsewhere. The router automatically adapts to Pinecone's keyword-incapable status (it stops picking keyword and lets the reranker carry precision).

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
import { Augur, PgVectorAdapter, OpenAIEmbedder } from "@augur/core";

const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
await client.connect();

const qb = new Augur({
  adapter: new PgVectorAdapter({
    client: { query: (sql, params) => client.query(sql, params).then((r) => ({ rows: r.rows })) },
    table: "chunks",
    dimension: 1536,
  }),
  embedder: new OpenAIEmbedder(),
});
```

Vector + keyword + hybrid all in one place, no extra services.

---

## 7. Trace introspection

The whole point of the system. Every search produces a trace; every trace is enough to explain (or debug) the result.

```ts
const { trace } = await qb.search({ query: "how to deploy with zero downtime" });

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
  qb = new Augur({ /* ... */ });
  lc_namespace = ["custom", "augur"];
  async _getRelevantDocuments(query: string) {
    const { results } = await this.qb.search({ query });
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
