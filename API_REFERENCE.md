# API Reference

Two surfaces: the **TypeScript SDK** (`@augur-rag/core`) and the **HTTP API** (`@augur-rag/server`). Both expose the same conceptual operations: index, search, inspect traces.

---

## SDK

### `new Augur(options?)`

```ts
import { Augur, LocalEmbedder } from "@augur-rag/core";

const augr = new Augur({
  embedder,                // Embedder             — REQUIRED (e.g. new LocalEmbedder())
  adapter,                 // VectorAdapter        — default: InMemoryAdapter
  chunker,                 // Chunker | AsyncChunker — default: SentenceChunker
  router,                  // Router               — default: HeuristicRouter
  reranker,                // Reranker | null      — default: null (no reranker)
  traceStore,              // TraceStore           — optional capture store
  autoIndexAdHocDocuments, // boolean              — default: true
  adHocCacheSize,          // number               — default: 8 (LRU; 0 disables)
  autoLanguageFilter,      // boolean              — default: false
});
```

`embedder` is the only required option. The default `reranker` is `null`
— pass `new LocalReranker()` (zero-API-key on-device cross-encoder) to
get the headline accuracy mode where the cross-encoder votes on every
query.

### `augr.index(documents)`

Chunks → embeds → upserts.

```ts
const result = await augr.index([
  { id: "1", content: "...", metadata: { source: "wiki" } },
]);
// result = { documents: 1, chunks: 4, trace: { chunkingMs, embeddingMs, upsertMs, totalMs } }
```

### `augr.search(request)`

```ts
const { results, trace } = await augr.search({
  query: "...",
  documents,           // optional — ad-hoc inline docs (LRU-cached by content fingerprint)
  topK: 10,            // default 10
  forceStrategy,       // "vector" | "keyword" | "hybrid" | "rerank"
  latencyBudgetMs,     // soft budget — affects rerank decision
  filter: { source: "wiki" },
  context: { userId: "u1" },  // forwarded to the router
  minScore: 0.4,       // optional confidence floor — drops results below this score
});
```

Returns:

```ts
{
  results: SearchResult[],
  trace: SearchTrace,
}
```

### Types — quick reference

```ts
interface Document {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface Chunk {
  id: string;            // `${documentId}:${index}`
  documentId: string;
  content: string;
  index: number;
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

interface SearchResult {
  chunk: Chunk;
  score: number;             // higher = more relevant
  rawScores?: Record<string, number>;
}

interface SearchTrace {
  id: string;
  query: string;
  startedAt: string;
  totalMs: number;
  decision: {
    strategy: "vector" | "keyword" | "hybrid" | "rerank";
    reasons: string[];
    reranked: boolean;
    signals: QuerySignals;
  };
  spans: Array<{ name; startMs; endMs; durationMs; attributes? }>;
  candidates: number;
  adapter: string;          // bare adapter name — never mutated with "(ad-hoc)" suffixes
  embeddingModel?: string;
  adHoc?: boolean;          // true when the query used a scratch adapter from req.documents
  adHocCacheHit?: boolean;  // true when an ad-hoc query reused a cached scratch adapter
  autoLanguageFilter?: string;       // BCP-47 code the auto-filter pinned to, if fired
  autoLanguageFilterDropped?: boolean; // true if the auto-filter would have emptied the pool
}
```

### `augr.clear()`

Wipes the underlying adapter.

---

## HTTP API

Default base URL: `http://localhost:3001`. All endpoints accept and return JSON.

### Auth

If the server was started with `AUGUR_API_KEY=<secret>`, every request must include `x-api-key: <secret>`. Otherwise no auth.

### `GET /health`

Returns the live configuration:

```json
{
  "status": "ok",
  "adapter": "in-memory",
  "embedder": "local:Xenova/all-MiniLM-L6-v2",
  "chunker": "sentence",
  "router": "heuristic-v1",
  "reranker": null,
  "capabilities": { "vector": true, "keyword": true, "hybrid": true, ... }
}
```

`reranker` is `null` when the server was started without one configured.
The default `@augur-rag/server` CLI ships no reranker — pass
`new LocalReranker()` (or any provider's reranker) when you wire your
own `buildServer({ embedder, reranker })` to keep cross-encoder voting
on.

Use this to verify that the deployed instance is wired the way you think it is.

### `GET /openapi.json`

Full OpenAPI 3.1 spec.

### `GET /docs`

Swagger UI rendered against `/openapi.json`.

### `POST /index`

```json
{
  "documents": [
    { "id": "1", "content": "...", "metadata": { "source": "wiki" } }
  ]
}
```

Response: `IndexResponse` (see types).

### `POST /search`

Request body matches the `SearchRequest` type. The most common shape:

```json
{
  "query": "How do I configure connection pooling?",
  "topK": 5
}
```

Response:

```json
{
  "results": [ ... ],
  "trace": { ... }
}
```

The trace includes the routing decision, the reasons, the spans, the candidate count, and the adapter name.

### `GET /traces?limit=100`

Most recent traces, newest first. `limit` is capped at 500.

### `GET /traces/:id`

Single trace by ID. Returns 404 if expired/missing.

### `DELETE /traces`

Clears the trace store.

### `GET /admin/stats`

```json
{ "chunks": 1284, "traces": 412 }
```

### `POST /admin/clear`

Wipes the underlying adapter. Idempotent.

---

## Common patterns

### Force a strategy for testing

```ts
await augr.search({ query: "...", forceStrategy: "keyword" });
```

The decision still ends up in the trace, with `forceStrategy=...` as the recorded reason.

### Bound latency

```ts
await augr.search({ query: "...", latencyBudgetMs: 250 });
```

A budget under 800ms disables reranking automatically. The trace will say "reranking skipped (latency budget)".

### Filter by metadata

```ts
await augr.search({ query: "...", filter: { source: "wiki", lang: "en" } });
```

Adapters that report `capabilities.filtering = true` (all of ours) honor this. Filters are AND-combined.

### Custom router with logging

```ts
class LoggingRouter extends HeuristicRouter {
  decide(req, caps) {
    const decision = super.decide(req, caps);
    console.log("[route]", req.query, "→", decision.strategy);
    return decision;
  }
}
```

### Wrap the SDK in your own service

```ts
import { Augur } from "@augur-rag/core";
import express from "express";

const augr = new Augur({ /* prod config */ });
const app = express();
app.post("/search", async (req, res) => {
  const result = await augr.search(req.body);
  res.json(result);
});
```

You don't need `@augur-rag/server` for this. Use it when you want OpenAPI, traces UI, and auth out of the box.
