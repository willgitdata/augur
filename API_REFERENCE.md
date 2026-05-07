# API Reference

Two surfaces: the **TypeScript SDK** (`@augur/core`) and the **HTTP API** (`@augur/server`). Both expose the same conceptual operations: index, search, inspect traces.

---

## SDK

### `new Augur(options?)`

```ts
import { Augur } from "@augur/core";

const qb = new Augur({
  adapter,        // VectorAdapter        — default: InMemoryAdapter
  embedder,       // Embedder             — default: HashEmbedder
  chunker,        // Chunker              — default: SentenceChunker
  router,         // Router               — default: HeuristicRouter
  reranker,       // Reranker             — default: HeuristicReranker
  traceStore,     // TraceStore           — optional capture store
  autoIndexAdHocDocuments, // boolean     — default: true
});
```

All options are optional; with none, you get a fully functional in-memory pipeline.

### `qb.index(documents)`

Chunks → embeds → upserts.

```ts
const result = await qb.index([
  { id: "1", content: "...", metadata: { source: "wiki" } },
]);
// result = { documents: 1, chunks: 4, trace: { chunkingMs, embeddingMs, upsertMs, totalMs } }
```

### `qb.search(request)`

```ts
const { results, trace } = await qb.search({
  query: "...",
  documents,           // optional — ad-hoc inline docs
  topK: 10,            // default 10
  forceStrategy,       // "vector" | "keyword" | "hybrid" | "rerank"
  latencyBudgetMs,     // soft budget — affects rerank decision
  filter: { source: "wiki" },
  context: { userId: "u1" },  // forwarded to the router
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
  adapter: string;
  embeddingModel?: string;
}
```

### `qb.clear()`

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
  "embedder": "hash-embedder",
  "chunker": "sentence",
  "router": "heuristic-v1",
  "reranker": "heuristic-reranker",
  "capabilities": { "vector": true, "keyword": true, "hybrid": true, ... }
}
```

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
await qb.search({ query: "...", forceStrategy: "keyword" });
```

The decision still ends up in the trace, with `forceStrategy=...` as the recorded reason.

### Bound latency

```ts
await qb.search({ query: "...", latencyBudgetMs: 250 });
```

A budget under 800ms disables reranking automatically. The trace will say "reranking skipped (latency budget)".

### Filter by metadata

```ts
await qb.search({ query: "...", filter: { source: "wiki", lang: "en" } });
```

Adapters that report `capabilities.filtering = true` (all of ours except HashEmbedder-mode) honor this. Filters are AND-combined.

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
import { Augur } from "@augur/core";
import express from "express";

const qb = new Augur({ /* prod config */ });
const app = express();
app.post("/search", async (req, res) => {
  const result = await qb.search(req.body);
  res.json(result);
});
```

You don't need `@augur/server` for this. Use it when you want OpenAPI, traces UI, and auth out of the box.
