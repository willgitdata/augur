# Architecture

Augur is a retrieval orchestrator. You bring an embedder, an adapter (your vector DB), and optionally a reranker. Augur decides per query which strategy to run, returns the results with a trace.

```
index()   →  Chunker → Embedder → Adapter.upsert()
search()  →  Router  → Embedder → Adapter.searchX()  →  Reranker (optional)
                                                         │
                                                         ▼
                                                    SearchTrace
```

The repo is a small pnpm workspace with two publishable packages:

- `@augur-rag/core`: the SDK. Zero runtime deps. `@huggingface/transformers` is an optional peer for `LocalEmbedder` / `LocalReranker`.
- `@augur-rag/server`: a thin Fastify wrapper. Skip it if you're embedding the SDK in your own app.

## Layout

```
packages/core/src/
├── augur.ts             # the orchestrator
├── types.ts             # shared types
├── adapters/            # storage backends (in-memory, pgvector, pinecone, turbopuffer)
├── chunking/            # sync + async chunkers
├── embeddings/          # LocalEmbedder + tokenizers
├── routing/             # HeuristicRouter + signal extraction
├── reranking/           # local / heuristic / MMR / cascaded
└── observability/       # Tracer + TraceStore
```

Code is grouped by role, not by feature. A new adapter goes in `adapters/`, a new router in `routing/`; the orchestrator doesn't change.

## The five interfaces

Each component is one TypeScript interface. Drop-in replacements work without touching user code.

```ts
interface VectorAdapter {
  readonly capabilities: { vector, keyword, hybrid, computesEmbeddings, filtering };
  upsert(chunks): Promise<void>;
  searchVector(opts): Promise<SearchResult[]>;
  searchKeyword(opts): Promise<SearchResult[]>;
  searchHybrid?(opts): Promise<SearchResult[]>;
  delete(ids): Promise<void>;
  count(): Promise<number>;
  clear(): Promise<void>;
}

interface Chunker      { name; chunk(doc): Chunk[]; }
interface AsyncChunker { name; chunkAsync(doc): Promise<Chunk[]>; }
interface Embedder     { name; dimension; embed(texts): Promise<number[][]>; }
interface Router       { name; decide(req, caps): RoutingDecision; }
interface Reranker     { name; rerank(query, results, topK): Promise<SearchResult[]>; }
```

`capabilities` on `VectorAdapter` is what lets the router pick safely. Pinecone says `keyword: false`, so the router never picks keyword against it. `BaseAdapter` provides a default RRF-fused `searchHybrid` so most adapters just need `searchVector` + `searchKeyword`.

## Routing

`HeuristicRouter` is a small rule tree over query signals (`packages/core/src/routing/signals.ts`):

- quoted phrase, code-like token, or short identifier → keyword
- very short query → keyword (if available)
- natural-language question ≥ 5 words → vector
- otherwise → hybrid
- reranking layered on top when the latency budget allows

Every decision records its reasons in the trace. If the router is wrong on your corpus, the trace tells you why.

## Two-stage retrieval

When reranking is on, retrieval gathers top-50 from both vector and keyword paths in parallel and RRF-fuses them. The cross-encoder rescores the top-30 of the fused pool. This is the ANN-then-rerank pattern Cohere Rerank, Turbopuffer, and Vespa use. When reranking is off, a single retrieval call goes straight to `topK`.

## Trace

Every `search()` returns its `SearchTrace`. Fields you'll care about:

```ts
{
  decision: { strategy, reasons, signals, reranked },
  spans:    [{ name, durationMs, attributes }],
  candidates, adapter, embeddingModel, totalMs,
}
```

The HTTP server keeps recent traces in an in-memory ring buffer (`TraceStore`) and exposes them at `/traces`. Plug in your own store for persistence.
