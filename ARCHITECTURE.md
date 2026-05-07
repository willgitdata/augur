# Augur Architecture

This document describes how Augur is organized, why each piece exists, and how to extend it. If you're a mid-level engineer who wants to understand the system end-to-end before contributing, start here.

## High-level shape

```
                 ┌──────────────────────────────────────────────┐
                 │                 Augur                   │
                 │     (orchestrator — packages/core)           │
                 │                                              │
   index() ──►   │  Chunker → Embedder → Adapter.upsert()       │
                 │                                              │
   search() ──►  │  Router  → Embedder → Adapter.searchX()      │
                 │                       └► Reranker (optional) │
                 │                                              │
                 │  Tracer wraps every step → SearchTrace       │
                 └────────────────┬─────────────────────────────┘
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
              @augur/server       Dashboard (Next.js)
              (Fastify HTTP API)       (trace explorer + playground)
```

Five components, one orchestrator. Each component is a single TypeScript interface; each shipping implementation is one file.

## Why this layout

We picked a **monorepo with three deployable units** (`core`, `server`, `dashboard`) because:

- The SDK has to work standalone — `npm install @augur/core` and that's it. So `core` is its own package with zero runtime dependencies.
- The HTTP server is a thin wrapper. Many users will skip it entirely (embed the SDK in their own app). It's a separate package so they can.
- The dashboard is a separate Next.js app because it's a frontend concern; coupling it to the server would force everyone to ship Next.js to production.

Within `core`, code is grouped by **role**, not by feature:

```
packages/core/src/
├── augur.ts        # the orchestrator — top-level entry
├── types.ts             # shared types — single source of truth
├── adapters/            # storage backends
│   ├── adapter.ts       #   the interface + BaseAdapter (RRF hybrid)
│   ├── in-memory.ts     #   reference implementation, BM25 + brute-force vec
│   ├── pinecone.ts
│   ├── turbopuffer.ts
│   └── pgvector.ts
├── chunking/            # document-to-chunk transforms
├── embeddings/          # text-to-vector
├── routing/             # query-to-strategy decisions
│   ├── router.ts        #   HeuristicRouter (rule-based)
│   └── signals.ts       #   pure feature extraction — also future ML inputs
├── reranking/           # second-pass scoring
└── observability/       # the Tracer + TraceStore
```

This grouping is deliberate: when you add an ML router, you put it in `routing/`. When you add a new adapter, you put it in `adapters/`. The orchestrator (`augur.ts`) never has to change.

## The five interfaces (and why they look the way they do)

### `VectorAdapter` — storage

```ts
interface VectorAdapter {
  readonly capabilities: { vector, keyword, hybrid, computesEmbeddings, filtering };
  upsert(chunks: Chunk[]): Promise<void>;
  searchVector(opts): Promise<SearchResult[]>;
  searchKeyword(opts): Promise<SearchResult[]>;
  searchHybrid?(opts): Promise<SearchResult[]>;
  delete(ids: string[]): Promise<void>;
  count(): Promise<number>;
  clear(): Promise<void>;
}
```

Three design choices worth flagging:

1. **`capabilities` is part of the interface.** The router queries this before deciding. Adapters that don't support keyword (e.g. Pinecone) say so, and the router doesn't pick keyword. This is how we avoid 500s from "this backend can't do that".
2. **Hybrid is optional.** `BaseAdapter` provides RRF on top of `searchVector` + `searchKeyword`. Adapters with native hybrid (Turbopuffer) override. Adapters without keyword (Pinecone) just leave it off.
3. **No transactions, schemas, or migrations.** Those are the underlying store's job. We are not a database.

### `Chunker` — document → chunks

```ts
interface Chunker {
  readonly name: string;
  chunk(doc: Document): Chunk[];
}
```

Sync because most chunking is sync. `SemanticChunker` needs an embedder, so it adds an async path (`chunkAsync`). `chunkDocument()` is the polymorphic helper.

We picked sentence-pack as the default because it's a strict improvement over fixed-size for prose without the latency cost of semantic chunking. Heuristic, but a *good* heuristic — most teams will be better off with sentence-pack than what they have today.

### `Embedder` — text → vectors

```ts
interface Embedder {
  readonly name: string;
  readonly dimension: number;
  embed(texts: string[]): Promise<number[][]>;
}
```

Three lines. Caching, rate limiting, batching are wrapper concerns — keeping the core interface small means *anyone* can add a provider in five minutes.

We ship `HashEmbedder` (deterministic, offline, useless for semantics) and `OpenAIEmbedder` (real). The reason we ship `HashEmbedder` at all is dev-experience: `new Augur()` should not require an API key.

### `Router` — query → decision

```ts
interface Router {
  readonly name: string;
  decide(req: SearchRequest, caps: AdapterCapabilities): RoutingDecision;
}
```

The `RoutingDecision` carries the strategy *and the reasons*. This is the explainability requirement made concrete. The dashboard shows the reasons; the API returns them; tests assert on them.

Today: `HeuristicRouter`. Tomorrow: `MLRouter` trained on click data. They share the same interface — adoption is `new Augur({ router: new MLRouter(...) })`.

### `Reranker` — top-N → reordered top-K

```ts
interface Reranker {
  readonly name: string;
  rerank(query: string, results: SearchResult[], topK: number): Promise<SearchResult[]>;
}
```

We ship `HeuristicReranker` (token overlap + proximity) and `CohereReranker`. Both implement the same interface; users can wire in any cross-encoder by writing 20 lines.

## How routing actually works

The `HeuristicRouter` is a small decision tree on three inputs: query signals, adapter capabilities, and the latency budget. The full algorithm is documented inline in `routing/router.ts`, but the priority order is:

1. Forced strategy — honored.
2. Adapter capabilities — degrade gracefully if requested strategy isn't supported.
3. Query signals:
    - quoted phrase or short specific identifiers → **keyword**
    - very short query → **keyword** (if available)
    - natural-language question of length ≥ 5 → **vector**
    - high ambiguity → **vector**
    - otherwise → **hybrid**
4. Reranking decision — separately, on top of the chosen strategy. Disabled if the latency budget is < 800ms or the strategy is keyword-only.

Each step records a human-readable reason in the trace. When you see "default → hybrid (no strong signal either way)" in the dashboard, that's the router telling you it's flying blind. That's the cue to either tune the heuristics, write a domain-specific router, or train an ML router.

## How chunking is decided

Augur doesn't auto-pick a chunker — chunking is set at construction time, applied uniformly during `index()`. This is a deliberate choice:

- Chunking decisions are about *content type* (code vs prose vs transcripts), not query type.
- Mixing chunkers within a single index breaks score comparability.

What Augur *does* do is make swapping chunkers cheap: change the constructor argument, re-index. The chunker name is captured in the index trace, so the dashboard can show "this index was built with `sentence`".

## How adapters work

`upsert(chunks)` is the write path. Chunks arrive with embeddings already attached (computed by the orchestrator using the configured embedder). Adapters store them.

`searchVector(opts)` and `searchKeyword(opts)` are the read paths. The orchestrator chooses which to call based on the routing decision. Both return `SearchResult[]` with normalized fields.

`searchHybrid(opts)` defaults to RRF over the two. Adapters with native hybrid override it.

The `capabilities` block is where adapters honestly tell the world what they can and can't do. Lying here is an anti-pattern: the router will pick the wrong strategy and you'll get 500s.

## Observability

The `Tracer` is a tiny class that:

- Times spans (`tracer.span(name, fn)` is the common path).
- Collects them.
- Returns a `SearchTrace` at the end with the routing decision, query signals, candidates, adapter name, and spans.

Every search returns its trace. The HTTP server stores recent traces in a bounded `TraceStore` (default 2000) so the dashboard can show them.

We deliberately did **not** build OpenTelemetry into core. OTel is a fine ecosystem but it's fire-and-forget — it's *not* designed to put structured trace data into the response payload. We want the trace as data, not a Jaeger UI side-effect. Users who want OTel export wrap our tracer; the wrapper is ~30 lines.

## Why this isn't a microservice

The original requirements explicitly say: no microservices, no Kubernetes, no auth/multi-tenancy. The reasoning is correct:

- A retrieval orchestrator is a CPU-light, IO-bound layer. One Node process can do a lot of QPS.
- Splitting routing, embedding, indexing into separate services adds operational complexity and latency for no quality gain at MVP scale.
- The user's data already lives in their vector DB. We don't need our own database, our own auth, or our own tenancy model.

If a user needs HA, they run multiple instances behind a load balancer. If they need multi-tenancy, they namespace their adapter (every adapter we ship supports namespaces).

## Tradeoffs we made

- **In-memory by default.** Fast onboarding, terrible at scale. The user's first encounter with the product *works*; they migrate to a real adapter when the dataset outgrows it.
- **Hand-rolled OpenAPI spec.** Not generated from types. Pro: zero dependencies, readable, version-controlled. Con: drift risk between code and spec. Acceptable until the API has > 15 endpoints.
- **Heuristic router as default.** Will be wrong some of the time. The trace makes it obvious *when* it's wrong, which is the prerequisite for fixing it (whether by tuning heuristics or training a model).
- **Trace store is in-memory + bounded.** Fine for dev and demo. Production teams should plug in a Postgres/ClickHouse `TraceStore` — interface-compatible drop-in.
- **No request-level auth in core.** The server has an optional `apiKey` flag for trivial protection. Anything more (per-key rate limits, OAuth) is the user's reverse-proxy concern.

## Future extensibility

- **MLRouter**: train a logistic regression / small classifier on `(QuerySignals → click-through)` pairs. Drop in via `new Augur({ router: new MLRouter() })`.
- **Online learning**: `traceStore` already records candidate sets. Add a `recordClick(traceId, chunkId)` method, persist, retrain.
- **A/B retrieval**: `req.context.bucket` flows into the router. Implement a `RoundRobinRouter` that reports which strategy each bucket got.
- **Replay testing**: serialize traces → replay against a new index → diff results. The shape of `SearchTrace` was chosen to make this feasible.
- **LangChain / LlamaIndex bindings**: tiny adapter layer that exposes Augur as a `Retriever`. ~30 lines each.

## How to read the code (in order)

1. `packages/core/src/types.ts` — the vocabulary
2. `packages/core/src/augur.ts` — the orchestrator
3. `packages/core/src/routing/signals.ts` + `router.ts` — the brain
4. `packages/core/src/adapters/in-memory.ts` — reference adapter
5. `packages/core/src/observability/tracer.ts` — observability
6. `packages/server/src/server.ts` — HTTP wrapper

That's the system. Everything else (other adapters, other rerankers, the dashboard) is variation on these themes.
