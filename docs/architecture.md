# Architecture

How Augur is organized, why each piece exists, and how to extend it. If you want to understand the system end-to-end before contributing, start here.

## High-level shape

```
                 ┌──────────────────────────────────────────────┐
                 │                 Augur                        │
                 │     (orchestrator, packages/core)            │
                 │                                              │
   index() ──►   │  Chunker → Embedder → Adapter.upsert()       │
                 │                                              │
   search() ──►  │  Router  → Embedder → Adapter.searchX()      │
                 │                       └► Reranker (optional) │
                 │                                              │
                 │  Tracer wraps every step → SearchTrace       │
                 └────────────────┬─────────────────────────────┘
                                  │
                                  ▼
                         @augur-rag/server (optional)
                         Fastify HTTP API + OpenAPI
```

Five components, one orchestrator. Each component is a single TypeScript interface; each shipping implementation is one file.

## Why this layout

The repo ships two publishable packages:

- `@augur-rag/core`: the SDK. `npm install @augur-rag/core` and that's it. Zero runtime dependencies (`@huggingface/transformers` is an optional peer for `LocalEmbedder`).
- `@augur-rag/server`: a thin Fastify HTTP wrapper. Many users will skip it and embed the SDK in their own app; it's a separate package so they can.

A trace-explorer dashboard (Next.js) and an eval harness (BEIR runner + bundled corpus) used to live in this repo and are kept out of tree now. They were development tools, not production dependencies. Both remain in git history and may be republished as standalone sister repos. Stripping them keeps the install lean.

Within `core`, code is grouped by role, not by feature:

```
packages/core/src/
├── augur.ts             # the orchestrator, top-level entry
├── types.ts             # shared types, single source of truth
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
│   └── signals.ts       #   pure feature extraction, also future ML inputs
├── reranking/           # second-pass scoring
└── observability/       # the Tracer + TraceStore
```

When you add an ML router, you put it in `routing/`. When you add a new adapter, you put it in `adapters/`. The orchestrator (`augur.ts`) never has to change.

## The five interfaces

### `VectorAdapter`: storage

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

1. `capabilities` is part of the interface. The router queries this before deciding. Adapters that don't support keyword (e.g. Pinecone) say so, and the router doesn't pick keyword. This is how we avoid 500s from "this backend can't do that".
2. Hybrid is optional. `BaseAdapter` provides RRF on top of `searchVector` + `searchKeyword`. Adapters with native hybrid (Turbopuffer) override. Adapters without keyword (Pinecone) leave it off.
3. No transactions, schemas, or migrations. Those are the underlying store's job. We are not a database.

### `Chunker`: document to chunks

```ts
interface Chunker {
  readonly name: string;
  chunk(doc: Document): Chunk[];
}

interface AsyncChunker {
  readonly name: string;
  chunkAsync(doc: Document): Promise<Chunk[]>;
}
```

Two interfaces, deliberately. Most chunking is sync (`SentenceChunker`, `FixedSizeChunker`, `MetadataChunker`). Async chunkers (`SemanticChunker`, `Doc2QueryChunker`, `ContextualChunker`) need to call out to an embedder or an LLM and can't satisfy the sync contract. Splitting the interfaces means a misuse (passing an async chunker where a sync one is expected) fails at compile time, not at runtime. `chunkDocument()` is the polymorphic entry point Augur uses internally; it accepts `Chunker | AsyncChunker`.

We picked sentence-pack as the default because it's a strict improvement over fixed-size for prose without the latency cost of semantic chunking.

### `Embedder`: text to vectors

```ts
interface Embedder {
  readonly name: string;
  readonly dimension: number;
  embed(texts: string[]): Promise<number[][]>;
}
```

Three lines. Caching, rate limiting, and batching are wrapper concerns; keeping the core interface small means anyone can add a provider in a few minutes.

We ship one offline embedder: `LocalEmbedder` (on-device ONNX, default `Xenova/all-MiniLM-L6-v2`, ~22MB). `embedder` is a required `Augur` constructor argument; there is no placeholder default, since placeholder vectors produce results that look like product bugs. Hosted providers (OpenAI, Cohere, Voyage) are a 30-line `Embedder` implementation against the provider's official SDK. See examples.md §5.

### `Router`: query to decision

```ts
interface Router {
  readonly name: string;
  decide(req: SearchRequest, caps: AdapterCapabilities): RoutingDecision;
}
```

The `RoutingDecision` carries the strategy and the reasons. This is the explainability requirement made concrete. Trace consumers (logs, dashboards, your own UI) display the reasons; the API returns them; tests assert on them.

Today: `HeuristicRouter`. Later: `MLRouter` trained on click data. They share the same interface, so adoption is `new Augur({ router: new MLRouter(...) })`.

### `Reranker`: top-N to reordered top-K

```ts
interface Reranker {
  readonly name: string;
  rerank(query: string, results: SearchResult[], topK: number): Promise<SearchResult[]>;
}
```

We ship `HeuristicReranker` (token overlap + proximity), `LocalReranker` (on-device cross-encoder ONNX), `MMRReranker` (diversity), and `CascadedReranker` (chain rerankers). All implement a one-method interface; users wire in Cohere, Voyage, or Jina by writing ~20 lines against the provider's SDK.

The default reranker is `null`, meaning bare retrieval if `reranker` is omitted. Pass `new LocalReranker()` (or any provider's reranker) to enable the cross-encoder rerank stage, which is what the recommended auto stack (and the eval-smoke regression net in CI) exercises. The previous default, a token-overlap heuristic, gave traces a "yes I rerank" line while doing close to nothing, so we removed it; an explicit reranker is now required for the rerank stage to fire.

## How routing actually works

The `HeuristicRouter` is a small decision tree on three inputs: query signals, adapter capabilities, and the latency budget. The full algorithm is documented inline in `routing/router.ts`, but the priority order is:

1. Forced strategy: honored.
2. Adapter capabilities: degrade gracefully if the requested strategy isn't supported.
3. Query signals:
    - quoted phrase or short specific identifiers → keyword
    - very short query → keyword (if available)
    - natural-language question of length ≥ 5 → vector
    - high ambiguity → vector
    - otherwise → hybrid
4. Reranking decision: separately, on top of the chosen strategy. Disabled if the latency budget is < 800ms or the strategy is keyword-only.

Each step records a human-readable reason in the trace. When you see "default → hybrid (no strong signal either way)" in your trace logs, that's the router telling you it's flying blind. That's the cue to either tune the heuristics, write a domain-specific router, or train an ML router.

## How retrieval executes

The router picks a strategy; the actual retrieval pipeline is two-stage and works the same way Turbopuffer, Vespa, and Cohere Rerank do it in production.

**Stage 1: candidate generation (recall-oriented).** When the router decided to rerank, we ignore the strategy decision for retrieval purposes and pull top-50 from both vector and keyword retrievers in parallel. The two ranked lists are then RRF-fused into a single pre-ranked candidate pool. This is the "ANN first, exact rank later" pattern, except the keyword side is BM25, not ANN. The point is the same: don't ask the slow precise scorer to look at every doc; ask the cheap recall-oriented scorers to find candidates first.

When the router decided not to rerank, the strategy decision drives a single retrieval call directly to topK; no point paying for a wider pool we won't re-score.

**Stage 2: reranking (precision-oriented).** The cross-encoder scores the top-30 of the fused pool end-to-end (reading both query and doc together). Its output is the final ordering. Trusting the cross-encoder over the original retriever scores is the whole reason production stacks have a separate rerank stage.

Why fuse before reranking instead of dedupe-and-pass-everything? Eval showed the local cross-encoder couldn't reliably re-order a noisy raw union; recall went up but NDCG slipped. RRF fusion gives the cross-encoder a pre-ranked pool where docs that scored well on either side have already risen to the top. The cross-encoder then refines a curated list rather than fighting noise. See `packages/core/src/augur.ts:gatherCandidatePool`.

## How chunking is decided

Augur doesn't auto-pick a chunker. Chunking is set at construction time and applied uniformly during `index()`. This is deliberate:

- Chunking decisions are about content type (code vs prose vs transcripts), not query type.
- Mixing chunkers within a single index breaks score comparability.

What Augur does do is make swapping chunkers cheap: change the constructor argument, re-index. The chunker name is captured in the index trace, so observability can show "this index was built with `sentence`".

## How adapters work

`upsert(chunks)` is the write path. Chunks arrive with embeddings already attached (computed by the orchestrator using the configured embedder). Adapters store them.

`searchVector(opts)` and `searchKeyword(opts)` are the read paths. The orchestrator chooses which to call based on the routing decision. Both return `SearchResult[]` with normalized fields.

`searchHybrid(opts)` defaults to RRF over the two. Adapters with native hybrid override it.

The `capabilities` block is where adapters honestly tell the world what they can and can't do. Lying here is an anti-pattern: the router will pick the wrong strategy and you'll get 500s.

## Observability

The `Tracer` is a small class that:

- Times spans (`tracer.span(name, fn)` is the common path).
- Collects them.
- Returns a `SearchTrace` at the end with the routing decision, query signals, candidates, adapter name, and spans.

Every search returns its trace. The HTTP server stores recent traces in a bounded `TraceStore` (default 2000) so a trace-consuming UI can show them.

We deliberately did not build OpenTelemetry into core. OTel is a fine ecosystem but it's not designed to put structured trace data into the response payload. We want the trace as data, not a Jaeger UI side-effect. Users who want OTel export wrap our tracer; the wrapper is ~30 lines.

## Why this isn't a microservice

The original requirements were: no microservices, no Kubernetes, no auth or multi-tenancy. The reasoning:

- A retrieval orchestrator is CPU-light, IO-bound. One Node process can do a lot of QPS.
- Splitting routing, embedding, and indexing into separate services adds operational complexity and latency for no quality gain at MVP scale.
- The user's data already lives in their vector DB. We don't need our own database, our own auth, or our own tenancy model.

If a user needs HA, they run multiple instances behind a load balancer. If they need multi-tenancy, they namespace their adapter (every adapter we ship supports namespaces).

## Tradeoffs

- In-memory by default. Fast onboarding, terrible at scale. The user's first encounter with the product works; they migrate to a real adapter when the dataset outgrows it.
- Hand-rolled OpenAPI spec, not generated from types. Pro: zero dependencies, readable, version-controlled. Con: drift risk between code and spec. Acceptable until the API has > 15 endpoints.
- Heuristic router as default. Will be wrong some of the time. The trace makes it obvious when it's wrong, which is the prerequisite for fixing it (whether by tuning heuristics or training a model).
- Trace store is in-memory + bounded. Fine for dev and demo. Production teams should plug in a Postgres or ClickHouse `TraceStore` (interface-compatible drop-in).
- No request-level auth in core. The server has an optional `apiKey` flag for trivial protection. Anything more (per-key rate limits, OAuth) is the user's reverse-proxy concern.

## Future extensibility

- `MLRouter`: train a logistic regression or small classifier on `(QuerySignals → click-through)` pairs. Drop in via `new Augur({ router: new MLRouter() })`.
- Online learning: `traceStore` already records candidate sets. Add a `recordClick(traceId, chunkId)` method, persist, retrain.
- A/B retrieval: `req.context.bucket` flows into the router. Implement a `RoundRobinRouter` that reports which strategy each bucket got.
- Replay testing: serialize traces, replay against a new index, diff results. The shape of `SearchTrace` was chosen to make this feasible.
- LangChain / LlamaIndex bindings: a tiny adapter layer that exposes Augur as a `Retriever`. ~30 lines each.

## How to read the code (in order)

1. `packages/core/src/types.ts`: the vocabulary
2. `packages/core/src/augur.ts`: the orchestrator
3. `packages/core/src/routing/signals.ts` and `router.ts`: the brain
4. `packages/core/src/adapters/in-memory.ts`: reference adapter
5. `packages/core/src/observability/tracer.ts`: observability
6. `packages/server/src/server.ts`: HTTP wrapper

That's the system. Everything else (other adapters, other rerankers, trace UIs) is variation on these themes.
