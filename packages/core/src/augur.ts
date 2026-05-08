import type { VectorAdapter } from "./adapters/adapter.js";
import { InMemoryAdapter } from "./adapters/in-memory.js";
import { type Chunker, FixedSizeChunker, SentenceChunker, SemanticChunker, chunkDocument } from "./chunking/chunker.js";
import type { Embedder } from "./embeddings/embedder.js";
import { Tracer, TraceStore } from "./observability/tracer.js";
import { HeuristicReranker, type Reranker } from "./reranking/reranker.js";
import { HeuristicRouter, type Router } from "./routing/router.js";
import type {
  Document,
  IndexResponse,
  SearchRequest,
  SearchResponse,
  SearchResult,
} from "./types.js";

export interface AugurOptions {
  /**
   * Embedder used for both indexing and querying. **Required**. Use
   * `LocalEmbedder` for a fully on-device default, or implement the
   * `Embedder` interface against your provider's SDK (see EXAMPLES.md §5).
   */
  embedder: Embedder;
  /** Storage adapter. Defaults to InMemoryAdapter. */
  adapter?: VectorAdapter;
  /** Chunker used during `index()`. Defaults to SentenceChunker. */
  chunker?: Chunker | SemanticChunker;
  /** Routing engine. Defaults to HeuristicRouter. */
  router?: Router;
  /** Reranker. Defaults to HeuristicReranker. */
  reranker?: Reranker;
  /** Optional trace store — when provided, every search trace is captured. */
  traceStore?: TraceStore;
  /**
   * Auto-indexing: if a search request includes `documents`, automatically
   * index them in a scratch in-memory adapter for that request only.
   * Defaults to true. Set false to require explicit `index()` calls.
   */
  autoIndexAdHocDocuments?: boolean;
}

/**
 * Augur — the unified retrieval orchestration entry point.
 *
 *   import { Augur, LocalEmbedder } from "@augur/core";
 *
 *   const augr = new Augur({ embedder: new LocalEmbedder() });
 *   await augr.index([{ id: "1", content: "..." }]);
 *   const { results, trace } = await augr.search({ query: "hello" });
 *
 * `embedder` is required. Pick `LocalEmbedder` for fully on-device, or
 * implement the `Embedder` interface against your provider's SDK (OpenAI,
 * Cohere, Voyage, etc) — see EXAMPLES.md §5 for snippets.
 *
 * Why everything is constructor-injected:
 * - Testability: every component is mockable in isolation.
 * - Forward compatibility: when MLRouter ships, swap one constructor arg.
 * - Aligns with the philosophy: "augment, don't replace". Users keep their
 *   existing embedder/store/reranker; Augur is just the conductor.
 */
export class Augur {
  readonly adapter: VectorAdapter;
  readonly embedder: Embedder;
  readonly chunker: Chunker | SemanticChunker;
  readonly router: Router;
  readonly reranker: Reranker;
  readonly traceStore?: TraceStore;
  private autoIndex: boolean;

  constructor(opts: AugurOptions) {
    if (!opts || !opts.embedder) {
      throw new Error(
        "Augur: `embedder` is required. Use `new LocalEmbedder()` for an on-device " +
          "default, or implement the Embedder interface against your provider " +
          "(see EXAMPLES.md §5)."
      );
    }
    this.embedder = opts.embedder;
    this.adapter = opts.adapter ?? new InMemoryAdapter();
    this.chunker = opts.chunker ?? new SentenceChunker();
    this.router = opts.router ?? new HeuristicRouter();
    this.reranker = opts.reranker ?? new HeuristicReranker();
    if (opts.traceStore) this.traceStore = opts.traceStore;
    this.autoIndex = opts.autoIndexAdHocDocuments ?? true;
  }

  /**
   * Index a batch of documents. Returns timing breakdown for observability.
   */
  async index(documents: Document[]): Promise<IndexResponse> {
    const t0 = performance.now();
    let chunkingMs = 0;
    let embeddingMs = 0;
    let upsertMs = 0;

    // 1. Chunk
    const c0 = performance.now();
    const allChunks = (
      await Promise.all(documents.map((d) => chunkDocument(this.chunker, d)))
    ).flat();
    chunkingMs = performance.now() - c0;

    // 2. Embed (skip if adapter computes embeddings itself)
    if (!this.adapter.capabilities.computesEmbeddings && allChunks.length > 0) {
      const e0 = performance.now();
      const texts = allChunks.map((c) => c.content);
      // Let stateful embedders ingest the corpus before embedding, so
      // query-time vocabulary / IDFs reflect indexed content.
      this.embedder.fit?.(texts);
      // Prefer embedDocuments() — embedders that distinguish doc vs query
      // task types (Gemini, Cohere v3) score noticeably higher when the
      // role is explicit.
      const vecs = await (this.embedder.embedDocuments
        ? this.embedder.embedDocuments(texts)
        : this.embedder.embed(texts));
      vecs.forEach((v, i) => {
        allChunks[i]!.embedding = v;
      });
      embeddingMs = performance.now() - e0;
    }

    // 3. Upsert
    if (allChunks.length > 0) {
      const u0 = performance.now();
      await this.adapter.upsert(allChunks);
      upsertMs = performance.now() - u0;
    }

    return {
      documents: documents.length,
      chunks: allChunks.length,
      trace: {
        chunkingMs,
        embeddingMs,
        upsertMs,
        totalMs: performance.now() - t0,
      },
    };
  }

  /**
   * Search — the main entry point. Routes the query through the appropriate
   * strategy and returns results plus a full execution trace.
   *
   * Two-stage pipeline (the production pattern from Turbopuffer / Vespa /
   * Cohere Rerank):
   *
   *   1. **Candidate generation** — cheap, recall-oriented.
   *      When reranking is enabled, we ignore the strategy decision for
   *      retrieval purposes and pull a wide pool from BOTH vector and
   *      keyword in parallel, then dedupe. The strategy decision still
   *      drives whether we rerank at all (and is recorded in the trace
   *      for explainability), but the candidate pool is multi-source so
   *      the reranker has the right doc available regardless of which
   *      retriever found it.
   *      When reranking is disabled, the strategy decision drives a
   *      single-retriever lookup directly — no point paying for a pool
   *      we won't re-score.
   *
   *   2. **Reranking** — precision.
   *      Cross-encoder reads (query, candidate) pairs end-to-end and
   *      produces the final top-K. This is the source of truth for the
   *      ordering, so we trust it with the wider pool.
   */
  async search(req: SearchRequest): Promise<SearchResponse> {
    const tracer = new Tracer(req.query);
    const topK = req.topK ?? 10;

    // If this is an ad-hoc request with documents inline, build a scratch
    // adapter for this query only. This matches the Vercel-y feel: zero
    // setup, useful out of the box.
    let activeAdapter = this.adapter;
    let scratchUsed = false;
    if (req.documents && req.documents.length > 0 && this.autoIndex) {
      const scratch = new InMemoryAdapter();
      const scratchQB = new Augur({
        adapter: scratch,
        embedder: this.embedder,
        chunker: this.chunker,
      });
      await tracer.span("ad-hoc:index", () => scratchQB.index(req.documents!));
      activeAdapter = scratch;
      scratchUsed = true;
    }

    const decision = this.router.decide(req, activeAdapter.capabilities);
    const willRerank = decision.reranked;
    const filter = req.filter;

    let candidates: SearchResult[] = [];

    if (willRerank) {
      // Stage 1: pull a wide multi-source pool. Cap at POOL_PER_SIDE per
      // backend so the cross-encoder doesn't have to score thousands of
      // pairs. 50 each → up to ~100 unique candidates after dedupe.
      candidates = await this.gatherCandidatePool(req, activeAdapter, tracer, filter);
    } else {
      // Stage 1 fast path: no rerank → strategy decision drives a single
      // retrieval call directly to topK.
      candidates = await this.runStrategy(req, decision, activeAdapter, topK, tracer, filter);
    }

    let results = candidates;
    if (willRerank && candidates.length > 0) {
      // Stage 2: cross-encoder picks the final top-K from the wide pool.
      results = await tracer.span(
        "rerank",
        () => this.reranker.rerank(req.query, candidates, topK),
        { reranker: this.reranker.name, candidates: candidates.length }
      );
    } else {
      results = candidates.slice(0, topK);
    }

    // Strip embeddings from results — they're an internal detail and bloat
    // the response payload. Users who need them can re-fetch chunks by ID.
    results = results.map((r) => {
      if (!r.chunk.embedding) return r;
      const { embedding: _embedding, ...rest } = r.chunk;
      return { ...r, chunk: rest };
    });

    const trace = tracer.finish({
      decision,
      candidates: candidates.length,
      adapter: scratchUsed ? `${activeAdapter.name} (ad-hoc)` : activeAdapter.name,
      embeddingModel: this.embedder.name,
    });
    this.traceStore?.push(trace);

    return { results, trace };
  }

  /** Convenience helper to wipe the underlying adapter. */
  async clear(): Promise<void> {
    await this.adapter.clear();
  }

  /**
   * Stage-1 candidate generation when reranking is on.
   *
   * Pulls top-N from each available retriever in parallel, then RRF-fuses
   * the lists into a single ranked candidate pool. The cross-encoder
   * re-scores the top of this pool in stage 2.
   *
   * Why RRF before reranking (and not raw dedupe)?
   *   - Raw dedupe gives the cross-encoder a wider pool but in arbitrary
   *     order. On the bundled eval the local cross-encoder struggled to
   *     re-order the noise — recall went up but NDCG slipped.
   *   - RRF gives a *pre-ranked* pool where docs that scored well on
   *     either side rise to the top. The cross-encoder then refines a
   *     pre-curated list. Production stacks (Cohere, Vespa, Turbopuffer)
   *     all use a fused retrieval pool followed by reranking for
   *     precisely this reason.
   *
   * Pool sizing: 50 per side → up to ~100 unique fused, then we hand the
   * top 30 to the cross-encoder. That's the production sweet spot — wide
   * enough that the right doc is almost always present, narrow enough
   * that the cross-encoder pays for precision, not noise filtering.
   */
  private async gatherCandidatePool(
    req: SearchRequest,
    activeAdapter: VectorAdapter,
    tracer: Tracer,
    filter: Record<string, unknown> | undefined
  ): Promise<SearchResult[]> {
    const POOL_PER_SIDE = 50;
    const RERANK_POOL_CAP = 30;
    const caps = activeAdapter.capabilities;

    const embedding = caps.vector
      ? await tracer.span("embed:query", async () => {
          if (this.embedder.embedQuery) return this.embedder.embedQuery(req.query);
          const [v] = await this.embedder.embed([req.query]);
          return v!;
        })
      : null;

    const filt = filter ? { filter } : {};
    const [vec, kw] = await Promise.all([
      caps.vector && embedding
        ? tracer.span("pool:vector", () =>
            activeAdapter.searchVector({ embedding, topK: POOL_PER_SIDE, ...filt })
          )
        : Promise.resolve<SearchResult[]>([]),
      caps.keyword
        ? tracer.span("pool:keyword", () =>
            activeAdapter.searchKeyword({ query: req.query, topK: POOL_PER_SIDE, ...filt })
          )
        : Promise.resolve<SearchResult[]>([]),
    ]);

    return rrfFuse(vec, kw).slice(0, RERANK_POOL_CAP);
  }

  /**
   * Stage-1 fast path when reranking is off. Strategy decision drives a
   * single retrieval call — no point pulling a wide pool we won't re-score.
   */
  private async runStrategy(
    req: SearchRequest,
    decision: import("./types.js").RoutingDecision,
    activeAdapter: VectorAdapter,
    topK: number,
    tracer: Tracer,
    filter: Record<string, unknown> | undefined
  ): Promise<SearchResult[]> {
    const filt = filter ? { filter } : {};
    if (decision.strategy === "keyword") {
      return tracer.span("search:keyword", () =>
        activeAdapter.searchKeyword({ query: req.query, topK, ...filt })
      );
    }
    const embedding = await tracer.span("embed:query", async () => {
      if (this.embedder.embedQuery) return this.embedder.embedQuery(req.query);
      const [v] = await this.embedder.embed([req.query]);
      return v!;
    });
    if (decision.strategy === "vector" || decision.strategy === "rerank") {
      return tracer.span("search:vector", () =>
        activeAdapter.searchVector({ embedding, topK, ...filt })
      );
    }
    // hybrid — query-aware vector weight, RRF fusion in BaseAdapter.
    const vectorWeight = pickVectorWeight(decision.signals);
    return tracer.span("search:hybrid", () =>
      (activeAdapter.searchHybrid ?? hybridFallback).call(activeAdapter, {
        embedding,
        query: req.query,
        topK,
        vectorWeight,
        ...filt,
      })
    );
  }
}

/**
 * Map query signals to a hybrid vector/keyword weight. The output is the
 * fraction the vector side gets in RRF / score combination; (1 - weight)
 * goes to BM25.
 *
 * Heuristic, no learned weights — a tiny query-aware ramp:
 *
 *   - Quoted phrase or specific identifier present → BM25 carries (0.3).
 *     The user is asking for an exact match; vector tends to dilute.
 *   - Very short query (≤2 tokens) → BM25 leans (0.4). Bi-encoders embed
 *     single terms poorly.
 *   - Long natural-language question (≥6 tokens, no specific tokens) →
 *     vector leans (0.7). Semantic match dominates lexical at length.
 *   - Default → 0.5. Equal mix when the query gives no strong signal.
 *
 * Note: we tried lowering the weight further (0.2) for `hasNegation` queries
 * on the theory that bi-encoders can't read "not"/"without". On the bundled
 * eval it regressed negation NDCG (-0.014) — BM25 alone wasn't ranking the
 * right docs either. The fix probably has to come from a stronger reranker
 * (which is already enabled on negation per the router) rather than a weight
 * tweak. Leaving the slot for when we revisit.
 */
function pickVectorWeight(signals: import("./types.js").QuerySignals): number {
  if (signals.hasQuotedPhrase || signals.hasSpecificTokens || signals.hasCodeLike) return 0.3;
  if (signals.tokens <= 2) return 0.4;
  if (signals.isQuestion && signals.tokens >= 6) return 0.7;
  return 0.5;
}

/**
 * Reciprocal Rank Fusion of two ranked lists into one. The de-facto
 * standard fusion method for hybrid retrieval — k=60 is the canonical
 * value (Cormack 2009). Each side contributes equally; if you want a
 * skew, weight the term inside the loop.
 */
function rrfFuse(
  a: SearchResult[],
  b: SearchResult[],
  k: number = 60
): SearchResult[] {
  const fused = new Map<string, { result: SearchResult; score: number }>();
  a.forEach((r, rank) => {
    fused.set(r.chunk.id, { result: r, score: 1 / (k + rank + 1) });
  });
  b.forEach((r, rank) => {
    const score = 1 / (k + rank + 1);
    const existing = fused.get(r.chunk.id);
    if (existing) existing.score += score;
    else fused.set(r.chunk.id, { result: r, score });
  });
  return Array.from(fused.values())
    .sort((x, y) => y.score - x.score)
    .map(({ result, score }) => ({ ...result, score }));
}

/** Fallback hybrid for adapters that didn't override `searchHybrid`. */
async function hybridFallback(
  this: VectorAdapter,
  opts: {
    embedding: number[];
    query: string;
    topK: number;
    vectorWeight: number;
    filter?: Record<string, unknown>;
  }
): Promise<SearchResult[]> {
  // BaseAdapter provides this. If we got here, the adapter doesn't extend
  // BaseAdapter. The router should have steered away from hybrid in that
  // case, but if it didn't, do the cheapest reasonable thing: vector only.
  return this.searchVector({ embedding: opts.embedding, topK: opts.topK, ...(opts.filter ? { filter: opts.filter } : {}) });
}

/** Re-export for convenience: `import { FixedSizeChunker } from '@augur/core'`. */
export { FixedSizeChunker, SentenceChunker, SemanticChunker };
