import type { VectorAdapter } from "./adapters/adapter.js";
import { InMemoryAdapter } from "./adapters/in-memory.js";
import { type Chunker, FixedSizeChunker, SentenceChunker, SemanticChunker, chunkDocument } from "./chunking/chunker.js";
import { type Embedder, HashEmbedder } from "./embeddings/embedder.js";
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
  /** Storage adapter. Defaults to InMemoryAdapter (zero deps). */
  adapter?: VectorAdapter;
  /** Embedder used for both indexing and querying. Defaults to HashEmbedder. */
  embedder?: Embedder;
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
 * The "Stripe-feel" design constraint shows up here:
 *
 *   const augr = new Augur();              // works
 *   await augr.index([{ id: "1", content: "..." }]);
 *   const { results, trace } = await augr.search({ query: "hello" });
 *
 * No config files, no SDKs, no API keys. The defaults are real defaults that
 * produce real answers. Swapping in production-grade pieces (OpenAIEmbedder,
 * PineconeAdapter, CohereReranker) is a constructor argument away.
 *
 * Why everything is constructor-injected:
 * - Testability: every component is mockable in isolation.
 * - Forward compatibility: when MLRouter ships, `new Augur({ router: new MLRouter(...) })`
 *   is the entire migration path.
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

  constructor(opts: AugurOptions = {}) {
    this.adapter = opts.adapter ?? new InMemoryAdapter();
    this.embedder = opts.embedder ?? new HashEmbedder();
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
      // Let stateful embedders (TfIdfEmbedder etc) ingest the corpus before
      // embedding, so query-time IDFs reflect indexed content.
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

    let candidates: SearchResult[] = [];
    // Pull a wider net when reranking — precision is the reranker's job.
    const expandedTopK = decision.reranked ? Math.max(topK * 3, 30) : topK;

    if (decision.strategy === "vector") {
      const embedding = await tracer.span("embed:query", async () => {
        if (this.embedder.embedQuery) return this.embedder.embedQuery(req.query);
        const [v] = await this.embedder.embed([req.query]);
        return v!;
      });
      candidates = await tracer.span("search:vector", () =>
        activeAdapter.searchVector({ embedding, topK: expandedTopK, ...(req.filter ? { filter: req.filter } : {}) })
      );
    } else if (decision.strategy === "keyword") {
      candidates = await tracer.span("search:keyword", () =>
        activeAdapter.searchKeyword({ query: req.query, topK: expandedTopK, ...(req.filter ? { filter: req.filter } : {}) })
      );
    } else if (decision.strategy === "hybrid") {
      const embedding = await tracer.span("embed:query", async () => {
        if (this.embedder.embedQuery) return this.embedder.embedQuery(req.query);
        const [v] = await this.embedder.embed([req.query]);
        return v!;
      });
      // Query-aware hybrid weight: short / specific queries lean BM25; long
      // natural-language queries lean vector. Production hybrid systems (Vespa,
      // Pinecone hybrid) all do some version of this — a fixed 0.5/0.5 mix
      // under-weights whichever side is wrong for the current query shape.
      const vectorWeight = pickVectorWeight(decision.signals);
      candidates = await tracer.span("search:hybrid", () =>
        (activeAdapter.searchHybrid ?? hybridFallback).call(activeAdapter, {
          embedding,
          query: req.query,
          topK: expandedTopK,
          vectorWeight,
          ...(req.filter ? { filter: req.filter } : {}),
        })
      );
    } else if (decision.strategy === "rerank") {
      // "rerank" as a top-level strategy means "do vector then rerank, period".
      const embedding = await tracer.span("embed:query", async () => {
        if (this.embedder.embedQuery) return this.embedder.embedQuery(req.query);
        const [v] = await this.embedder.embed([req.query]);
        return v!;
      });
      candidates = await tracer.span("search:vector", () =>
        activeAdapter.searchVector({ embedding, topK: expandedTopK, ...(req.filter ? { filter: req.filter } : {}) })
      );
    }

    let results = candidates;
    if (decision.reranked && candidates.length > 0) {
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
 */
function pickVectorWeight(signals: import("./types.js").QuerySignals): number {
  if (signals.hasQuotedPhrase || signals.hasSpecificTokens || signals.hasCodeLike) return 0.3;
  if (signals.tokens <= 2) return 0.4;
  if (signals.isQuestion && signals.tokens >= 6) return 0.7;
  return 0.5;
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
