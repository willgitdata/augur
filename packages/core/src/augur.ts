import type { VectorAdapter } from "./adapters/adapter.js";
import { InMemoryAdapter } from "./adapters/in-memory.js";
import {
  type AsyncChunker,
  type Chunker,
  SentenceChunker,
  chunkDocument,
} from "./chunking/chunker.js";
import type { Embedder } from "./embeddings/embedder.js";
import { fingerprintDocs } from "./fingerprint.js";
import {
  adaptWeightByConfidence,
  composeFilter,
  pickVectorWeight,
  weightedRrfFuse,
} from "./fusion.js";
import { Tracer, TraceStore } from "./observability/tracer.js";
import { type Reranker } from "./reranking/reranker.js";
import { HeuristicRouter, type Router } from "./routing/router.js";
import { detectLanguage } from "./routing/signals.js";
import type {
  Document,
  IndexResponse,
  RoutingDecision,
  SearchRequest,
  SearchResponse,
  SearchResult,
} from "./types.js";

export interface AugurOptions {
  /**
   * Embedder used for both indexing and querying. **Required**. Use
   * `LocalEmbedder` for a fully on-device default, or implement the
   * `Embedder` interface against your provider's SDK (see docs/examples.md).
   */
  embedder: Embedder;
  /** Storage adapter. Defaults to InMemoryAdapter. */
  adapter?: VectorAdapter;
  /** Chunker used during `index()`. Defaults to SentenceChunker. */
  chunker?: Chunker | AsyncChunker;
  /** Routing engine. Defaults to HeuristicRouter. */
  router?: Router;
  /**
   * Reranker. **No default** — bare retrieval if omitted. For the
   * recommended cross-encoder voting on every query (the "auto = best"
   * path that produces the headline NDCG@10), pass
   * `new LocalReranker()` (zero-API-key 22 MB ONNX cross-encoder), or
   * any hosted provider implementing the one-method `Reranker`
   * interface (Cohere, Voyage, Jina — see docs/examples.md).
   */
  reranker?: Reranker;
  /** Optional trace store — when provided, every search trace is captured. */
  traceStore?: TraceStore;
  /**
   * Auto-indexing: if a search request includes `documents`, automatically
   * index them in a scratch in-memory adapter for that request only.
   * Defaults to true. Set false to require explicit `index()` calls.
   */
  autoIndexAdHocDocuments?: boolean;
  /**
   * LRU cache size for scratch adapters built from ad-hoc `req.documents`.
   * Repeat searches over the same documents reuse the cached scratch
   * adapter, skipping re-chunking + re-embedding. Defaults to 8.
   * Set to 0 to disable caching (always rebuild — useful when ad-hoc
   * corpora rotate every request).
   */
  adHocCacheSize?: number;
  /**
   * When the query's detected language is non-English, automatically
   * filter the candidate pool to chunks tagged with that language.
   * Chunks are auto-tagged at index time from `metadata.lang` if present,
   * otherwise via Unicode-script detection on content. Soft fallback:
   * if the filter empties the pool, search retries without it so users
   * with monolingual corpora still get answers.
   *
   * **Default: false** — opt-in. The right default depends on your
   * corpus shape:
   *   - Turn ON when you have language-localized canonical answers
   *     (e.g. one Japanese doc and one English doc per topic). Japanese
   *     queries should get Japanese results, English queries English.
   *   - Leave OFF when canonical answers may be cross-language (e.g. a
   *     primarily-English knowledge base queried in many languages).
   *     Hard-filtering Japanese queries would exclude the English
   *     canonical answer even when it's the most relevant document.
   *
   * Always opt-in is the safer default; production users who know their
   * corpus has localized content turn it on once.
   */
  autoLanguageFilter?: boolean;
}

/**
 * Augur — retrieval orchestration entry point. Every component is
 * constructor-injected for swapability; `embedder` is the only required
 * field. See docs/examples.md for hosted-provider Embedder snippets.
 *
 *   const augr = new Augur({ embedder: new LocalEmbedder() });
 *   await augr.index([{ id: "1", content: "..." }]);
 *   const { results, trace } = await augr.search({ query: "hello" });
 */
export class Augur {
  readonly adapter: VectorAdapter;
  readonly embedder: Embedder;
  readonly chunker: Chunker | AsyncChunker;
  readonly router: Router;
  readonly reranker: Reranker | null;
  readonly traceStore?: TraceStore;
  private autoIndex: boolean;
  private autoLanguageFilter: boolean;
  /**
   * Bounded LRU of scratch adapters built from ad-hoc `req.documents`,
   * keyed by a deterministic fingerprint of (id+content). Lets repeat
   * searches over the same docs skip re-chunking + re-embedding — the
   * dominant cost for any non-trivial corpus. Bounded at 8 to avoid
   * unbounded memory growth in long-lived servers; tune via
   * `adHocCacheSize` if you have a different hit-rate profile.
   */
  private adHocCache = new Map<string, VectorAdapter>();
  private adHocCacheSize: number;

  constructor(opts: AugurOptions) {
    if (!opts || !opts.embedder) {
      throw new Error(
        "Augur: `embedder` is required. Use `new LocalEmbedder()` for an on-device " +
          "default, or implement the Embedder interface against your provider " +
          "(see docs/examples.md)."
      );
    }
    this.embedder = opts.embedder;
    this.adapter = opts.adapter ?? new InMemoryAdapter();
    this.chunker = opts.chunker ?? new SentenceChunker();
    this.router = opts.router ?? new HeuristicRouter();
    // No default reranker. The previous default (HeuristicReranker:
    // token overlap + proximity) gave fake "yes I rerank" comfort while
    // doing close to nothing — users on the auto path got bare
    // retrieval-shaped output that pretended it was reranked. Now an
    // explicit reranker is required for the cross-encoder voting step
    // to fire. Pass `new LocalReranker()` (zero-API-key cross-encoder
    // ONNX) for the headline accuracy mode, or any provider's reranker
    // implementing the one-method interface.
    this.reranker = opts.reranker ?? null;
    if (opts.traceStore) this.traceStore = opts.traceStore;
    this.autoIndex = opts.autoIndexAdHocDocuments ?? true;
    this.autoLanguageFilter = opts.autoLanguageFilter ?? false;
    this.adHocCacheSize = Math.max(0, opts.adHocCacheSize ?? 8);
  }

  /**
   * Index a batch of documents. Returns timing breakdown for observability.
   */
  async index(documents: Document[]): Promise<IndexResponse> {
    const t0 = performance.now();
    let chunkingMs = 0;
    let embeddingMs = 0;
    let upsertMs = 0;

    // 1. Chunk + auto-tag with language. The lang tag drives the
    //    language-aware filter at search time. Honor user-supplied
    //    metadata.lang on the source doc; otherwise detect from content.
    const c0 = performance.now();
    const perDoc = await Promise.all(
      documents.map(async (d) => {
        const chunks = await chunkDocument(this.chunker, d);
        const lang =
          (d.metadata?.lang as string | undefined) ?? detectLanguage(d.content);
        for (const ch of chunks) {
          ch.metadata = { ...ch.metadata, lang };
        }
        return chunks;
      })
    );
    const allChunks = perDoc.flat();
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
   * Search — two-stage pipeline (the Turbopuffer/Vespa/Cohere Rerank
   * pattern):
   *   1. Recall: when reranking is on, gather a wide pool from both vector
   *      and keyword retrievers in parallel and RRF-fuse. When off, the
   *      strategy decision drives a single retrieval call.
   *   2. Precision: cross-encoder re-scores the top of the fused pool and
   *      produces the final top-K.
   */
  async search(req: SearchRequest): Promise<SearchResponse> {
    const tracer = new Tracer(req.query);
    const topK = req.topK ?? 10;

    // If this is an ad-hoc request with documents inline, build a scratch
    // adapter for this query only. This matches the Vercel-y feel: zero
    // setup, useful out of the box. Cache by (id, content) fingerprint so
    // repeat searches over the same documents skip re-chunking +
    // re-embedding — the dominant cost for any non-trivial corpus.
    let activeAdapter = this.adapter;
    let scratchUsed = false;
    let scratchCacheHit = false;
    if (req.documents && req.documents.length > 0 && this.autoIndex) {
      const cacheKey = this.adHocCacheSize > 0 ? fingerprintDocs(req.documents) : null;
      const cached = cacheKey ? this.adHocCache.get(cacheKey) : undefined;
      if (cached) {
        // Move to most-recently-used position (LRU bookkeeping).
        this.adHocCache.delete(cacheKey!);
        this.adHocCache.set(cacheKey!, cached);
        activeAdapter = cached;
        scratchCacheHit = true;
      } else {
        const scratch = new InMemoryAdapter();
        const scratchQB = new Augur({
          adapter: scratch,
          embedder: this.embedder,
          chunker: this.chunker,
        });
        await tracer.span("ad-hoc:index", () => scratchQB.index(req.documents!));
        activeAdapter = scratch;
        if (cacheKey) {
          // Evict oldest if at capacity (Map iteration order = insertion order).
          if (this.adHocCache.size >= this.adHocCacheSize) {
            const oldest = this.adHocCache.keys().next().value;
            if (oldest !== undefined) this.adHocCache.delete(oldest);
          }
          this.adHocCache.set(cacheKey, scratch);
        }
      }
      scratchUsed = true;
    }

    const decision = this.router.decide(
      req,
      activeAdapter.capabilities,
      this.reranker !== null
    );
    const willRerank = decision.reranked;
    const userFilter = req.filter;

    // Compose the active filter from user-supplied filter + the
    // auto-language filter. The auto-language filter only applies when
    // (a) it's enabled, (b) the query is non-English, and (c) the user
    // didn't already pin `lang` themselves.
    const autoLang =
      this.autoLanguageFilter &&
      decision.signals.language !== "en" &&
      !(userFilter && "lang" in userFilter)
        ? decision.signals.language
        : null;
    const filter = composeFilter(userFilter, autoLang);

    let candidates = await this.retrieve(
      req, decision, activeAdapter, topK, tracer, filter, willRerank
    );
    let langFilterDropped = false;
    // Soft fallback: if the language filter emptied the result, retry
    // without it. Better to surface the closest match in another
    // language than to return nothing at all.
    if (autoLang && candidates.length === 0) {
      langFilterDropped = true;
      candidates = await this.retrieve(
        req, decision, activeAdapter, topK, tracer, userFilter, willRerank
      );
    }

    let results = candidates;
    const reranker = this.reranker;
    if (willRerank && reranker && candidates.length > 0) {
      results = await tracer.span(
        "rerank",
        () => reranker.rerank(req.query, candidates, topK),
        { reranker: reranker.name, candidates: candidates.length }
      );
    } else {
      // No reranker configured (or routing decision said skip) — return
      // the raw retrieval order, sliced to topK.
      results = candidates.slice(0, topK);
    }

    // Optional confidence floor: drop results below `req.minScore`. Useful
    // when "no answer" is a better signal to the LLM than "noisy answer".
    if (typeof req.minScore === "number") {
      results = results.filter((r) => r.score >= req.minScore!);
    }

    // Strip embeddings from results — internal detail, bloats the payload.
    results = results.map((r) => {
      if (!r.chunk.embedding) return r;
      const { embedding: _embedding, ...rest } = r.chunk;
      return { ...r, chunk: rest };
    });

    const trace = tracer.finish({
      decision,
      candidates: candidates.length,
      adapter: activeAdapter.name,
      embeddingModel: this.embedder.name,
      ...(scratchUsed ? { adHoc: true } : {}),
      ...(scratchCacheHit ? { adHocCacheHit: true } : {}),
      ...(autoLang ? { autoLanguageFilter: autoLang } : {}),
      ...(langFilterDropped ? { autoLanguageFilterDropped: true } : {}),
    });
    this.traceStore?.push(trace);

    return { results, trace };
  }

  /** Single internal helper — picks the right stage-1 path based on willRerank. */
  private retrieve(
    req: SearchRequest,
    decision: RoutingDecision,
    activeAdapter: VectorAdapter,
    topK: number,
    tracer: Tracer,
    filter: Record<string, unknown> | undefined,
    willRerank: boolean
  ): Promise<SearchResult[]> {
    return willRerank
      ? this.gatherCandidatePool(req, decision, activeAdapter, tracer, filter)
      : this.runStrategy(req, decision, activeAdapter, topK, tracer, filter);
  }

  /** Convenience helper to wipe the underlying adapter. */
  async clear(): Promise<void> {
    await this.adapter.clear();
  }

  /**
   * Recall stage. Pulls top-N from each retriever in parallel, RRF-fuses
   * the two lists with an adaptive weight (query-signal prior shifted by
   * observed retrieval confidence), and hands the top of the fused pool
   * to the cross-encoder. Pool sizing (50/side → top 30) is the
   * production sweet spot — wide enough recall, narrow enough that the
   * reranker is doing precision and not noise filtering.
   */
  private async gatherCandidatePool(
    req: SearchRequest,
    decision: RoutingDecision,
    activeAdapter: VectorAdapter,
    tracer: Tracer,
    filter: Record<string, unknown> | undefined
  ): Promise<SearchResult[]> {
    const POOL_PER_SIDE = 50;
    const RERANK_POOL_CAP = 30;
    const caps = activeAdapter.capabilities;
    const filt = filter ? { filter } : {};
    const embedding = caps.vector ? await this.embedQuery(req.query, tracer) : null;

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

    const baseWeight = pickVectorWeight(decision.signals);
    const adaptiveWeight = adaptWeightByConfidence(baseWeight, vec, kw);
    return weightedRrfFuse(vec, kw, adaptiveWeight).slice(0, RERANK_POOL_CAP);
  }

  /** Single-retriever fast path when reranking is off. */
  private async runStrategy(
    req: SearchRequest,
    decision: RoutingDecision,
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
    const embedding = await this.embedQuery(req.query, tracer);
    if (decision.strategy === "vector" || decision.strategy === "rerank") {
      return tracer.span("search:vector", () =>
        activeAdapter.searchVector({ embedding, topK, ...filt })
      );
    }
    // hybrid: query-aware vector weight, RRF fusion in BaseAdapter (or
    // the adapter's vector path when it doesn't extend BaseAdapter).
    const vectorWeight = pickVectorWeight(decision.signals);
    return tracer.span("search:hybrid", () => {
      if (activeAdapter.searchHybrid) {
        return activeAdapter.searchHybrid({
          embedding,
          query: req.query,
          topK,
          vectorWeight,
          ...filt,
        });
      }
      return activeAdapter.searchVector({ embedding, topK, ...filt });
    });
  }

  /** One-shot query embedding with span tracing — used by both retrieval paths. */
  private embedQuery(query: string, tracer: Tracer): Promise<number[]> {
    return tracer.span("embed:query", async () => {
      if (this.embedder.embedQuery) return this.embedder.embedQuery(query);
      const [v] = await this.embedder.embed([query]);
      return v!;
    });
  }
}

