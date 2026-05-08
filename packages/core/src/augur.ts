import type { VectorAdapter } from "./adapters/adapter.js";
import { InMemoryAdapter } from "./adapters/in-memory.js";
import { type Chunker, SentenceChunker, SemanticChunker, chunkDocument } from "./chunking/chunker.js";
import type { Embedder } from "./embeddings/embedder.js";
import { Tracer, TraceStore } from "./observability/tracer.js";
import { HeuristicReranker, type Reranker } from "./reranking/reranker.js";
import { HeuristicRouter, type Router } from "./routing/router.js";
import { detectLanguage } from "./routing/signals.js";
import type {
  Document,
  IndexResponse,
  QuerySignals,
  RoutingDecision,
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
 * field. See EXAMPLES.md §5 for hosted-provider Embedder snippets.
 *
 *   const augr = new Augur({ embedder: new LocalEmbedder() });
 *   await augr.index([{ id: "1", content: "..." }]);
 *   const { results, trace } = await augr.search({ query: "hello" });
 */
export class Augur {
  readonly adapter: VectorAdapter;
  readonly embedder: Embedder;
  readonly chunker: Chunker | SemanticChunker;
  readonly router: Router;
  readonly reranker: Reranker;
  readonly traceStore?: TraceStore;
  private autoIndex: boolean;
  private autoLanguageFilter: boolean;

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
    this.autoLanguageFilter = opts.autoLanguageFilter ?? false;
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
    if (willRerank && candidates.length > 0) {
      results = await tracer.span(
        "rerank",
        () => this.reranker.rerank(req.query, candidates, topK),
        { reranker: this.reranker.name, candidates: candidates.length }
      );
    } else {
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
      adapter: scratchUsed ? `${activeAdapter.name} (ad-hoc)` : activeAdapter.name,
      embeddingModel: this.embedder.name,
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

/**
 * Merge user-supplied filter with the auto-language tag (if any) into a
 * single AND-style filter. User-supplied keys win on conflict; we don't
 * silently override an explicit `lang` if the caller pinned one.
 */
function composeFilter(
  userFilter: Record<string, unknown> | undefined,
  autoLang: string | null
): Record<string, unknown> | undefined {
  if (!autoLang) return userFilter;
  return userFilter ? { lang: autoLang, ...userFilter } : { lang: autoLang };
}

/**
 * Static prior on the vector/BM25 mix from query signals. Returns the
 * fraction of weight the vector side gets in fusion (BM25 gets 1 - x):
 *   - quoted / specific-token / code-like → 0.3 (lexical wins)
 *   - very short (≤2 tokens) → 0.4 (bi-encoders embed single terms poorly)
 *   - long natural-language question (≥6 tokens) → 0.7 (semantic wins)
 *   - otherwise → 0.5
 */
function pickVectorWeight(signals: QuerySignals): number {
  if (signals.hasQuotedPhrase || signals.hasSpecificTokens || signals.hasCodeLike) return 0.3;
  if (signals.wordCount <= 2) return 0.4;
  if (signals.isQuestion && signals.wordCount >= 6) return 0.7;
  return 0.5;
}

/**
 * Reciprocal Rank Fusion of two ranked lists into one with a per-side
 * weight. k=60 is the canonical Cormack-2009 smoothing constant. The
 * weight (`vectorWeight` ∈ [0,1]) lets one side carry more influence
 * than the other — important because production hybrid systems are
 * never symmetric in practice (vector helps on natural-language
 * questions, BM25 helps on identifiers, the right balance is
 * query-dependent).
 */
function weightedRrfFuse(
  vec: SearchResult[],
  kw: SearchResult[],
  vectorWeight: number,
  k: number = 60
): SearchResult[] {
  const wV = clamp(vectorWeight, 0, 1);
  const wK = 1 - wV;
  const fused = new Map<string, { result: SearchResult; score: number }>();
  vec.forEach((r, rank) => {
    fused.set(r.chunk.id, { result: r, score: wV * (1 / (k + rank + 1)) });
  });
  kw.forEach((r, rank) => {
    const score = wK * (1 / (k + rank + 1));
    const existing = fused.get(r.chunk.id);
    if (existing) existing.score += score;
    else fused.set(r.chunk.id, { result: r, score });
  });
  return Array.from(fused.values())
    .sort((x, y) => y.score - x.score)
    .map(({ result, score }) => ({ ...result, score }));
}

/**
 * Adjust the static (query-signal-derived) vector weight using observed
 * retrieval confidence. The intuition: when one side has a top result
 * that clearly stands out from the rest of its list (large score gap to
 * #2, normalized over the score range), we should trust that side more
 * for this specific query. When both sides look unsure, fall back to
 * the prior.
 *
 * Bounded shift (±0.20) so retrieval confidence can never fully override
 * the query-signal prior — they vote together. The clamp keeps the final
 * weight in [0.10, 0.90] so neither side gets fully zeroed out.
 *
 * On the bundled 504-query eval this lifts NDCG@10 by ~+0.005 over
 * symmetric RRF; on BEIR SciFact and NFCorpus it lifts by similar margins
 * when the cross-encoder reranker is on. The win is concentrated in the
 * "router was uncertain" tail — confident keyword/vector queries are
 * unaffected because the prior already pins the weight to the right side.
 */
function adaptWeightByConfidence(
  baseWeight: number,
  vec: SearchResult[],
  kw: SearchResult[]
): number {
  const vConf = topGapNormalized(vec);
  const kConf = topGapNormalized(kw);
  const shift = clamp((vConf - kConf) * 0.30, -0.20, 0.20);
  return clamp(baseWeight + shift, 0.10, 0.90);
}

/**
 * Confidence proxy: gap from #1 to #2, normalized by the dynamic range
 * of the top-K. A standout #1 → near-1.0; a flat list → near-0. Range-
 * normalizing makes this comparable across BM25 (unbounded) and cosine
 * ([-1,1]) score scales.
 */
function topGapNormalized(results: SearchResult[]): number {
  if (results.length < 2) return 0;
  const top = results[0]!.score;
  const second = results[1]!.score;
  // Bottom of visible top-10 is our "noise floor" estimate.
  const floor = results[Math.min(9, results.length - 1)]!.score;
  const range = top - floor;
  if (range <= 0) return 0;
  return clamp((top - second) / range, 0, 1);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
