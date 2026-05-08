import type { Chunk, SearchResult } from "../types.js";

/**
 * Storage adapter interface.
 *
 * Why this interface looks the way it does:
 *
 * 1) `upsert` takes already-chunked, optionally-embedded chunks. Embedding
 *    is the orchestrator's job, not the adapter's, because:
 *    - Different adapters have different ideas about who owns embeddings
 *      (Pinecone wants you to bring them; pgvector accepts either).
 *    - Centralizing embedding lets us reuse a single Embedder for both
 *      indexing and querying — critical for consistent vector spaces.
 *
 * 2) Search is split into vector / keyword / both. The adapter is allowed
 *    to say "I don't support keyword" by returning an empty array — the
 *    routing engine handles fallbacks. Adapters that natively support
 *    hybrid (e.g. some pgvector setups) can override `searchHybrid`.
 *
 * 3) `delete` and `count` are minimal but enable observability to show
 *    real numbers. We deliberately do NOT mandate transactions, schemas,
 *    or migrations — those belong to the underlying store.
 */
export interface VectorAdapter {
  /** Stable name surfaced in traces and observability output. */
  readonly name: string;
  /** Whether this adapter natively supports keyword search. */
  readonly capabilities: AdapterCapabilities;

  /** Insert or update chunks. Embeddings should be present unless `capabilities.computesEmbeddings` is true. */
  upsert(chunks: Chunk[]): Promise<void>;

  /** Vector similarity search. */
  searchVector(opts: VectorSearchOpts): Promise<SearchResult[]>;

  /** Keyword (BM25-ish) search. Adapters without native support may throw. */
  searchKeyword(opts: KeywordSearchOpts): Promise<SearchResult[]>;

  /** Optional native hybrid path. Default implementation is provided in `BaseAdapter`. */
  searchHybrid?(opts: HybridSearchOpts): Promise<SearchResult[]>;

  /** Delete chunks by ID. */
  delete(ids: string[]): Promise<void>;

  /** Total chunk count — surfaced via the `/health` endpoint and traces. */
  count(): Promise<number>;

  /** Clear all data. */
  clear(): Promise<void>;
}

export interface AdapterCapabilities {
  vector: boolean;
  keyword: boolean;
  hybrid: boolean;
  /** Adapter computes its own embeddings (e.g. some managed services). */
  computesEmbeddings: boolean;
  /** Adapter supports metadata filters. */
  filtering: boolean;
}

export interface VectorSearchOpts {
  embedding: number[];
  topK: number;
  filter?: Record<string, unknown>;
}

export interface KeywordSearchOpts {
  query: string;
  topK: number;
  filter?: Record<string, unknown>;
}

export interface HybridSearchOpts extends VectorSearchOpts, KeywordSearchOpts {
  /** Weight for the vector score, 0..1. The keyword weight is `1 - vectorWeight`. */
  vectorWeight: number;
}

/**
 * BaseAdapter provides default implementations that work for most stores.
 *
 * Subclasses only need to implement vector + keyword and they get hybrid
 * via reciprocal rank fusion (RRF) for free. RRF beats weighted-sum-of-scores
 * in practice because it doesn't require score normalization across
 * heterogeneous backends.
 */
export abstract class BaseAdapter implements VectorAdapter {
  abstract readonly name: string;
  abstract readonly capabilities: AdapterCapabilities;
  abstract upsert(chunks: Chunk[]): Promise<void>;
  abstract searchVector(opts: VectorSearchOpts): Promise<SearchResult[]>;
  abstract searchKeyword(opts: KeywordSearchOpts): Promise<SearchResult[]>;
  abstract delete(ids: string[]): Promise<void>;
  abstract count(): Promise<number>;
  abstract clear(): Promise<void>;

  /**
   * Default hybrid: reciprocal rank fusion of vector + keyword.
   *
   * RRF score for a doc = vectorWeight * 1/(k+rank_v) + (1-vectorWeight) * 1/(k+rank_k)
   * where k is a smoothing constant (60 is the de-facto standard).
   */
  async searchHybrid(opts: HybridSearchOpts): Promise<SearchResult[]> {
    const k = 60;
    const expandedTopK = opts.topK * 4; // pull more candidates so fusion has signal

    const [vec, kw] = await Promise.all([
      this.searchVector({ embedding: opts.embedding, topK: expandedTopK, filter: opts.filter }),
      this.searchKeyword({ query: opts.query, topK: expandedTopK, filter: opts.filter }),
    ]);

    const fused = new Map<string, { result: SearchResult; score: number }>();

    vec.forEach((r, rank) => {
      const score = opts.vectorWeight * (1 / (k + rank + 1));
      fused.set(r.chunk.id, {
        result: { ...r, rawScores: { ...r.rawScores, vector: r.score } },
        score,
      });
    });
    kw.forEach((r, rank) => {
      const score = (1 - opts.vectorWeight) * (1 / (k + rank + 1));
      const existing = fused.get(r.chunk.id);
      if (existing) {
        existing.score += score;
        existing.result.rawScores = {
          ...existing.result.rawScores,
          keyword: r.score,
        };
      } else {
        fused.set(r.chunk.id, {
          result: { ...r, rawScores: { ...r.rawScores, keyword: r.score } },
          score,
        });
      }
    });

    return Array.from(fused.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, opts.topK)
      .map(({ result, score }) => ({ ...result, score }));
  }
}
