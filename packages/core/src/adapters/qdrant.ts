import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type HybridSearchOpts,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";
import type { SparseEncoder, SparseVector } from "./sparse.js";

/**
 * QdrantAdapter — adapter against Qdrant's REST API.
 *
 * Two modes (parallel to PineconeAdapter):
 *
 * 1. **Dense-only (default).** Vector search via the named "dense"
 *    vector. `capabilities.keyword` and `capabilities.hybrid` are
 *    false; the router falls back to vector.
 *
 * 2. **Sparse-dense hybrid.** If you pass a `sparseEncoder`, the
 *    adapter sends Qdrant's native sparse+dense Query API (Qdrant ≥
 *    1.10). Each point gets a "dense" vector and a "sparse" vector;
 *    `searchHybrid` does a fusion-side query with RRF or DBSF. Pure
 *    sparse-only keyword search is also supported in this mode via
 *    `searchKeyword` — Qdrant has a real sparse-only endpoint, unlike
 *    Pinecone.
 *
 * **Collection schema.** Augur does not create the collection for you.
 * Qdrant expects a one-time setup like:
 *
 *   PUT /collections/<name>
 *   {
 *     "vectors": { "dense": { "size": 384, "distance": "Cosine" } },
 *     "sparse_vectors": { "sparse": {} }   // only if you'll use sparse
 *   }
 *
 * No depednency on `@qdrant/js-client-rest` — this implementation uses
 * the documented REST shape directly so `@augur-rag/core` stays
 * zero-dep at runtime. Users with the official SDK already in their
 * tree can write a 20-line wrapper of the same shape and pass that
 * instead.
 */
export interface QdrantAdapterOptions {
  /** Base URL of the Qdrant instance (e.g. `https://xyz.qdrant.cloud:6333`). */
  url: string;
  /** Collection name. The collection must already exist. */
  collection: string;
  /** Optional API key. Sent as the `api-key` header. */
  apiKey?: string;
  /**
   * Named-vector name to use for dense embeddings. Default `"dense"`.
   * Override if your collection schema uses a different name.
   */
  denseVectorName?: string;
  /**
   * Named-vector name to use for sparse embeddings (only used when
   * `sparseEncoder` is set). Default `"sparse"`.
   */
  sparseVectorName?: string;
  /**
   * Optional sparse encoder. When set, the adapter computes a sparse
   * vector per chunk on upsert and supports BM25-style keyword search
   * + sparse-dense hybrid via Qdrant's Query API. Capabilities flip to
   * include keyword + hybrid in that case.
   */
  sparseEncoder?: SparseEncoder;
}

export class QdrantAdapter extends BaseAdapter {
  readonly name = "qdrant";
  readonly capabilities: AdapterCapabilities;

  private url: string;
  private collection: string;
  private apiKey: string | undefined;
  private denseName: string;
  private sparseName: string;
  private sparseEncoder: SparseEncoder | null;
  private sparseFitted = false;

  constructor(opts: QdrantAdapterOptions) {
    super();
    this.url = opts.url.replace(/\/$/, "");
    this.collection = opts.collection;
    this.apiKey = opts.apiKey;
    this.denseName = opts.denseVectorName ?? "dense";
    this.sparseName = opts.sparseVectorName ?? "sparse";
    this.sparseEncoder = opts.sparseEncoder ?? null;
    this.capabilities = {
      vector: true,
      keyword: this.sparseEncoder !== null,
      hybrid: this.sparseEncoder !== null,
      computesEmbeddings: false,
      filtering: true,
    };
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    if (chunks.length === 0) return;
    if (this.sparseEncoder) this.ensureSparseFitted(chunks);

    const points = chunks.map((c) => {
      if (!c.embedding) {
        throw new Error(`QdrantAdapter: chunk ${c.id} has no embedding`);
      }
      // Qdrant's "vector" field accepts a map { <vector_name>: <vector> }
      // when the collection has named vectors.
      const vector: Record<string, number[] | SparseVector> = {
        [this.denseName]: c.embedding,
      };
      if (this.sparseEncoder) {
        const sv = this.sparseEncoder.encode(c.content);
        if (sv.indices.length > 0) {
          vector[this.sparseName] = sv;
        }
      }
      return {
        id: c.id,
        vector,
        payload: {
          documentId: c.documentId,
          content: c.content,
          index: c.index,
          ...(c.metadata ?? {}),
        },
      };
    });

    const res = await this.fetch(
      `/collections/${this.collection}/points`,
      "PUT",
      { points }
    );
    if (!res.ok) throw new Error(`Qdrant upsert failed: ${await res.text()}`);
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    const body: Record<string, unknown> = {
      query: opts.embedding,
      using: this.denseName,
      limit: opts.topK,
      with_payload: true,
    };
    if (opts.filter) body.filter = qdrantFilter(opts.filter);
    const res = await this.fetch(
      `/collections/${this.collection}/points/query`,
      "POST",
      body
    );
    if (!res.ok) throw new Error(`Qdrant query failed: ${await res.text()}`);
    return parseQdrantResults(await res.json());
  }

  async searchKeyword(opts: KeywordSearchOpts): Promise<SearchResult[]> {
    if (!this.sparseEncoder) {
      throw new Error(
        "QdrantAdapter: keyword search requires a sparseEncoder. " +
          "Pass `sparseEncoder: new BM25SparseEncoder()` (or any SparseEncoder) " +
          "to enable BM25-style keyword retrieval."
      );
    }
    if (this.sparseEncoder.isFitted && !this.sparseEncoder.isFitted()) {
      // No corpus to query against — caller should have indexed first.
      return [];
    }
    const sparse = this.sparseEncoder.encode(opts.query);
    if (sparse.indices.length === 0) return [];

    const body: Record<string, unknown> = {
      query: sparse,
      using: this.sparseName,
      limit: opts.topK,
      with_payload: true,
    };
    if (opts.filter) body.filter = qdrantFilter(opts.filter);
    const res = await this.fetch(
      `/collections/${this.collection}/points/query`,
      "POST",
      body
    );
    if (!res.ok) throw new Error(`Qdrant keyword query failed: ${await res.text()}`);
    return parseQdrantResults(await res.json());
  }

  /**
   * Sparse-dense hybrid via Qdrant's Query API "prefetch + RRF fusion"
   * pattern. Two prefetch passes (dense, sparse) feed into an RRF
   * fusion step on the server. The router's `vectorWeight` is not
   * passed through — Qdrant's fusion is rank-based RRF (no per-side
   * weight); the weighting in this adapter is best-effort by limit-
   * ing each prefetch to the same depth, which is the standard recipe.
   */
  override async searchHybrid(opts: HybridSearchOpts): Promise<SearchResult[]> {
    if (!this.sparseEncoder) {
      return this.searchVector(opts);
    }
    if (this.sparseEncoder.isFitted && !this.sparseEncoder.isFitted()) {
      return this.searchVector(opts);
    }
    const sparse = this.sparseEncoder.encode(opts.query);

    // Prefetch depth — wider than topK so fusion has signal beyond the
    // trim point. 4× is the same multiplier BaseAdapter.searchHybrid
    // uses for its in-process RRF; reusing it keeps behaviour
    // predictable across adapters.
    const prefetchLimit = Math.max(opts.topK * 4, 20);

    const prefetch: unknown[] = [
      {
        query: opts.embedding,
        using: this.denseName,
        limit: prefetchLimit,
      },
    ];
    if (sparse.indices.length > 0) {
      prefetch.push({
        query: sparse,
        using: this.sparseName,
        limit: prefetchLimit,
      });
    }

    const body: Record<string, unknown> = {
      prefetch,
      query: { fusion: "rrf" },
      limit: opts.topK,
      with_payload: true,
    };
    if (opts.filter) body.filter = qdrantFilter(opts.filter);

    const res = await this.fetch(
      `/collections/${this.collection}/points/query`,
      "POST",
      body
    );
    if (!res.ok) throw new Error(`Qdrant hybrid query failed: ${await res.text()}`);
    return parseQdrantResults(await res.json());
  }

  async delete(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    const res = await this.fetch(
      `/collections/${this.collection}/points/delete`,
      "POST",
      { points: ids }
    );
    if (!res.ok) throw new Error(`Qdrant delete failed: ${await res.text()}`);
  }

  async count(): Promise<number> {
    const res = await this.fetch(
      `/collections/${this.collection}/points/count`,
      "POST",
      { exact: true }
    );
    if (!res.ok) return 0;
    const json = (await res.json()) as { result?: { count?: number } };
    return json.result?.count ?? 0;
  }

  async clear(): Promise<void> {
    // Qdrant has no atomic "delete all points keeping schema" — the
    // recommended path is filter-delete with a match-all filter.
    const res = await this.fetch(
      `/collections/${this.collection}/points/delete`,
      "POST",
      { filter: {} }
    );
    if (!res.ok) throw new Error(`Qdrant clear failed: ${await res.text()}`);
  }

  private ensureSparseFitted(chunks: Chunk[]): void {
    if (this.sparseFitted || !this.sparseEncoder) return;
    if (this.sparseEncoder.isFitted?.()) {
      this.sparseFitted = true;
      return;
    }
    if (typeof this.sparseEncoder.fit !== "function") {
      this.sparseFitted = true;
      return;
    }
    this.sparseEncoder.fit(chunks.map((c) => c.content));
    this.sparseFitted = true;
  }

  private fetch(path: string, method: string, body: unknown): Promise<Response> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) headers["api-key"] = this.apiKey;
    return fetch(`${this.url}${path}`, {
      method,
      headers,
      body: JSON.stringify(body),
    });
  }
}

/**
 * Translate the Augur-level flat filter (`{ key: value }`, AND-combined)
 * into Qdrant's `must` list. Augur's filter contract is intentionally
 * limited to equality on top-level metadata keys; richer Qdrant
 * filters (range, geo, nested) can be passed by wrapping the adapter
 * and overriding `searchVector` / `searchKeyword` directly.
 */
function qdrantFilter(filter: Record<string, unknown>): {
  must: Array<{ key: string; match: { value: unknown } }>;
} {
  const must = Object.entries(filter).map(([k, v]) => ({
    key: k,
    match: { value: v },
  }));
  return { must };
}

function parseQdrantResults(json: unknown): SearchResult[] {
  // Qdrant's Query API returns { result: { points: [{id, score, payload}, ...] } }
  const j = json as {
    result?: {
      points?: Array<{
        id: string | number;
        score: number;
        payload?: Record<string, unknown>;
      }>;
    };
  };
  const points = j.result?.points ?? [];
  return points.map((p) => {
    const payload = p.payload ?? {};
    return {
      score: p.score,
      chunk: {
        id: String(p.id),
        documentId: String(payload.documentId ?? ""),
        content: String(payload.content ?? ""),
        index: Number(payload.index ?? 0),
        metadata: payload,
      },
    };
  });
}
