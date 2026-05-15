import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type HybridSearchOpts,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";
import type { SparseEncoder } from "./sparse.js";

/**
 * PineconeAdapter — adapter against Pinecone's REST API.
 *
 * Implementation note: we deliberately don't depend on `@pinecone-database/pinecone`
 * to keep `@augur-rag/core` dependency-free. Users who want the official SDK
 * can write a 30-line wrapper of the same shape. This implementation uses
 * fetch against the data-plane URL.
 *
 * **Two modes, depending on whether a `sparseEncoder` is supplied:**
 *
 * 1. **Dense-only (default).** Vector search only. `capabilities.keyword`
 *    and `capabilities.hybrid` are both false; the router knows to fall
 *    back to vector. `searchKeyword` throws.
 *
 * 2. **Sparse-dense hybrid.** If you pass a `sparseEncoder` (use the
 *    built-in `BM25SparseEncoder`, or wire a hosted encoder like
 *    Pinecone's `pinecone-text` / SPLADE / Cohere's tokenizer), the
 *    adapter computes a sparse vector for each chunk at upsert and for
 *    each query at search time. `capabilities.hybrid = true`; the router
 *    picks the hybrid path for mixed-intent queries, and the adapter
 *    overrides `searchHybrid` to send Pinecone's native sparse-dense
 *    query (no client-side RRF needed).
 *
 *    `capabilities.keyword` stays false because Pinecone has no pure-
 *    sparse query endpoint — its sparse path requires a dense vector to
 *    accompany the sparse one. Pure-keyword strategy still falls back to
 *    a different adapter or to dense-only.
 *
 * The sparse encoder is fit lazily on the first upsert if it isn't
 * already fit. For the cleanest result, pre-fit the encoder over the
 * full corpus content before constructing the adapter — incremental
 * fits across batched upserts produce vocabulary drift.
 */
export interface PineconeAdapterOptions {
  indexHost: string;
  apiKey: string;
  namespace?: string;
  /**
   * Optional sparse encoder for sparse-dense hybrid mode. When set,
   * `capabilities.hybrid` flips to true and the router will consider
   * the hybrid strategy.
   */
  sparseEncoder?: SparseEncoder;
}

export class PineconeAdapter extends BaseAdapter {
  readonly name = "pinecone";
  readonly capabilities: AdapterCapabilities;

  private indexHost: string;
  private apiKey: string;
  private namespace: string;
  private sparseEncoder: SparseEncoder | null;
  private sparseFitted = false;

  constructor(opts: PineconeAdapterOptions) {
    super();
    this.indexHost = opts.indexHost.replace(/\/$/, "");
    this.apiKey = opts.apiKey;
    this.namespace = opts.namespace ?? "default";
    this.sparseEncoder = opts.sparseEncoder ?? null;
    this.capabilities = {
      vector: true,
      keyword: false,
      hybrid: this.sparseEncoder !== null,
      computesEmbeddings: false,
      filtering: true,
    };
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    if (this.sparseEncoder) this.ensureSparseFitted(chunks);
    const vectors = chunks.map((c) => {
      if (!c.embedding) {
        throw new Error(`PineconeAdapter: chunk ${c.id} has no embedding`);
      }
      const out: {
        id: string;
        values: number[];
        sparseValues?: { indices: number[]; values: number[] };
        metadata: Record<string, unknown>;
      } = {
        id: c.id,
        values: c.embedding,
        metadata: {
          documentId: c.documentId,
          content: c.content,
          index: c.index,
          ...(c.metadata ?? {}),
        },
      };
      if (this.sparseEncoder) {
        const sv = this.sparseEncoder.encode(c.content);
        // Pinecone rejects sparse vectors with zero non-zero entries;
        // omit the field rather than sending an empty struct.
        if (sv.indices.length > 0) out.sparseValues = sv;
      }
      return out;
    });

    const res = await this.fetch("/vectors/upsert", "POST", {
      vectors,
      namespace: this.namespace,
    });
    if (!res.ok) throw new Error(`Pinecone upsert failed: ${await res.text()}`);
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    const body: Record<string, unknown> = {
      vector: opts.embedding,
      topK: opts.topK,
      namespace: this.namespace,
      includeMetadata: true,
    };
    if (opts.filter) body.filter = opts.filter;

    const res = await this.fetch("/query", "POST", body);
    if (!res.ok) throw new Error(`Pinecone query failed: ${await res.text()}`);
    return parseMatches(await res.json());
  }

  async searchKeyword(_opts: KeywordSearchOpts): Promise<SearchResult[]> {
    throw new Error(
      "PineconeAdapter does not support pure keyword search. " +
        "Pinecone's sparse path requires a dense vector to accompany it. " +
        "Pass a `sparseEncoder` to PineconeAdapter and let the router pick " +
        "the hybrid strategy, or use a different adapter for keyword."
    );
  }

  /**
   * Sparse-dense hybrid query, when a sparseEncoder is wired. Per
   * Pinecone's published recipe, both vectors are scaled client-side:
   * the dense vector by `alpha` (= vectorWeight) and the sparse vector
   * by `1 - alpha`. Pinecone then returns a single ranked list combining
   * both sides — no client-side RRF needed.
   *
   * Without a sparseEncoder, falls through to vector-only. The router
   * shouldn't reach this path because `capabilities.hybrid = false`
   * there, but the fallback keeps the contract honest.
   */
  override async searchHybrid(opts: HybridSearchOpts): Promise<SearchResult[]> {
    if (!this.sparseEncoder) {
      return this.searchVector(opts);
    }
    // Query-time encoders that need a corpus fit must have been fit
    // either at upsert (ensureSparseFitted) or by the user up-front. We
    // do NOT auto-fit on an empty corpus here — that would replace any
    // existing vocabulary with an empty one. If the encoder reports
    // unfit, fall back to vector-only to avoid garbage results.
    if (this.sparseEncoder.isFitted && !this.sparseEncoder.isFitted()) {
      return this.searchVector(opts);
    }

    const alpha = clamp01(opts.vectorWeight);
    const sparse = this.sparseEncoder.encode(opts.query);
    const denseScaled = opts.embedding.map((v) => v * alpha);
    const sparseScaled = {
      indices: sparse.indices,
      values: sparse.values.map((v) => v * (1 - alpha)),
    };

    const body: Record<string, unknown> = {
      vector: denseScaled,
      topK: opts.topK,
      namespace: this.namespace,
      includeMetadata: true,
    };
    // Pinecone rejects empty sparseVector; only attach if non-empty.
    if (sparseScaled.indices.length > 0) body.sparseVector = sparseScaled;
    if (opts.filter) body.filter = opts.filter;

    const res = await this.fetch("/query", "POST", body);
    if (!res.ok) throw new Error(`Pinecone hybrid query failed: ${await res.text()}`);
    return parseMatches(await res.json());
  }

  async delete(ids: string[]): Promise<void> {
    const res = await this.fetch("/vectors/delete", "POST", {
      ids,
      namespace: this.namespace,
    });
    if (!res.ok) throw new Error(`Pinecone delete failed: ${await res.text()}`);
  }

  async count(): Promise<number> {
    const res = await this.fetch("/describe_index_stats", "POST", {});
    if (!res.ok) throw new Error(`Pinecone stats failed: ${await res.text()}`);
    const json = (await res.json()) as {
      namespaces?: Record<string, { vectorCount: number }>;
      totalVectorCount?: number;
    };
    return json.namespaces?.[this.namespace]?.vectorCount ?? json.totalVectorCount ?? 0;
  }

  async clear(): Promise<void> {
    const res = await this.fetch("/vectors/delete", "POST", {
      deleteAll: true,
      namespace: this.namespace,
    });
    if (!res.ok) throw new Error(`Pinecone clear failed: ${await res.text()}`);
  }

  /**
   * If the configured sparse encoder hasn't been fit yet, fit it now on
   * the contents of the current upsert batch. Three short-circuit cases:
   *   - we've already auto-fit on a previous upsert → skip
   *   - the encoder reports `isFitted()` true → user pre-fit it; respect that
   *   - the encoder has no `fit` method (pre-trained SPLADE etc.) → skip
   * Without these checks, an upsert after pre-fit would silently re-fit
   * with the current batch and blow away vocabulary statistics computed
   * over the full corpus.
   */
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
    return fetch(`${this.indexHost}${path}`, {
      method,
      headers: {
        "Api-Key": this.apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  }
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function parseMatches(json: unknown): SearchResult[] {
  const j = json as {
    matches?: Array<{
      id: string;
      score: number;
      metadata?: Record<string, unknown>;
    }>;
  };
  const matches = j.matches ?? [];
  return matches.map((m) => {
    const md = m.metadata ?? {};
    return {
      score: m.score,
      chunk: {
        id: m.id,
        documentId: String(md.documentId ?? ""),
        content: String(md.content ?? ""),
        index: Number(md.index ?? 0),
        metadata: md,
      },
    };
  });
}
