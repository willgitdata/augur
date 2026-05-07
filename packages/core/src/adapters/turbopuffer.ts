import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";

/**
 * TurbopufferAdapter — adapter against Turbopuffer's REST API.
 *
 * Turbopuffer supports BM25 + vector + hybrid natively, so this adapter
 * exposes the full capability surface. We still inherit RRF from BaseAdapter
 * for users who prefer rank-fusion over Turbopuffer's native combine.
 *
 * As with the Pinecone adapter, we use fetch directly to avoid an SDK
 * dependency. The wire format mirrors Turbopuffer's documented HTTP API.
 */
export class TurbopufferAdapter extends BaseAdapter {
  readonly name = "turbopuffer";
  readonly capabilities: AdapterCapabilities = {
    vector: true,
    keyword: true,
    hybrid: true,
    computesEmbeddings: false,
    filtering: true,
  };

  private apiKey: string;
  private namespace: string;
  private baseURL: string;

  constructor(opts: { apiKey: string; namespace: string; baseURL?: string }) {
    super();
    this.apiKey = opts.apiKey;
    this.namespace = opts.namespace;
    this.baseURL = opts.baseURL ?? "https://api.turbopuffer.com/v1";
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    const ids = chunks.map((c) => c.id);
    const vectors = chunks.map((c) => {
      if (!c.embedding) {
        throw new Error(`TurbopufferAdapter: chunk ${c.id} has no embedding`);
      }
      return c.embedding;
    });
    const attributes = {
      documentId: chunks.map((c) => c.documentId),
      content: chunks.map((c) => c.content),
      index: chunks.map((c) => c.index),
    };

    const res = await this.fetch(`/namespaces/${this.namespace}`, "POST", {
      upserts: { ids, vectors, attributes },
    });
    if (!res.ok) throw new Error(`Turbopuffer upsert failed: ${await res.text()}`);
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    const res = await this.fetch(
      `/namespaces/${this.namespace}/query`,
      "POST",
      {
        vector: opts.embedding,
        top_k: opts.topK,
        include_attributes: true,
        ...(opts.filter ? { filters: opts.filter } : {}),
      }
    );
    if (!res.ok) throw new Error(`Turbopuffer query failed: ${await res.text()}`);
    return parseTpufResults(await res.json());
  }

  async searchKeyword(opts: KeywordSearchOpts): Promise<SearchResult[]> {
    const res = await this.fetch(
      `/namespaces/${this.namespace}/query`,
      "POST",
      {
        query: opts.query,
        top_k: opts.topK,
        include_attributes: true,
        rank_by: ["content", "BM25", opts.query],
        ...(opts.filter ? { filters: opts.filter } : {}),
      }
    );
    if (!res.ok) throw new Error(`Turbopuffer keyword failed: ${await res.text()}`);
    return parseTpufResults(await res.json());
  }

  async delete(ids: string[]): Promise<void> {
    const res = await this.fetch(`/namespaces/${this.namespace}`, "POST", {
      deletes: ids,
    });
    if (!res.ok) throw new Error(`Turbopuffer delete failed: ${await res.text()}`);
  }

  async count(): Promise<number> {
    const res = await this.fetch(`/namespaces/${this.namespace}`, "GET", null);
    if (!res.ok) return 0;
    const json = (await res.json()) as { approx_count?: number };
    return json.approx_count ?? 0;
  }

  async clear(): Promise<void> {
    const res = await this.fetch(`/namespaces/${this.namespace}`, "DELETE", null);
    if (!res.ok && res.status !== 404)
      throw new Error(`Turbopuffer clear failed: ${await res.text()}`);
  }

  private fetch(
    path: string,
    method: string,
    body: unknown
  ): Promise<Response> {
    return fetch(`${this.baseURL}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      ...(body !== null ? { body: JSON.stringify(body) } : {}),
    });
  }
}

interface TpufResp {
  results?: Array<{
    id: string;
    dist?: number;
    score?: number;
    attributes?: Record<string, unknown>;
  }>;
}

function parseTpufResults(json: unknown): SearchResult[] {
  const j = json as TpufResp;
  if (!j.results) return [];
  return j.results.map((r) => {
    // dist is distance (lower = better) for vector queries; score is for keyword.
    const score = r.score !== undefined ? r.score : r.dist !== undefined ? 1 - r.dist : 0;
    const attrs = r.attributes ?? {};
    return {
      score,
      chunk: {
        id: r.id,
        documentId: String(attrs.documentId ?? ""),
        content: String(attrs.content ?? ""),
        index: Number(attrs.index ?? 0),
        metadata: attrs,
      },
    };
  });
}
