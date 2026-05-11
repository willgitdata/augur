import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";

/**
 * PineconeAdapter — adapter against Pinecone's REST API.
 *
 * Implementation note: we deliberately don't depend on `@pinecone-database/pinecone`
 * to keep `@augur-rag/core` dependency-free. Users who want the official SDK
 * can write a 30-line wrapper of the same shape. This implementation uses
 * fetch against the data-plane URL.
 *
 * Keyword search: Pinecone does not natively support BM25, so `searchKeyword`
 * throws. The router knows to fall back to vector-only when an adapter is
 * keyword-incapable. Hybrid via Pinecone's sparse-dense vectors is a
 * reasonable v2 — tracked in docs/architecture.md.
 */
export class PineconeAdapter extends BaseAdapter {
  readonly name = "pinecone";
  readonly capabilities: AdapterCapabilities = {
    vector: true,
    keyword: false,
    hybrid: false,
    computesEmbeddings: false,
    filtering: true,
  };

  private indexHost: string;
  private apiKey: string;
  private namespace: string;

  constructor(opts: { indexHost: string; apiKey: string; namespace?: string }) {
    super();
    this.indexHost = opts.indexHost.replace(/\/$/, "");
    this.apiKey = opts.apiKey;
    this.namespace = opts.namespace ?? "default";
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    const vectors = chunks.map((c) => {
      if (!c.embedding) {
        throw new Error(`PineconeAdapter: chunk ${c.id} has no embedding`);
      }
      return {
        id: c.id,
        values: c.embedding,
        metadata: {
          documentId: c.documentId,
          content: c.content,
          index: c.index,
          ...(c.metadata ?? {}),
        },
      };
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
    const json = (await res.json()) as {
      matches: Array<{
        id: string;
        score: number;
        metadata: Record<string, unknown>;
      }>;
    };
    return json.matches.map((m) => ({
      score: m.score,
      chunk: {
        id: m.id,
        documentId: String(m.metadata.documentId ?? ""),
        content: String(m.metadata.content ?? ""),
        index: Number(m.metadata.index ?? 0),
        metadata: m.metadata,
      },
    }));
  }

  async searchKeyword(_opts: KeywordSearchOpts): Promise<SearchResult[]> {
    throw new Error(
      "PineconeAdapter does not support keyword search. The router should " +
        "have fallen back to vector. If you're forcing keyword/hybrid, this is " +
        "the wrong adapter."
    );
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
