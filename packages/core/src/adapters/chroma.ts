import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";

/**
 * ChromaAdapter — adapter against Chroma's HTTP API.
 *
 * Vector-only. Chroma supports rich `where` and `where_document`
 * filters but no native BM25 ranking — `capabilities.keyword` is
 * therefore false and the router falls back to vector on Chroma. A
 * sparse-encoder + Chroma sparse-vector path could land later if
 * Chroma exposes one; not in scope here.
 *
 * No dependency on `chromadb` (the official Node SDK) — the REST
 * shape is small enough that fetch keeps `@augur-rag/core` zero-dep.
 * Users with the SDK in their tree can write a wrapper of the same
 * shape and pass that instead.
 *
 * **Collection setup.** The adapter does NOT create the Chroma
 * collection for you. Run the one-time setup yourself:
 *
 *   POST /api/v2/tenants/<tenant>/databases/<db>/collections
 *   { "name": "<collection>", "metadata": { "hnsw:space": "cosine" } }
 *
 * or use the chromadb SDK once on first deploy. Cosine distance is
 * recommended for sentence-transformer embeddings.
 */
export interface ChromaAdapterOptions {
  /** Base URL of the Chroma server, e.g. `http://localhost:8000`. */
  url: string;
  /** Collection name. Must already exist. */
  collection: string;
  /** Default `"default_tenant"`. */
  tenant?: string;
  /** Default `"default_database"`. */
  database?: string;
  /**
   * Optional bearer auth token. Sent as `Authorization: Bearer <token>`.
   * Chroma Cloud and self-hosted Chroma with auth enabled both accept
   * this header.
   */
  authToken?: string;
}

export class ChromaAdapter extends BaseAdapter {
  readonly name = "chroma";
  readonly capabilities: AdapterCapabilities = {
    vector: true,
    keyword: false,
    hybrid: false,
    computesEmbeddings: false,
    filtering: true,
  };

  private url: string;
  private collection: string;
  private tenant: string;
  private database: string;
  private authToken: string | undefined;

  constructor(opts: ChromaAdapterOptions) {
    super();
    this.url = opts.url.replace(/\/$/, "");
    this.collection = opts.collection;
    this.tenant = opts.tenant ?? "default_tenant";
    this.database = opts.database ?? "default_database";
    this.authToken = opts.authToken;
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    if (chunks.length === 0) return;
    const ids: string[] = [];
    const embeddings: number[][] = [];
    const documents: string[] = [];
    const metadatas: Array<Record<string, unknown>> = [];

    for (const c of chunks) {
      if (!c.embedding) {
        throw new Error(`ChromaAdapter: chunk ${c.id} has no embedding`);
      }
      ids.push(c.id);
      embeddings.push(c.embedding);
      documents.push(c.content);
      metadatas.push({
        documentId: c.documentId,
        index: c.index,
        ...(c.metadata ?? {}),
      });
    }

    const res = await this.fetch("upsert", {
      ids,
      embeddings,
      documents,
      metadatas,
    });
    if (!res.ok) throw new Error(`Chroma upsert failed: ${await res.text()}`);
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    const body: Record<string, unknown> = {
      query_embeddings: [opts.embedding],
      n_results: opts.topK,
      include: ["distances", "documents", "metadatas"],
    };
    if (opts.filter && Object.keys(opts.filter).length > 0) {
      body.where = chromaWhere(opts.filter);
    }
    const res = await this.fetch("query", body);
    if (!res.ok) throw new Error(`Chroma query failed: ${await res.text()}`);
    return parseChromaResults(await res.json());
  }

  async searchKeyword(_opts: KeywordSearchOpts): Promise<SearchResult[]> {
    throw new Error(
      "ChromaAdapter does not support keyword search — Chroma has no native " +
        "BM25 ranking. The router should have fallen back to vector. If you're " +
        "forcing keyword/hybrid, this is the wrong adapter."
    );
  }

  async delete(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    const res = await this.fetch("delete", { ids });
    if (!res.ok) throw new Error(`Chroma delete failed: ${await res.text()}`);
  }

  async count(): Promise<number> {
    const res = await this.fetch("count", undefined, "GET");
    if (!res.ok) return 0;
    const json = (await res.json()) as number | { count?: number };
    if (typeof json === "number") return json;
    return json.count ?? 0;
  }

  async clear(): Promise<void> {
    // Chroma v2 has no "delete all keep schema" endpoint — fetch IDs
    // and delete them in batch. For collections > 10k chunks this is
    // expensive; a real production path is to drop the collection and
    // re-create it. We do the safe thing (preserve schema).
    const ids = await this.allIds();
    if (ids.length > 0) {
      const res = await this.fetch("delete", { ids });
      if (!res.ok) throw new Error(`Chroma clear failed: ${await res.text()}`);
    }
  }

  /** Page through `get` to collect every point ID. Used only by `clear()`. */
  private async allIds(): Promise<string[]> {
    const out: string[] = [];
    const PAGE = 1000;
    let offset = 0;
    // Defensive cap — clearing more than 1M chunks via this path is a
    // sign you should drop+recreate the collection, not iterate.
    for (let safety = 0; safety < 1000; safety++) {
      const res = await this.fetch("get", {
        include: [],
        limit: PAGE,
        offset,
      });
      if (!res.ok) throw new Error(`Chroma get failed: ${await res.text()}`);
      const page = (await res.json()) as { ids?: string[] };
      if (!page.ids || page.ids.length === 0) return out;
      out.push(...page.ids);
      if (page.ids.length < PAGE) return out;
      offset += page.ids.length;
    }
    return out;
  }

  private fetch(
    operation: string,
    body: unknown,
    method: "GET" | "POST" = "POST"
  ): Promise<Response> {
    const path = `/api/v2/tenants/${encodeURIComponent(this.tenant)}/databases/${encodeURIComponent(
      this.database
    )}/collections/${encodeURIComponent(this.collection)}/${operation}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.authToken) headers["Authorization"] = `Bearer ${this.authToken}`;
    const init: RequestInit = { method, headers };
    if (method !== "GET" && body !== undefined) {
      init.body = JSON.stringify(body);
    }
    return fetch(`${this.url}${path}`, init);
  }
}

/**
 * Translate Augur's flat AND-filter into Chroma's `where` operator.
 * Chroma uses `{ key: value }` for equality and `{ "$and": [...] }`
 * for conjunctions. We always emit the equality form: single-key
 * filters stay flat, multi-key filters wrap in `$and`.
 */
function chromaWhere(filter: Record<string, unknown>): Record<string, unknown> {
  const entries = Object.entries(filter);
  if (entries.length === 1) {
    const [k, v] = entries[0]!;
    return { [k]: v };
  }
  return {
    $and: entries.map(([k, v]) => ({ [k]: v })),
  };
}

function parseChromaResults(json: unknown): SearchResult[] {
  // Chroma's query response is column-of-rows: top-level arrays are
  // per-query; element [0] is our single query's batch.
  const j = json as {
    ids?: string[][];
    distances?: number[][];
    documents?: string[][];
    metadatas?: Array<Array<Record<string, unknown> | null>>;
  };
  const ids = j.ids?.[0] ?? [];
  const distances = j.distances?.[0] ?? [];
  const documents = j.documents?.[0] ?? [];
  const metadatas = j.metadatas?.[0] ?? [];
  return ids.map((id, i) => {
    const md = metadatas[i] ?? {};
    // Chroma returns cosine *distance* (lower = better). Convert to a
    // similarity-like score in [0, 1] so it composes with the rest of
    // Augur's score pipeline.
    const dist = distances[i] ?? 0;
    const score = 1 - dist;
    return {
      score,
      chunk: {
        id,
        documentId: String(md.documentId ?? ""),
        content: documents[i] ?? "",
        index: Number(md.index ?? 0),
        metadata: md,
      },
    };
  });
}
