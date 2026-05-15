import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";

/**
 * Strict SQL-identifier regex shared by the table name, FTS dictionary,
 * and filter-key validators. Anything outside `[a-zA-Z_][a-zA-Z0-9_]*`
 * is rejected — Postgres can't parameter-bind inside `metadata->>'key'`
 * or in identifier positions like table names, so we whitelist instead
 * of trying to escape.
 */
const IDENT_RE = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

/**
 * Minimal Postgres client interface — keeps `@augur-rag/core` zero-dep.
 *
 * Users pass any client that implements this shape. Both `pg` (`new Client()`)
 * and `postgres` (`postgres()`) can be wrapped in a few lines:
 *
 *   import pg from "pg";
 *   const client = new pg.Client(...); await client.connect();
 *   const adapter = new PgVectorAdapter({
 *     client: { query: (sql, params) => client.query(sql, params).then(r => ({ rows: r.rows })) },
 *     table: "chunks",
 *     dimension: 1536,
 *   });
 */
export interface PgClient {
  query<T = unknown>(text: string, params?: unknown[]): Promise<{ rows: T[] }>;
}

/**
 * Options for `PgVectorAdapter.migrate()`. Only `dimension` is required.
 */
export interface PgVectorMigrationOptions {
  /** Target table. Default `"chunks"`. Validated against the identifier regex. */
  table?: string;
  /**
   * Vector column dimension. Must match the embedder you'll use. Mismatch
   * surfaces on the first upsert as a per-chunk error from the adapter,
   * but baking the dimension into the schema means a tsvector / index
   * mismatch would be silent — so we ask up-front.
   */
  dimension: number;
  /**
   * pgvector index type. `ivfflat` (default) is broadly available;
   * `hnsw` (pgvector ≥ 0.5.0) gives better recall/latency at the cost
   * of build time and memory. Pick `hnsw` for production-scale corpora
   * where you can afford the build cost.
   */
  vectorIndex?: "ivfflat" | "hnsw";
  /**
   * Postgres FTS dictionary for the generated `content_tsv` column.
   * Default `"english"`. Common alternatives: `simple` (no stemming),
   * `german`, `french`, `russian`, etc. — anything `to_tsvector`
   * recognises. Identifier-validated; passing arbitrary strings is
   * rejected.
   */
  ftsLanguage?: string;
}

/**
 * PgVectorAdapter — adapter for Postgres + the pgvector extension.
 *
 * Why this is the recommended adapter for most teams:
 * - You probably already have Postgres.
 * - pgvector + tsvector gives you vector + keyword + hybrid in one place.
 * - Migrations are simple SQL; backups/replication "just work".
 *
 * Schema (created idempotently by `PgVectorAdapter.migrate()`):
 *
 *   CREATE EXTENSION IF NOT EXISTS vector;
 *   CREATE TABLE chunks (
 *     id TEXT PRIMARY KEY,
 *     document_id TEXT NOT NULL,
 *     content TEXT NOT NULL,
 *     index INT NOT NULL,
 *     metadata JSONB,
 *     embedding VECTOR(<dimension>),
 *     content_tsv tsvector
 *       GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
 *   );
 *   CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
 *   CREATE INDEX ON chunks USING gin (content_tsv);
 *   CREATE INDEX ON chunks (document_id);
 *
 * Run it once before constructing the adapter:
 *
 *   await PgVectorAdapter.migrate(client, { dimension: 384 });
 *   const adapter = new PgVectorAdapter({ client, dimension: 384 });
 *
 * For metadata filtering, add additional indexes on JSONB paths as needed.
 */
export class PgVectorAdapter extends BaseAdapter {
  readonly name = "pgvector";
  readonly capabilities: AdapterCapabilities = {
    vector: true,
    keyword: true,
    hybrid: true,
    computesEmbeddings: false,
    filtering: true,
  };

  private client: PgClient;
  private table: string;
  private dimension: number;

  constructor(opts: { client: PgClient; table?: string; dimension: number }) {
    super();
    this.client = opts.client;
    this.table = opts.table ?? "chunks";
    this.dimension = opts.dimension;
    if (!IDENT_RE.test(this.table)) {
      throw new Error("PgVectorAdapter: invalid table identifier");
    }
  }

  /**
   * Idempotent one-shot schema setup. Creates the pgvector extension,
   * the chunks table (with the generated tsvector column), and the
   * vector / FTS / document_id indexes — all with `IF NOT EXISTS`, so
   * running twice is a no-op. Safe to call from app boot.
   *
   * Doesn't enforce dimension consistency on an existing table: if the
   * table already exists with a different dimension, this call is a
   * no-op and the upsert path will surface the mismatch chunk-by-chunk.
   * Drop the table manually if you need to change the dimension.
   */
  static async migrate(
    client: PgClient,
    opts: PgVectorMigrationOptions
  ): Promise<void> {
    const table = opts.table ?? "chunks";
    if (!IDENT_RE.test(table)) {
      throw new Error(
        `PgVectorAdapter.migrate: invalid table identifier ${JSON.stringify(table)}`
      );
    }
    if (!Number.isInteger(opts.dimension) || opts.dimension <= 0) {
      throw new Error(
        `PgVectorAdapter.migrate: dimension must be a positive integer (got ${opts.dimension})`
      );
    }
    const vectorIndex = opts.vectorIndex ?? "ivfflat";
    if (vectorIndex !== "ivfflat" && vectorIndex !== "hnsw") {
      throw new Error(
        `PgVectorAdapter.migrate: vectorIndex must be "ivfflat" or "hnsw" (got ${vectorIndex})`
      );
    }
    const lang = opts.ftsLanguage ?? "english";
    if (!IDENT_RE.test(lang)) {
      throw new Error(
        `PgVectorAdapter.migrate: invalid ftsLanguage ${JSON.stringify(lang)}`
      );
    }

    await client.query(`CREATE EXTENSION IF NOT EXISTS vector`);
    await client.query(
      `CREATE TABLE IF NOT EXISTS ${table} (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        content TEXT NOT NULL,
        index INT NOT NULL,
        metadata JSONB,
        embedding VECTOR(${opts.dimension}),
        content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('${lang}', content)) STORED
      )`
    );
    await client.query(
      `CREATE INDEX IF NOT EXISTS ${table}_embedding_idx ON ${table} USING ${vectorIndex} (embedding vector_cosine_ops)`
    );
    await client.query(
      `CREATE INDEX IF NOT EXISTS ${table}_content_tsv_idx ON ${table} USING gin (content_tsv)`
    );
    await client.query(
      `CREATE INDEX IF NOT EXISTS ${table}_document_id_idx ON ${table} (document_id)`
    );
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    if (chunks.length === 0) return;
    // We chunk the upsert into manageable batches because Postgres has
    // a hard limit on parameter count (~65k).
    const BATCH = 200;
    for (let i = 0; i < chunks.length; i += BATCH) {
      const slice = chunks.slice(i, i + BATCH);
      const params: unknown[] = [];
      const valuesSql = slice
        .map((c, idx) => {
          if (!c.embedding) {
            throw new Error(`PgVectorAdapter: chunk ${c.id} missing embedding`);
          }
          if (c.embedding.length !== this.dimension) {
            throw new Error(
              `PgVectorAdapter: dimension mismatch for ${c.id} (got ${c.embedding.length}, expected ${this.dimension})`
            );
          }
          params.push(
            c.id,
            c.documentId,
            c.content,
            c.index,
            JSON.stringify(c.metadata ?? {}),
            formatVector(c.embedding)
          );
          const b = idx * 6;
          return `($${b + 1}, $${b + 2}, $${b + 3}, $${b + 4}, $${b + 5}::jsonb, $${b + 6}::vector)`;
        })
        .join(", ");
      const sql = `
        INSERT INTO ${this.table} (id, document_id, content, index, metadata, embedding)
        VALUES ${valuesSql}
        ON CONFLICT (id) DO UPDATE SET
          document_id = EXCLUDED.document_id,
          content     = EXCLUDED.content,
          index       = EXCLUDED.index,
          metadata    = EXCLUDED.metadata,
          embedding   = EXCLUDED.embedding
      `;
      await this.client.query(sql, params);
    }
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    // pgvector's <=> operator is cosine distance (lower = better).
    const filterSql = buildFilterSql(opts.filter, 2);
    const sql = `
      SELECT id, document_id, content, index, metadata,
             1 - (embedding <=> $1::vector) AS score
      FROM ${this.table}
      ${filterSql.where}
      ORDER BY embedding <=> $1::vector ASC
      LIMIT $${filterSql.nextParam}
    `;
    const params = [formatVector(opts.embedding), ...filterSql.params, opts.topK];
    const { rows } = await this.client.query<RawRow>(sql, params);
    return rows.map(rowToResult);
  }

  async searchKeyword(opts: KeywordSearchOpts): Promise<SearchResult[]> {
    const filterSql = buildFilterSql(opts.filter, 2);
    const sql = `
      SELECT id, document_id, content, index, metadata,
             ts_rank_cd(content_tsv, plainto_tsquery('english', $1)) AS score
      FROM ${this.table}
      WHERE content_tsv @@ plainto_tsquery('english', $1)
      ${filterSql.where ? "AND " + filterSql.where.replace(/^WHERE /, "") : ""}
      ORDER BY score DESC
      LIMIT $${filterSql.nextParam}
    `;
    const params = [opts.query, ...filterSql.params, opts.topK];
    const { rows } = await this.client.query<RawRow>(sql, params);
    return rows.map(rowToResult);
  }

  async delete(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    await this.client.query(`DELETE FROM ${this.table} WHERE id = ANY($1)`, [ids]);
  }

  async count(): Promise<number> {
    const { rows } = await this.client.query<{ count: string }>(
      `SELECT COUNT(*)::text AS count FROM ${this.table}`
    );
    return Number(rows[0]?.count ?? 0);
  }

  async clear(): Promise<void> {
    await this.client.query(`TRUNCATE ${this.table}`);
  }
}

interface RawRow {
  id: string;
  document_id: string;
  content: string;
  index: number;
  metadata: Record<string, unknown> | null;
  score: number | string;
}

function rowToResult(row: RawRow): SearchResult {
  return {
    score: Number(row.score),
    chunk: {
      id: row.id,
      documentId: row.document_id,
      content: row.content,
      index: row.index,
      metadata: row.metadata ?? undefined,
    },
  };
}

function formatVector(v: number[]): string {
  return `[${v.join(",")}]`;
}

function buildFilterSql(
  filter: Record<string, unknown> | undefined,
  startParam: number
): { where: string; params: unknown[]; nextParam: number } {
  if (!filter || Object.keys(filter).length === 0) {
    return { where: "", params: [], nextParam: startParam };
  }
  const clauses: string[] = [];
  const params: unknown[] = [];
  let p = startParam;
  for (const [k, v] of Object.entries(filter)) {
    // Filter keys are interpolated into SQL as JSONB path operands; Postgres
    // doesn't allow parameter binding inside `metadata->>'key'`. We require
    // strict identifier syntax to make the interpolation unambiguously safe
    // and reject anything else loudly. Same rule as the table identifier
    // check in the constructor.
    if (!IDENT_RE.test(k)) {
      throw new Error(
        `PgVectorAdapter: invalid filter key ${JSON.stringify(k)} ` +
          "(must match /^[a-zA-Z_][a-zA-Z0-9_]*$/)"
      );
    }
    clauses.push(`metadata->>'${k}' = $${p}`);
    params.push(String(v));
    p += 1;
  }
  return { where: `WHERE ${clauses.join(" AND ")}`, params, nextParam: p };
}
