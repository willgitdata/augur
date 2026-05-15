import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";

/**
 * Minimal SQLite client interface — keeps `@augur-rag/core` zero-dep
 * the same way `PgClient` does.
 *
 * The shape matches `better-sqlite3`'s synchronous API, which is the
 * common choice for sqlite-vec workloads (sqlite-vec ships native
 * bindings and works with better-sqlite3 out of the box). Async
 * clients (e.g. `node:sqlite`) can be adapted by wrapping their async
 * `db.run` / `db.all` into a small synchronous-looking shim if your
 * codebase tolerates that.
 *
 * Loading sqlite-vec is the caller's job:
 *
 *   import Database from "better-sqlite3";
 *   import * as sqliteVec from "sqlite-vec";
 *   const raw = new Database("data.db");
 *   sqliteVec.load(raw);
 *   const adapter = new SqliteVecAdapter({ db: raw, ... });
 *
 * Augur stays out of the load path because the bridge is platform-
 * specific (Mac wheels vs Linux .so vs the WASM build for edge
 * runtimes) and the user already chose their SQLite distribution.
 */
export interface SqliteStatement {
  run(...params: unknown[]): { changes: number; lastInsertRowid: number | bigint };
  get<T = unknown>(...params: unknown[]): T | undefined;
  all<T = unknown>(...params: unknown[]): T[];
}

export interface SqliteDb {
  prepare(sql: string): SqliteStatement;
  exec(sql: string): void;
}

/**
 * Options for `SqliteVecAdapter`.
 */
export interface SqliteVecAdapterOptions {
  /** Pre-loaded SQLite handle. Caller is responsible for loading sqlite-vec. */
  db: SqliteDb;
  /** Vector dimension. Must match the embedder and the vec0 schema. Default 384. */
  dimension?: number;
  /**
   * Name of the data table holding chunk metadata. Default `"chunks"`.
   * Validated against the identifier regex.
   */
  table?: string;
  /**
   * Name of the sqlite-vec virtual table holding embeddings. Default
   * `"vec_chunks"`. Validated against the identifier regex.
   */
  vecTable?: string;
}

const IDENT_RE = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

/**
 * SqliteVecAdapter — adapter for SQLite + the sqlite-vec extension.
 *
 * Use cases:
 *
 * - **Local-first apps.** CLI tools, desktop apps, mobile apps via
 *   SQLite's broad platform support. No daemon, single-file storage.
 * - **Edge runtimes.** sqlite-vec compiles to WASM; pairs with Cloudflare
 *   D1 / Turso / libSQL for an entirely-serverless retrieval stack.
 * - **Tests.** Faster setup than spinning up pgvector in Docker.
 *
 * Capabilities are **vector-only** in this first cut. SQLite has FTS5
 * for native BM25, but joining FTS5 with sqlite-vec requires a
 * separate virtual table + rowid bridge, which is non-trivial to do
 * correctly across upsert / delete / dimension changes. That work is
 * tracked as a follow-up — see the GFI list in CONTRIBUTING.md.
 *
 * Schema (created by `SqliteVecAdapter.migrate()`):
 *
 *   CREATE TABLE IF NOT EXISTS chunks (
 *     id TEXT PRIMARY KEY,
 *     document_id TEXT NOT NULL,
 *     content TEXT NOT NULL,
 *     idx INTEGER NOT NULL,
 *     metadata TEXT  -- JSON
 *   );
 *   CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
 *     embedding FLOAT[<dim>]
 *   );
 *
 * The data table's hidden `rowid` is the bridge to the vec0 virtual
 * table's rowid; both stay in sync because the adapter writes both
 * sides of every upsert / delete inside the same prepared-statement
 * sequence.
 */
export class SqliteVecAdapter extends BaseAdapter {
  readonly name = "sqlite-vec";
  readonly capabilities: AdapterCapabilities = {
    vector: true,
    keyword: false,
    hybrid: false,
    computesEmbeddings: false,
    filtering: true,
  };

  private db: SqliteDb;
  private dimension: number;
  private table: string;
  private vecTable: string;

  constructor(opts: SqliteVecAdapterOptions) {
    super();
    this.db = opts.db;
    this.dimension = opts.dimension ?? 384;
    this.table = opts.table ?? "chunks";
    this.vecTable = opts.vecTable ?? "vec_chunks";
    if (!IDENT_RE.test(this.table)) {
      throw new Error(
        `SqliteVecAdapter: invalid table identifier ${JSON.stringify(this.table)}`
      );
    }
    if (!IDENT_RE.test(this.vecTable)) {
      throw new Error(
        `SqliteVecAdapter: invalid vecTable identifier ${JSON.stringify(this.vecTable)}`
      );
    }
    if (!Number.isInteger(this.dimension) || this.dimension <= 0) {
      throw new Error(
        `SqliteVecAdapter: dimension must be a positive integer (got ${this.dimension})`
      );
    }
  }

  /**
   * Idempotent schema setup: data table + sqlite-vec virtual table.
   * Caller must have already loaded the sqlite-vec extension on the
   * handle (via `sqliteVec.load(db)` or equivalent).
   */
  static migrate(opts: {
    db: SqliteDb;
    dimension: number;
    table?: string;
    vecTable?: string;
  }): void {
    const table = opts.table ?? "chunks";
    const vecTable = opts.vecTable ?? "vec_chunks";
    if (!IDENT_RE.test(table)) {
      throw new Error(`SqliteVecAdapter.migrate: invalid table ${JSON.stringify(table)}`);
    }
    if (!IDENT_RE.test(vecTable)) {
      throw new Error(`SqliteVecAdapter.migrate: invalid vecTable ${JSON.stringify(vecTable)}`);
    }
    if (!Number.isInteger(opts.dimension) || opts.dimension <= 0) {
      throw new Error(
        `SqliteVecAdapter.migrate: dimension must be a positive integer (got ${opts.dimension})`
      );
    }
    opts.db.exec(`
      CREATE TABLE IF NOT EXISTS ${table} (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        content TEXT NOT NULL,
        idx INTEGER NOT NULL,
        metadata TEXT
      );
    `);
    opts.db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS ${vecTable} USING vec0(
        embedding FLOAT[${opts.dimension}]
      );
    `);
    opts.db.exec(`
      CREATE INDEX IF NOT EXISTS ${table}_document_id_idx ON ${table} (document_id);
    `);
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    if (chunks.length === 0) return;

    // Replace semantics — delete first by id, then insert. Two
    // statements rather than ON CONFLICT because the vec0 virtual
    // table needs both rows-side and vector-side updates to stay in
    // sync, and SQLite's UPSERT inside a virtual table is fragile.
    const deleteData = this.db.prepare(`DELETE FROM ${this.table} WHERE id = ?`);
    const insertData = this.db.prepare(
      `INSERT INTO ${this.table} (id, document_id, content, idx, metadata) VALUES (?, ?, ?, ?, ?)`
    );
    const deleteVec = this.db.prepare(`DELETE FROM ${this.vecTable} WHERE rowid = ?`);
    const insertVec = this.db.prepare(
      `INSERT INTO ${this.vecTable} (rowid, embedding) VALUES (?, ?)`
    );
    const selectRowid = this.db.prepare(
      `SELECT rowid FROM ${this.table} WHERE id = ?`
    );

    for (const c of chunks) {
      if (!c.embedding) {
        throw new Error(`SqliteVecAdapter: chunk ${c.id} missing embedding`);
      }
      if (c.embedding.length !== this.dimension) {
        throw new Error(
          `SqliteVecAdapter: dimension mismatch for ${c.id} (got ${c.embedding.length}, expected ${this.dimension})`
        );
      }
      // Locate the existing rowid (if any) so we can drop the matching
      // vec0 row before re-inserting. Without this, repeat upserts
      // accumulate orphan vectors.
      const prior = selectRowid.get<{ rowid: number }>(c.id);
      if (prior !== undefined) deleteVec.run(prior.rowid);
      deleteData.run(c.id);
      const ins = insertData.run(
        c.id,
        c.documentId,
        c.content,
        c.index,
        c.metadata ? JSON.stringify(c.metadata) : null
      );
      // sqlite-vec wants embeddings as a JSON-encoded array literal.
      insertVec.run(ins.lastInsertRowid, JSON.stringify(c.embedding));
    }
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    const filter = opts.filter;
    const filterSql = buildFilterSql(filter);
    // The MATCH operator on a vec0 column does ANN search; rows come
    // back sorted by `distance` ascending. Convert to similarity-style
    // `score = 1 - distance` for parity with the rest of the pipeline.
    const sql = `
      SELECT c.id, c.document_id, c.content, c.idx, c.metadata, v.distance
      FROM ${this.vecTable} v
      JOIN ${this.table} c ON c.rowid = v.rowid
      WHERE v.embedding MATCH ? ${filterSql.where ? "AND " + filterSql.where : ""}
      ORDER BY v.distance ASC
      LIMIT ?
    `;
    const stmt = this.db.prepare(sql);
    const rows = stmt.all<RawRow>(
      JSON.stringify(opts.embedding),
      ...filterSql.params,
      opts.topK
    );
    return rows.map(rowToResult);
  }

  async searchKeyword(_opts: KeywordSearchOpts): Promise<SearchResult[]> {
    throw new Error(
      "SqliteVecAdapter: keyword search isn't supported in this version. " +
        "FTS5 + sqlite-vec bridging is tracked as a follow-up in CONTRIBUTING.md. " +
        "Use a different adapter (InMemoryAdapter, PgVectorAdapter) for keyword."
    );
  }

  async delete(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    const selectRowid = this.db.prepare(
      `SELECT rowid FROM ${this.table} WHERE id = ?`
    );
    const deleteData = this.db.prepare(`DELETE FROM ${this.table} WHERE id = ?`);
    const deleteVec = this.db.prepare(`DELETE FROM ${this.vecTable} WHERE rowid = ?`);
    for (const id of ids) {
      const r = selectRowid.get<{ rowid: number }>(id);
      if (r !== undefined) deleteVec.run(r.rowid);
      deleteData.run(id);
    }
  }

  async count(): Promise<number> {
    const stmt = this.db.prepare(`SELECT COUNT(*) AS n FROM ${this.table}`);
    const row = stmt.get<{ n: number | bigint }>();
    if (!row) return 0;
    return typeof row.n === "bigint" ? Number(row.n) : row.n;
  }

  async clear(): Promise<void> {
    this.db.exec(`DELETE FROM ${this.vecTable}`);
    this.db.exec(`DELETE FROM ${this.table}`);
  }
}

interface RawRow {
  id: string;
  document_id: string;
  content: string;
  idx: number;
  metadata: string | null;
  distance: number;
}

function rowToResult(row: RawRow): SearchResult {
  let metadata: Record<string, unknown> | undefined;
  if (row.metadata) {
    try {
      metadata = JSON.parse(row.metadata) as Record<string, unknown>;
    } catch {
      metadata = undefined;
    }
  }
  return {
    score: 1 - row.distance,
    chunk: {
      id: row.id,
      documentId: row.document_id,
      content: row.content,
      index: row.idx,
      ...(metadata !== undefined ? { metadata } : {}),
    },
  };
}

/**
 * Translate Augur's flat AND-equality filter into a SQL `WHERE`
 * clause over the JSON-encoded `metadata` column. Filter keys are
 * validated identifiers; JSON path traversal uses `json_extract`,
 * which has parameterized inputs for the path and is safe to
 * concatenate the validated key into.
 */
function buildFilterSql(
  filter: Record<string, unknown> | undefined
): { where: string; params: unknown[] } {
  if (!filter || Object.keys(filter).length === 0) {
    return { where: "", params: [] };
  }
  const clauses: string[] = [];
  const params: unknown[] = [];
  for (const [k, v] of Object.entries(filter)) {
    if (!IDENT_RE.test(k)) {
      throw new Error(
        `SqliteVecAdapter: invalid filter key ${JSON.stringify(k)} ` +
          "(must match /^[a-zA-Z_][a-zA-Z0-9_]*$/)"
      );
    }
    clauses.push(`json_extract(c.metadata, '$.${k}') = ?`);
    params.push(typeof v === "string" ? v : String(v));
  }
  return { where: clauses.join(" AND "), params };
}
