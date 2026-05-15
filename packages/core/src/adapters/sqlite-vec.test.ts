import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { SqliteVecAdapter, type SqliteDb, type SqliteStatement } from "./sqlite-vec.js";
import type { Chunk } from "../types.js";

/**
 * SqliteVecAdapter tests — `SqliteDb` is mocked at the statement
 * boundary, same posture as the pgvector tests. We exercise SQL shape
 * and parameter binding; the live sqlite-vec virtual table is not
 * involved (testing against a real loaded extension would require a
 * native binary in CI and is the integration-test layer's job).
 */

interface Recorded {
  sql: string;
  prepared: Array<{ kind: "run" | "get" | "all"; params: unknown[] }>;
}

let captured: Recorded[] = [];
let nextGet: unknown | undefined = undefined;
let nextAll: unknown[] = [];
let nextRunLastInsertRowid: number = 0;
let execCaptured: string[] = [];

function statementFor(sql: string): SqliteStatement {
  const rec: Recorded = { sql, prepared: [] };
  captured.push(rec);
  return {
    run(...params: unknown[]) {
      rec.prepared.push({ kind: "run", params });
      const id = ++nextRunLastInsertRowid;
      return { changes: 1, lastInsertRowid: id };
    },
    get<T = unknown>(...params: unknown[]) {
      rec.prepared.push({ kind: "get", params });
      return nextGet as T | undefined;
    },
    all<T = unknown>(...params: unknown[]) {
      rec.prepared.push({ kind: "all", params });
      return nextAll as T[];
    },
  };
}

const mockDb: SqliteDb = {
  prepare(sql: string) {
    return statementFor(sql);
  },
  exec(sql: string) {
    execCaptured.push(sql);
  },
};

beforeEach(() => {
  captured = [];
  nextGet = undefined;
  nextAll = [];
  nextRunLastInsertRowid = 0;
  execCaptured = [];
});

function ad(dim = 3): SqliteVecAdapter {
  return new SqliteVecAdapter({ db: mockDb, dimension: dim });
}

function chunk(id: string, content: string, embedding: number[]): Chunk {
  return { id, documentId: id, content, index: 0, embedding, metadata: {} };
}

test("SqliteVecAdapter: capabilities are vector-only", () => {
  const a = ad();
  assert.equal(a.capabilities.vector, true);
  assert.equal(a.capabilities.keyword, false);
  assert.equal(a.capabilities.hybrid, false);
});

test("SqliteVecAdapter: rejects malicious table identifiers at construction", () => {
  assert.throws(
    () => new SqliteVecAdapter({ db: mockDb, dimension: 3, table: "x; DROP" }),
    /invalid table identifier/
  );
  assert.throws(
    () => new SqliteVecAdapter({ db: mockDb, dimension: 3, vecTable: "x'" }),
    /invalid vecTable identifier/
  );
});

test("SqliteVecAdapter: rejects non-positive / non-integer dimensions", () => {
  assert.throws(
    () => new SqliteVecAdapter({ db: mockDb, dimension: 0 }),
    /dimension must be a positive integer/
  );
  assert.throws(
    () => new SqliteVecAdapter({ db: mockDb, dimension: 3.14 }),
    /dimension must be a positive integer/
  );
});

// ---------- migrate ----------

test("SqliteVecAdapter.migrate: emits data table + vec0 + index", () => {
  SqliteVecAdapter.migrate({ db: mockDb, dimension: 384 });
  const joined = execCaptured.join("\n");
  assert.match(joined, /CREATE TABLE IF NOT EXISTS chunks/);
  assert.match(joined, /CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0\(\s*embedding FLOAT\[384\]\s*\)/);
  assert.match(joined, /CREATE INDEX IF NOT EXISTS chunks_document_id_idx/);
});

test("SqliteVecAdapter.migrate: honors custom table + vecTable + dimension", () => {
  SqliteVecAdapter.migrate({
    db: mockDb,
    dimension: 1536,
    table: "my_chunks",
    vecTable: "my_vec",
  });
  const joined = execCaptured.join("\n");
  assert.match(joined, /CREATE TABLE IF NOT EXISTS my_chunks/);
  assert.match(joined, /CREATE VIRTUAL TABLE IF NOT EXISTS my_vec USING vec0\(\s*embedding FLOAT\[1536\]/);
});

test("SqliteVecAdapter.migrate: rejects malicious identifiers", () => {
  assert.throws(
    () => SqliteVecAdapter.migrate({ db: mockDb, dimension: 3, table: "x; DROP" }),
    /invalid table/
  );
});

// ---------- upsert ----------

test("SqliteVecAdapter: upsert inserts data and vector, paired via rowid", async () => {
  // Pretend the chunk doesn't exist yet — selectRowid returns undefined.
  nextGet = undefined;
  const a = ad();
  await a.upsert([chunk("c1", "hello", [0.1, 0.2, 0.3])]);

  // Statements prepared: selectRowid, deleteData, insertData, deleteVec, insertVec.
  const sqls = captured.map((c) => c.sql.replace(/\s+/g, " ").trim());
  assert.ok(sqls.some((s) => /SELECT rowid FROM chunks WHERE id = \?/.test(s)));
  assert.ok(sqls.some((s) => /DELETE FROM chunks WHERE id = \?/.test(s)));
  assert.ok(sqls.some((s) => /INSERT INTO chunks/.test(s)));
  assert.ok(sqls.some((s) => /INSERT INTO vec_chunks/.test(s)));

  // The insertVec call must use the rowid returned from insertData
  // (lastInsertRowid). Find the insertVec statement and assert params.
  const insertVecCall = captured
    .find((c) => /INSERT INTO vec_chunks/.test(c.sql))!
    .prepared.find((p) => p.kind === "run")!;
  // Mock assigns monotonic `lastInsertRowid` to every run() call. The
  // adapter calls deleteData (id 1), then insertData (id 2), then
  // forwards insertData's rowid (2) into insertVec. Test pins that
  // bridge: the embedding must land on the same rowid the data row
  // got.
  assert.equal(insertVecCall.params[0], 2);
  assert.equal(insertVecCall.params[1], "[0.1,0.2,0.3]");
});

test("SqliteVecAdapter: upsert deletes prior vec row when chunk already exists", async () => {
  nextGet = { rowid: 7 };
  const a = ad();
  await a.upsert([chunk("c1", "hello", [0.1, 0.2, 0.3])]);
  const deleteVec = captured.find((c) => /DELETE FROM vec_chunks/.test(c.sql))!;
  const run = deleteVec.prepared.find((p) => p.kind === "run")!;
  assert.equal(run.params[0], 7);
});

test("SqliteVecAdapter: upsert rejects dimension mismatch", async () => {
  const a = ad(3);
  await assert.rejects(
    () => a.upsert([chunk("c1", "x", [0.1, 0.2])]), // dim=2 vs 3
    /dimension mismatch/
  );
});

test("SqliteVecAdapter: upsert rejects missing embedding", async () => {
  const a = ad();
  await assert.rejects(
    () =>
      a.upsert([
        { id: "c1", documentId: "c1", content: "x", index: 0, metadata: {} },
      ]),
    /missing embedding/
  );
});

// ---------- searchVector ----------

test("SqliteVecAdapter: searchVector binds JSON-encoded embedding + topK", async () => {
  nextAll = [
    {
      id: "c1",
      document_id: "doc1",
      content: "hello",
      idx: 0,
      metadata: '{"topic":"k8s"}',
      distance: 0.2,
    },
  ];
  const a = ad();
  const out = await a.searchVector({ embedding: [0.1, 0.2, 0.3], topK: 5 });
  const stmt = captured[0]!;
  assert.match(stmt.sql, /v\.embedding MATCH \?/);
  assert.match(stmt.sql, /ORDER BY v\.distance ASC/);
  assert.match(stmt.sql, /LIMIT \?/);
  const call = stmt.prepared.find((p) => p.kind === "all")!;
  assert.equal(call.params[0], "[0.1,0.2,0.3]");
  assert.equal(call.params[1], 5);
  // Score is 1 - distance.
  assert.equal(out[0]!.score, 0.8);
  assert.deepEqual(out[0]!.chunk.metadata, { topic: "k8s" });
});

test("SqliteVecAdapter: searchVector with filter appends json_extract clauses", async () => {
  nextAll = [];
  const a = ad();
  await a.searchVector({
    embedding: [0.1, 0.2, 0.3],
    topK: 5,
    filter: { topic: "k8s", env: "prod" },
  });
  const stmt = captured[0]!;
  assert.match(stmt.sql, /json_extract\(c\.metadata, '\$\.topic'\) = \?/);
  assert.match(stmt.sql, /json_extract\(c\.metadata, '\$\.env'\) = \?/);
  const call = stmt.prepared.find((p) => p.kind === "all")!;
  // [embedding, "k8s", "prod", topK]
  assert.equal(call.params[0], "[0.1,0.2,0.3]");
  assert.equal(call.params[1], "k8s");
  assert.equal(call.params[2], "prod");
  assert.equal(call.params[3], 5);
});

test("SqliteVecAdapter: filter keys are validated", async () => {
  const a = ad();
  await assert.rejects(
    () =>
      a.searchVector({
        embedding: [0.1, 0.2, 0.3],
        topK: 5,
        filter: { "topic'; DROP --": "x" },
      }),
    /invalid filter key/
  );
});

// ---------- keyword / delete / count / clear ----------

test("SqliteVecAdapter: searchKeyword throws (not supported)", async () => {
  const a = ad();
  await assert.rejects(
    () => a.searchKeyword({ query: "any", topK: 5 }),
    /isn't supported in this version/
  );
});

test("SqliteVecAdapter: delete drops vec row + data row", async () => {
  nextGet = { rowid: 4 };
  const a = ad();
  await a.delete(["c1"]);
  const sqls = captured.map((c) => c.sql.replace(/\s+/g, " ").trim());
  assert.ok(sqls.some((s) => /SELECT rowid FROM chunks/.test(s)));
  assert.ok(sqls.some((s) => /DELETE FROM vec_chunks/.test(s)));
  assert.ok(sqls.some((s) => /DELETE FROM chunks/.test(s)));
});

test("SqliteVecAdapter: count returns the data table size", async () => {
  // Mocks: prepare → get returns {n}.
  nextGet = { n: 17 };
  const a = ad();
  const n = await a.count();
  assert.equal(n, 17);
});

test("SqliteVecAdapter: count coerces bigint result", async () => {
  nextGet = { n: 12n };
  const a = ad();
  const n = await a.count();
  assert.equal(n, 12);
});

test("SqliteVecAdapter: clear drops both tables in order (vec first)", async () => {
  const a = ad();
  await a.clear();
  // exec captures happen in order via the mockDb.exec hook.
  assert.equal(execCaptured.length, 2);
  assert.match(execCaptured[0]!, /DELETE FROM vec_chunks/);
  assert.match(execCaptured[1]!, /DELETE FROM chunks/);
});
