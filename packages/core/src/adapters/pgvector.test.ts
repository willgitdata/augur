import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { PgVectorAdapter, type PgClient } from "./pgvector.js";
import type { Chunk } from "../types.js";

/**
 * PgVectorAdapter tests — the `PgClient` interface is mocked at the
 * `query()` boundary so we exercise SQL shape and parameter binding
 * without standing up a real Postgres.
 *
 * What we verify here:
 *   - SQL targets the configured table (with the identifier-validation
 *     guard against `; DROP TABLE …`).
 *   - Parameter binding numbering survives the filter-clause concat
 *     (the failure mode would be a runtime "wrong number of params"
 *     from Postgres on real-world filter queries).
 *   - Filter-key string interpolation can't escape the JSON-path quote
 *     (single-quote injection regression test).
 *   - Insert batching at 200 chunks per round-trip (the parameter-count
 *     limit Postgres has on prepared statements).
 *   - Score and metadata coercion handle nullable / string-encoded
 *     return values correctly (real `pg` clients return `NUMERIC` as
 *     strings).
 */

let captured: Array<{ sql: string; params: unknown[] }> = [];
let nextRows: unknown[] = [];

const mockClient: PgClient = {
  async query<T = unknown>(sql: string, params?: unknown[]): Promise<{ rows: T[] }> {
    captured.push({ sql, params: params ?? [] });
    return { rows: nextRows as T[] };
  },
};

beforeEach(() => {
  captured = [];
  nextRows = [];
});

function ad(): PgVectorAdapter {
  return new PgVectorAdapter({
    client: mockClient,
    table: "chunks",
    dimension: 3,
  });
}

function chunk(id: string, content: string, embedding: number[]): Chunk {
  return { id, documentId: id, content, index: 0, embedding, metadata: {} };
}

test("PgVectorAdapter: capabilities = full vector + keyword + hybrid", () => {
  const a = ad();
  assert.equal(a.capabilities.vector, true);
  assert.equal(a.capabilities.keyword, true);
  assert.equal(a.capabilities.hybrid, true);
  assert.equal(a.capabilities.filtering, true);
});

test("PgVectorAdapter: rejects malicious table identifiers at construction", () => {
  // The constructor's identifier check is the line of defense between
  // `${this.table}` interpolation and SQL injection. Pin it.
  assert.throws(
    () =>
      new PgVectorAdapter({
        client: mockClient,
        table: "chunks; DROP TABLE users; --",
        dimension: 3,
      }),
    /invalid table identifier/
  );
  assert.throws(
    () => new PgVectorAdapter({ client: mockClient, table: "1bad", dimension: 3 }),
    /invalid table identifier/
  );
  assert.throws(
    () => new PgVectorAdapter({ client: mockClient, table: "ok name", dimension: 3 }),
    /invalid table identifier/
  );
});

test("PgVectorAdapter: upsert generates parameterized INSERT … ON CONFLICT", async () => {
  const a = ad();
  await a.upsert([chunk("c1", "hello", [0.1, 0.2, 0.3])]);
  assert.equal(captured.length, 1);
  const c = captured[0]!;
  assert.match(c.sql, /INSERT INTO chunks/);
  assert.match(c.sql, /ON CONFLICT \(id\) DO UPDATE/);
  // 6 params per row.
  assert.equal(c.params.length, 6);
  assert.equal(c.params[0], "c1");
  assert.equal(c.params[1], "c1");
  assert.equal(c.params[2], "hello");
  assert.equal(c.params[3], 0);
  assert.equal(c.params[4], "{}"); // JSON.stringify of empty metadata
  assert.equal(c.params[5], "[0.1,0.2,0.3]"); // formatVector output
});

test("PgVectorAdapter: upsert batches at 200 chunks per round-trip", async () => {
  const a = ad();
  const chunks = Array.from({ length: 250 }, (_, i) =>
    chunk(`c${i}`, "x", [0.1, 0.2, 0.3])
  );
  await a.upsert(chunks);
  // 250 / 200 = 2 batches.
  assert.equal(captured.length, 2);
  assert.equal(captured[0]!.params.length, 200 * 6);
  assert.equal(captured[1]!.params.length, 50 * 6);
});

test("PgVectorAdapter: upsert rejects mismatched embedding dimension", async () => {
  const a = ad();
  await assert.rejects(
    () => a.upsert([chunk("c1", "x", [0.1, 0.2])]), // dim=2 vs configured 3
    /dimension mismatch/
  );
});

test("PgVectorAdapter: upsert rejects chunks without an embedding", async () => {
  const a = ad();
  await assert.rejects(
    () =>
      a.upsert([
        { id: "c1", documentId: "c1", content: "x", index: 0, metadata: {} },
      ]),
    /missing embedding/
  );
});

test("PgVectorAdapter: searchVector binds vector + topK and decodes rows", async () => {
  nextRows = [
    {
      id: "c1",
      document_id: "doc1",
      content: "hello world",
      index: 0,
      metadata: { topic: "k8s" },
      score: "0.85", // pg NUMERIC arrives as string
    },
  ];
  const a = ad();
  const out = await a.searchVector({ embedding: [0.1, 0.2, 0.3], topK: 5 });
  const c = captured[0]!;
  assert.match(c.sql, /1 - \(embedding <=> \$1::vector\)/);
  assert.equal(c.params[0], "[0.1,0.2,0.3]");
  // Param 2 is topK in this no-filter call.
  assert.equal(c.params[1], 5);
  assert.equal(out.length, 1);
  assert.equal(out[0]!.score, 0.85); // coerced to number
  assert.equal(out[0]!.chunk.documentId, "doc1");
});

test("PgVectorAdapter: searchVector with filter renumbers parameters correctly", async () => {
  nextRows = [];
  const a = ad();
  await a.searchVector({
    embedding: [0.1, 0.2, 0.3],
    topK: 5,
    filter: { topic: "k8s", env: "prod" },
  });
  const c = captured[0]!;
  // $1 = vector, $2 = "k8s", $3 = "prod", $4 = topK
  assert.equal(c.params.length, 4);
  assert.equal(c.params[0], "[0.1,0.2,0.3]");
  assert.equal(c.params[1], "k8s");
  assert.equal(c.params[2], "prod");
  assert.equal(c.params[3], 5);
  assert.match(c.sql, /metadata->>'topic' = \$2/);
  assert.match(c.sql, /metadata->>'env' = \$3/);
  assert.match(c.sql, /LIMIT \$4/);
});

test("PgVectorAdapter: filter keys are validated (injection guard)", async () => {
  const a = ad();
  // Adversarial filter key. Postgres doesn't allow parameter binding inside
  // `metadata->>'key'`, so the key is interpolated. Rather than relying on
  // single-quote escaping (which is correct only under default
  // standard_conforming_strings), we reject anything that isn't a strict
  // identifier — same rule as the table-name check in the constructor.
  await assert.rejects(
    () =>
      a.searchVector({
        embedding: [0.1, 0.2, 0.3],
        topK: 1,
        filter: { "topic'; DROP TABLE chunks; --": "x" },
      }),
    /invalid filter key/
  );
  // No query should have been issued.
  assert.equal(captured.length, 0);
});

test("PgVectorAdapter: filter keys reject non-identifier characters", async () => {
  const a = ad();
  // Spaces, dashes, dots, leading digits — all rejected. JSONB paths in
  // the wild can contain these, but for retrieval-filter use we want a
  // strict whitelist.
  for (const bad of ["foo bar", "foo-bar", "foo.bar", "1abc", "", "foo'bar"]) {
    await assert.rejects(
      () =>
        a.searchVector({
          embedding: [0.1, 0.2, 0.3],
          topK: 1,
          filter: { [bad]: "x" },
        }),
      /invalid filter key/,
      `expected rejection for filter key ${JSON.stringify(bad)}`
    );
  }
});

test("PgVectorAdapter: searchKeyword uses plainto_tsquery + AND filters", async () => {
  nextRows = [];
  const a = ad();
  await a.searchKeyword({ query: "redis cluster", topK: 3, filter: { topic: "redis" } });
  const c = captured[0]!;
  assert.match(c.sql, /plainto_tsquery\('english', \$1\)/);
  assert.match(c.sql, /content_tsv @@ plainto_tsquery/);
  // Filter should be `AND metadata->>...`, NOT `WHERE metadata->>...`.
  assert.match(c.sql, /AND metadata->>'topic' = \$2/);
  assert.equal(c.params[0], "redis cluster");
  assert.equal(c.params[1], "redis");
  assert.equal(c.params[2], 3);
});

test("PgVectorAdapter: delete sends a single ANY($1) bind", async () => {
  const a = ad();
  await a.delete(["a", "b", "c"]);
  assert.equal(captured.length, 1);
  assert.match(captured[0]!.sql, /DELETE FROM chunks WHERE id = ANY\(\$1\)/);
  assert.deepEqual(captured[0]!.params[0], ["a", "b", "c"]);
});

test("PgVectorAdapter: delete on empty list is a no-op", async () => {
  const a = ad();
  await a.delete([]);
  assert.equal(captured.length, 0);
});

test("PgVectorAdapter: count() coerces NUMERIC string to number", async () => {
  nextRows = [{ count: "42" }];
  const a = ad();
  const n = await a.count();
  assert.equal(n, 42);
});

test("PgVectorAdapter: clear() truncates the table", async () => {
  const a = ad();
  await a.clear();
  assert.equal(captured.length, 1);
  assert.match(captured[0]!.sql, /TRUNCATE chunks/);
});

test("PgVectorAdapter: rowToResult preserves null metadata as undefined", async () => {
  nextRows = [
    {
      id: "c1",
      document_id: "doc1",
      content: "x",
      index: 0,
      metadata: null,
      score: 0.5,
    },
  ];
  const a = ad();
  const out = await a.searchVector({ embedding: [0.1, 0.2, 0.3], topK: 1 });
  assert.equal(out[0]!.chunk.metadata, undefined);
});
