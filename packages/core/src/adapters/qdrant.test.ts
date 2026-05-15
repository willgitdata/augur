import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { QdrantAdapter } from "./qdrant.js";
import { BM25SparseEncoder } from "./sparse.js";
import type { Chunk } from "../types.js";

/**
 * QdrantAdapter tests — mock fetch at the global level, exercise the
 * REST request shape (URL, method, headers, body) and the response-
 * decoding path. Same testing posture as the Pinecone adapter.
 */

type Captured = { url: string; init: RequestInit };
let captured: Captured[] = [];
let nextResponse: { ok: boolean; status?: number; body: unknown } = {
  ok: true,
  body: {},
};
const realFetch = globalThis.fetch;

beforeEach(() => {
  captured = [];
  nextResponse = { ok: true, body: {} };
  // @ts-expect-error — loose mock signature
  globalThis.fetch = async (url: string, init: RequestInit) => {
    captured.push({ url, init });
    return {
      ok: nextResponse.ok,
      status: nextResponse.status ?? 200,
      json: async () => nextResponse.body,
      text: async () =>
        typeof nextResponse.body === "string"
          ? nextResponse.body
          : JSON.stringify(nextResponse.body),
    } as unknown as Response;
  };
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

function ad(): QdrantAdapter {
  return new QdrantAdapter({
    url: "https://example.qdrant.io:6333",
    apiKey: "test-key",
    collection: "augur",
  });
}

function chunk(id: string, content: string, embedding: number[]): Chunk {
  return { id, documentId: id, content, index: 0, embedding, metadata: {} };
}

// ---------- capabilities ----------

test("QdrantAdapter: capabilities are vector-only without sparseEncoder", () => {
  const a = ad();
  assert.equal(a.capabilities.vector, true);
  assert.equal(a.capabilities.keyword, false);
  assert.equal(a.capabilities.hybrid, false);
  assert.equal(a.capabilities.filtering, true);
});

test("QdrantAdapter: capabilities flip when sparseEncoder is wired", () => {
  const a = new QdrantAdapter({
    url: "https://x.qdrant.io",
    collection: "c",
    sparseEncoder: new BM25SparseEncoder(),
  });
  assert.equal(a.capabilities.keyword, true);
  assert.equal(a.capabilities.hybrid, true);
});

// ---------- wire shape: upsert ----------

test("QdrantAdapter: upsert PUTs /collections/<c>/points with named dense vector", async () => {
  const a = ad();
  await a.upsert([chunk("c1", "hello", [0.1, 0.2, 0.3])]);
  assert.equal(captured.length, 1);
  const c = captured[0]!;
  assert.equal(c.init.method, "PUT");
  assert.ok(c.url.endsWith("/collections/augur/points"));
  assert.equal((c.init.headers as Record<string, string>)["api-key"], "test-key");
  const body = JSON.parse(c.init.body as string);
  assert.equal(body.points.length, 1);
  assert.equal(body.points[0].id, "c1");
  assert.deepEqual(body.points[0].vector, { dense: [0.1, 0.2, 0.3] });
  assert.equal(body.points[0].payload.content, "hello");
});

test("QdrantAdapter: upsert includes sparse vector when encoder is configured", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["postgres pooling", "redis cache"]);
  const a = new QdrantAdapter({
    url: "https://x.qdrant.io",
    collection: "c",
    sparseEncoder: enc,
  });
  await a.upsert([chunk("c1", "postgres connection pooling", [0.1, 0.2])]);
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.ok(body.points[0].vector.sparse, "sparse vector must be attached");
  assert.ok(Array.isArray(body.points[0].vector.sparse.indices));
});

test("QdrantAdapter: upsert lazily fits sparse encoder when not pre-fit", async () => {
  const enc = new BM25SparseEncoder();
  assert.equal(enc.isFitted(), false);
  const a = new QdrantAdapter({
    url: "https://x.qdrant.io",
    collection: "c",
    sparseEncoder: enc,
  });
  await a.upsert([chunk("c1", "postgres pooling", [0.1, 0.2])]);
  assert.equal(enc.isFitted(), true);
});

test("QdrantAdapter: upsert throws when chunk has no embedding", async () => {
  const a = ad();
  await assert.rejects(
    () =>
      a.upsert([
        { id: "c1", documentId: "c1", content: "x", index: 0, metadata: {} },
      ]),
    /has no embedding/
  );
});

// ---------- wire shape: searchVector ----------

test("QdrantAdapter: searchVector posts to /points/query with `using: dense`", async () => {
  nextResponse = {
    ok: true,
    body: {
      result: {
        points: [
          {
            id: "c1",
            score: 0.9,
            payload: { documentId: "doc1", content: "hello", index: 0 },
          },
        ],
      },
    },
  };
  const a = ad();
  const out = await a.searchVector({
    embedding: [0.1, 0.2],
    topK: 5,
    filter: { topic: "k8s" },
  });
  const c = captured[0]!;
  assert.ok(c.url.endsWith("/collections/augur/points/query"));
  const body = JSON.parse(c.init.body as string);
  assert.deepEqual(body.query, [0.1, 0.2]);
  assert.equal(body.using, "dense");
  assert.equal(body.limit, 5);
  assert.deepEqual(body.filter, {
    must: [{ key: "topic", match: { value: "k8s" } }],
  });
  assert.equal(out.length, 1);
  assert.equal(out[0]!.score, 0.9);
  assert.equal(out[0]!.chunk.documentId, "doc1");
});

// ---------- searchKeyword ----------

test("QdrantAdapter: searchKeyword throws without sparseEncoder", async () => {
  const a = ad();
  await assert.rejects(
    () => a.searchKeyword({ query: "any", topK: 5 }),
    /keyword search requires a sparseEncoder/
  );
});

test("QdrantAdapter: searchKeyword uses sparse named vector when configured", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["postgres pooling", "redis cache"]);
  const a = new QdrantAdapter({
    url: "https://x.qdrant.io",
    collection: "c",
    sparseEncoder: enc,
  });
  nextResponse = { ok: true, body: { result: { points: [] } } };
  await a.searchKeyword({ query: "postgres", topK: 5 });
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.equal(body.using, "sparse");
  assert.ok(body.query.indices, "query must be a sparse vector");
});

test("QdrantAdapter: searchKeyword returns [] when query is all OOV", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["alpha beta"]);
  const a = new QdrantAdapter({
    url: "https://x.qdrant.io",
    collection: "c",
    sparseEncoder: enc,
  });
  const out = await a.searchKeyword({ query: "gamma delta", topK: 5 });
  // No HTTP call was made — the OOV check short-circuits.
  assert.equal(captured.length, 0);
  assert.deepEqual(out, []);
});

// ---------- searchHybrid ----------

test("QdrantAdapter: searchHybrid uses Query API prefetch + RRF fusion", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["postgres pooling"]);
  const a = new QdrantAdapter({
    url: "https://x.qdrant.io",
    collection: "c",
    sparseEncoder: enc,
  });
  nextResponse = { ok: true, body: { result: { points: [] } } };
  await a.searchHybrid({
    embedding: [0.1, 0.2],
    query: "postgres pooling",
    topK: 5,
    vectorWeight: 0.7,
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.equal(body.query.fusion, "rrf");
  assert.equal(body.prefetch.length, 2);
  assert.equal(body.prefetch[0].using, "dense");
  assert.equal(body.prefetch[1].using, "sparse");
});

test("QdrantAdapter: searchHybrid without sparseEncoder falls back to vector-only", async () => {
  const a = ad();
  nextResponse = { ok: true, body: { result: { points: [] } } };
  await a.searchHybrid({
    embedding: [0.1, 0.2],
    query: "anything",
    topK: 3,
    vectorWeight: 0.5,
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  // No prefetch / fusion — straight vector query path.
  assert.equal(body.prefetch, undefined);
  assert.deepEqual(body.query, [0.1, 0.2]);
});

// ---------- delete / count / clear ----------

test("QdrantAdapter: delete posts ID list under `points` key", async () => {
  const a = ad();
  await a.delete(["a", "b"]);
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.points, ["a", "b"]);
});

test("QdrantAdapter: count reads result.count", async () => {
  nextResponse = { ok: true, body: { result: { count: 42 } } };
  const a = ad();
  const n = await a.count();
  assert.equal(n, 42);
});

test("QdrantAdapter: clear sends filter-delete with empty filter", async () => {
  const a = ad();
  await a.clear();
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.filter, {});
});

// ---------- naming ----------

test("QdrantAdapter: name is stable", () => {
  assert.equal(ad().name, "qdrant");
});

test("QdrantAdapter: denseVectorName / sparseVectorName override the defaults", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["alpha"]);
  const a = new QdrantAdapter({
    url: "https://x.qdrant.io",
    collection: "c",
    denseVectorName: "my_dense",
    sparseVectorName: "my_sparse",
    sparseEncoder: enc,
  });
  await a.upsert([chunk("c1", "alpha", [0.1])]);
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.ok(body.points[0].vector.my_dense, "uses overridden dense name");
  assert.ok(body.points[0].vector.my_sparse, "uses overridden sparse name");
});
