import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { PineconeAdapter } from "./pinecone.js";
import { BM25SparseEncoder } from "./sparse.js";
import type { Chunk } from "../types.js";

/**
 * PineconeAdapter tests — fetch is mocked at the global level so we
 * exercise the request shape (URL, method, headers, body) and the
 * response-decoding path without making real network calls.
 *
 * Why this matters: Pinecone is one of the three production adapters
 * users actually deploy. Anything that breaks the request shape silently
 * will surface as zero-result searches in production. These tests pin
 * the wire format so refactors can't accidentally regress it.
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
  // @ts-expect-error — mock signature is loose on purpose
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

function ad(): PineconeAdapter {
  return new PineconeAdapter({
    indexHost: "https://example-abcd.svc.us-east-1.pinecone.io",
    apiKey: "test-key",
    namespace: "test-ns",
  });
}

function chunk(id: string, content: string, embedding: number[]): Chunk {
  return { id, documentId: id, content, index: 0, embedding, metadata: {} };
}

test("PineconeAdapter: capabilities are vector-only", () => {
  const a = ad();
  assert.equal(a.capabilities.vector, true);
  assert.equal(a.capabilities.keyword, false);
  assert.equal(a.capabilities.hybrid, false);
  assert.equal(a.capabilities.filtering, true);
});

test("PineconeAdapter: trailing slash on indexHost is stripped", async () => {
  const a = new PineconeAdapter({
    indexHost: "https://example.pinecone.io/",
    apiKey: "k",
  });
  await a.count();
  assert.equal(
    captured[0]!.url,
    "https://example.pinecone.io/describe_index_stats"
  );
});

test("PineconeAdapter: upsert posts to /vectors/upsert with auth + namespace", async () => {
  const a = ad();
  await a.upsert([chunk("c1", "hello", [0.1, 0.2, 0.3])]);
  assert.equal(captured.length, 1);
  const c = captured[0]!;
  assert.ok(c.url.endsWith("/vectors/upsert"));
  assert.equal((c.init.headers as Record<string, string>)["Api-Key"], "test-key");
  const body = JSON.parse(c.init.body as string);
  assert.equal(body.namespace, "test-ns");
  assert.equal(body.vectors.length, 1);
  assert.equal(body.vectors[0].id, "c1");
  assert.deepEqual(body.vectors[0].values, [0.1, 0.2, 0.3]);
  assert.equal(body.vectors[0].metadata.content, "hello");
});

test("PineconeAdapter: upsert throws when chunk has no embedding", async () => {
  const a = ad();
  await assert.rejects(
    () =>
      a.upsert([
        { id: "c1", documentId: "c1", content: "no vec", index: 0, metadata: {} },
      ]),
    /no embedding/
  );
});

test("PineconeAdapter: upsert error response is surfaced", async () => {
  nextResponse = { ok: false, status: 500, body: "boom" };
  const a = ad();
  await assert.rejects(
    () => a.upsert([chunk("c1", "x", [0.1])]),
    /Pinecone upsert failed/
  );
});

test("PineconeAdapter: searchVector posts to /query with topK + filter", async () => {
  nextResponse = {
    ok: true,
    body: {
      matches: [
        {
          id: "c1",
          score: 0.9,
          metadata: { documentId: "doc1", content: "hello world", index: 0 },
        },
      ],
    },
  };
  const a = ad();
  const out = await a.searchVector({
    embedding: [0.1, 0.2],
    topK: 5,
    filter: { topic: "k8s" },
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.vector, [0.1, 0.2]);
  assert.equal(body.topK, 5);
  assert.equal(body.includeMetadata, true);
  assert.deepEqual(body.filter, { topic: "k8s" });
  assert.equal(out.length, 1);
  assert.equal(out[0]!.score, 0.9);
  assert.equal(out[0]!.chunk.documentId, "doc1");
  assert.equal(out[0]!.chunk.content, "hello world");
});

test("PineconeAdapter: searchVector handles missing metadata gracefully", async () => {
  nextResponse = {
    ok: true,
    body: { matches: [{ id: "c1", score: 0.5, metadata: {} }] },
  };
  const a = ad();
  const out = await a.searchVector({ embedding: [0.1], topK: 1 });
  // Defensive defaults — empty strings / 0 rather than throws.
  assert.equal(out[0]!.chunk.documentId, "");
  assert.equal(out[0]!.chunk.content, "");
  assert.equal(out[0]!.chunk.index, 0);
});

test("PineconeAdapter: searchKeyword throws (capability not declared)", async () => {
  const a = ad();
  await assert.rejects(
    () => a.searchKeyword({ query: "anything", topK: 5 }),
    /does not support pure keyword search/
  );
});

// ---------- sparse-dense hybrid mode ----------

test("PineconeAdapter (sparse-dense): capabilities flip when sparseEncoder is wired", () => {
  const a = new PineconeAdapter({
    indexHost: "https://x.pinecone.io",
    apiKey: "k",
    sparseEncoder: new BM25SparseEncoder(),
  });
  assert.equal(a.capabilities.vector, true);
  assert.equal(a.capabilities.keyword, false, "pinecone has no pure-sparse path");
  assert.equal(a.capabilities.hybrid, true, "sparse-dense hybrid is supported");
});

test("PineconeAdapter (sparse-dense): upsert includes sparseValues per vector", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["postgres pooling", "redis cache", "kubernetes probes"]);
  const a = new PineconeAdapter({
    indexHost: "https://x.pinecone.io",
    apiKey: "k",
    sparseEncoder: enc,
  });
  await a.upsert([chunk("c1", "postgres connection pooling", [0.1, 0.2])]);
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.equal(body.vectors.length, 1);
  assert.ok(body.vectors[0].sparseValues, "sparseValues must be attached");
  assert.ok(Array.isArray(body.vectors[0].sparseValues.indices));
  assert.ok(Array.isArray(body.vectors[0].sparseValues.values));
  assert.ok(
    body.vectors[0].sparseValues.indices.length > 0,
    "indices length > 0 because tokens are in-vocab"
  );
});

test("PineconeAdapter (sparse-dense): upsert lazily fits the encoder when not pre-fit", async () => {
  const enc = new BM25SparseEncoder();
  assert.equal(enc.isFitted(), false);
  const a = new PineconeAdapter({
    indexHost: "https://x.pinecone.io",
    apiKey: "k",
    sparseEncoder: enc,
  });
  await a.upsert([chunk("c1", "postgres connection pooling", [0.1, 0.2])]);
  assert.equal(enc.isFitted(), true);
});

test("PineconeAdapter (sparse-dense): upsert omits sparseValues for OOV-only content", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["alpha beta"]);
  const a = new PineconeAdapter({
    indexHost: "https://x.pinecone.io",
    apiKey: "k",
    sparseEncoder: enc,
  });
  // Content is all stopwords/punctuation under stem+drop-stopwords — empty sparse.
  await a.upsert([chunk("c1", "the the the the", [0.1, 0.2])]);
  const body = JSON.parse(captured[0]!.init.body as string);
  // Pinecone rejects empty sparse vectors — the field must be absent.
  assert.equal(
    body.vectors[0].sparseValues,
    undefined,
    "empty sparse vector must be omitted from the wire format"
  );
});

test("PineconeAdapter (sparse-dense): searchHybrid scales dense by alpha, sparse by 1-alpha", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["postgres pooling", "redis cache"]);
  const a = new PineconeAdapter({
    indexHost: "https://x.pinecone.io",
    apiKey: "k",
    sparseEncoder: enc,
  });
  nextResponse = { ok: true, body: { matches: [] } };
  await a.searchHybrid({
    embedding: [1, 1, 1],
    query: "postgres pooling",
    topK: 10,
    vectorWeight: 0.7,
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  // Dense vector: each component * alpha
  assert.deepEqual(body.vector, [0.7, 0.7, 0.7]);
  // Sparse vector: every value * (1 - alpha) = 0.3
  assert.ok(body.sparseVector, "sparseVector must be on the wire");
  for (const v of body.sparseVector.values) {
    assert.ok(
      Math.abs(v - 0.3 * (v / 0.3)) < 1e-9,
      "sparse values reflect the (1-alpha) scaling"
    );
  }
});

test("PineconeAdapter (sparse-dense): searchHybrid forwards filter when present", async () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["alpha"]);
  const a = new PineconeAdapter({
    indexHost: "https://x.pinecone.io",
    apiKey: "k",
    sparseEncoder: enc,
  });
  nextResponse = { ok: true, body: { matches: [] } };
  await a.searchHybrid({
    embedding: [0.1, 0.2],
    query: "alpha",
    topK: 5,
    vectorWeight: 0.5,
    filter: { topic: "k8s" },
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.filter, { topic: "k8s" });
});

test("PineconeAdapter (sparse-dense): searchHybrid without sparseEncoder falls back to vector-only", async () => {
  const a = ad();
  nextResponse = { ok: true, body: { matches: [] } };
  // Even if a caller reaches searchHybrid, behaviour must be safe.
  await a.searchHybrid({
    embedding: [0.1, 0.2],
    query: "anything",
    topK: 3,
    vectorWeight: 0.5,
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  // No sparseVector field — vector-only path.
  assert.equal(body.sparseVector, undefined);
  assert.deepEqual(body.vector, [0.1, 0.2]);
});

test("PineconeAdapter: delete posts the id list", async () => {
  const a = ad();
  await a.delete(["a", "b", "c"]);
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.ids, ["a", "b", "c"]);
  assert.equal(body.namespace, "test-ns");
});

test("PineconeAdapter: count() reads namespace-specific count", async () => {
  nextResponse = {
    ok: true,
    body: {
      namespaces: { "test-ns": { vectorCount: 42 }, other: { vectorCount: 99 } },
      totalVectorCount: 141,
    },
  };
  const a = ad();
  const n = await a.count();
  assert.equal(n, 42);
});

test("PineconeAdapter: count() falls back to totalVectorCount when namespace missing", async () => {
  nextResponse = {
    ok: true,
    body: { namespaces: {}, totalVectorCount: 7 },
  };
  const a = ad();
  const n = await a.count();
  assert.equal(n, 7);
});

test("PineconeAdapter: clear() sends deleteAll=true", async () => {
  const a = ad();
  await a.clear();
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.equal(body.deleteAll, true);
  assert.equal(body.namespace, "test-ns");
});

test("PineconeAdapter: default namespace is 'default' when omitted", async () => {
  const a = new PineconeAdapter({
    indexHost: "https://x.pinecone.io",
    apiKey: "k",
  });
  await a.upsert([chunk("c1", "x", [0.1])]);
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.equal(body.namespace, "default");
});
