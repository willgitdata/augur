import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { ChromaAdapter } from "./chroma.js";
import type { Chunk } from "../types.js";

/**
 * ChromaAdapter tests — mock fetch at the global level, exercise wire
 * shape against Chroma's v2 REST API.
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

function ad(): ChromaAdapter {
  return new ChromaAdapter({
    url: "http://localhost:8000",
    collection: "augur",
    authToken: "secret",
  });
}

function chunk(id: string, content: string, embedding: number[]): Chunk {
  return { id, documentId: id, content, index: 0, embedding, metadata: {} };
}

test("ChromaAdapter: capabilities are vector-only", () => {
  const a = ad();
  assert.equal(a.capabilities.vector, true);
  assert.equal(a.capabilities.keyword, false);
  assert.equal(a.capabilities.hybrid, false);
  assert.equal(a.capabilities.filtering, true);
});

test("ChromaAdapter: upsert posts to /upsert with parallel column arrays", async () => {
  const a = ad();
  await a.upsert([
    chunk("c1", "hello", [0.1, 0.2, 0.3]),
    chunk("c2", "world", [0.4, 0.5, 0.6]),
  ]);
  const c = captured[0]!;
  assert.match(c.url, /\/api\/v2\/tenants\/default_tenant\/databases\/default_database\/collections\/augur\/upsert$/);
  assert.equal(c.init.method, "POST");
  assert.equal(
    (c.init.headers as Record<string, string>).Authorization,
    "Bearer secret"
  );
  const body = JSON.parse(c.init.body as string);
  assert.deepEqual(body.ids, ["c1", "c2"]);
  assert.deepEqual(body.embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
  assert.deepEqual(body.documents, ["hello", "world"]);
  assert.equal(body.metadatas[0].documentId, "c1");
  assert.equal(body.metadatas[0].index, 0);
});

test("ChromaAdapter: upsert throws when chunk has no embedding", async () => {
  const a = ad();
  await assert.rejects(
    () =>
      a.upsert([
        { id: "c1", documentId: "c1", content: "x", index: 0, metadata: {} },
      ]),
    /has no embedding/
  );
});

test("ChromaAdapter: searchVector wraps the embedding in a batch", async () => {
  nextResponse = {
    ok: true,
    body: {
      ids: [["c1"]],
      distances: [[0.1]],
      documents: [["hello"]],
      metadatas: [[{ documentId: "doc1", index: 0 }]],
    },
  };
  const a = ad();
  const out = await a.searchVector({ embedding: [0.1, 0.2], topK: 3 });
  const body = JSON.parse(captured[0]!.init.body as string);
  // Embedding wrapped in a single-element batch — Chroma's API is
  // "many queries at once"; we always pass one.
  assert.deepEqual(body.query_embeddings, [[0.1, 0.2]]);
  assert.equal(body.n_results, 3);
  assert.equal(out.length, 1);
  // Score is 1 - distance.
  assert.equal(out[0]!.score, 0.9);
  assert.equal(out[0]!.chunk.documentId, "doc1");
  assert.equal(out[0]!.chunk.content, "hello");
});

test("ChromaAdapter: searchVector flat single-key filter → equality where", async () => {
  nextResponse = { ok: true, body: { ids: [[]] } };
  const a = ad();
  await a.searchVector({
    embedding: [0.1],
    topK: 5,
    filter: { topic: "k8s" },
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.where, { topic: "k8s" });
});

test("ChromaAdapter: searchVector multi-key filter → $and combinator", async () => {
  nextResponse = { ok: true, body: { ids: [[]] } };
  const a = ad();
  await a.searchVector({
    embedding: [0.1],
    topK: 5,
    filter: { topic: "k8s", env: "prod" },
  });
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.where, {
    $and: [{ topic: "k8s" }, { env: "prod" }],
  });
});

test("ChromaAdapter: searchKeyword throws (capability not declared)", async () => {
  const a = ad();
  await assert.rejects(
    () => a.searchKeyword({ query: "anything", topK: 5 }),
    /does not support keyword search/
  );
});

test("ChromaAdapter: delete posts IDs to /delete", async () => {
  const a = ad();
  await a.delete(["a", "b", "c"]);
  const body = JSON.parse(captured[0]!.init.body as string);
  assert.deepEqual(body.ids, ["a", "b", "c"]);
});

test("ChromaAdapter: count() decodes integer body or { count } object", async () => {
  // Old API shape — bare integer.
  nextResponse = { ok: true, body: 42 };
  let n = await ad().count();
  assert.equal(n, 42);
  // New API shape — { count }.
  nextResponse = { ok: true, body: { count: 99 } };
  n = await ad().count();
  assert.equal(n, 99);
});

test("ChromaAdapter: tenant + database are encoded into the path", async () => {
  const a = new ChromaAdapter({
    url: "http://localhost:8000",
    collection: "augur",
    tenant: "my tenant", // intentional space
    database: "production",
  });
  await a.upsert([chunk("c1", "x", [0.1])]);
  assert.match(captured[0]!.url, /tenants\/my%20tenant\/databases\/production\/collections\/augur/);
});

test("ChromaAdapter: clear() pages ids then deletes them", async () => {
  let phase = 0;
  // @ts-expect-error — mock override
  globalThis.fetch = async (url: string, init: RequestInit) => {
    captured.push({ url, init });
    if (phase === 0) {
      phase = 1;
      // First /get returns 3 ids, fewer than the page size → terminator.
      return {
        ok: true,
        status: 200,
        json: async () => ({ ids: ["a", "b", "c"] }),
        text: async () => "",
      } as unknown as Response;
    }
    // Second call is /delete.
    return {
      ok: true,
      status: 200,
      json: async () => ({}),
      text: async () => "",
    } as unknown as Response;
  };
  await ad().clear();
  assert.equal(captured.length, 2);
  assert.match(captured[0]!.url, /\/get$/);
  assert.match(captured[1]!.url, /\/delete$/);
  const body = JSON.parse(captured[1]!.init.body as string);
  assert.deepEqual(body.ids, ["a", "b", "c"]);
});

test("ChromaAdapter: clear() with an empty collection is a no-op", async () => {
  // @ts-expect-error — mock override
  globalThis.fetch = async (url: string, init: RequestInit) => {
    captured.push({ url, init });
    return {
      ok: true,
      status: 200,
      json: async () => ({ ids: [] }),
      text: async () => "",
    } as unknown as Response;
  };
  await ad().clear();
  // Only the /get call; no /delete because there's nothing to delete.
  assert.equal(captured.length, 1);
  assert.match(captured[0]!.url, /\/get$/);
});
