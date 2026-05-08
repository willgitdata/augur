import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  CascadedReranker,
  CohereReranker,
  HeuristicReranker,
  JinaReranker,
} from "./reranker.js";
import type { SearchResult } from "../types.js";

function chunk(id: string, content: string): SearchResult {
  return {
    chunk: { id, documentId: id.split(":")[0]!, content, index: 0 },
    score: 0.5,
  };
}

const candidates: SearchResult[] = [
  chunk("a:0", "PostgreSQL connection pooling with PgBouncer."),
  chunk("b:0", "Kubernetes liveness probes determine restarts."),
  chunk("c:0", "Redis cache eviction policies allkeys-lru."),
];

// ---------- HeuristicReranker (baseline, no network) ----------

test("HeuristicReranker boosts results with high token overlap", async () => {
  const r = new HeuristicReranker();
  const reordered = await r.rerank("postgres connection pooling", candidates, 3);
  assert.equal(reordered[0]!.chunk.id, "a:0");
});

test("HeuristicReranker handles empty input", async () => {
  const r = new HeuristicReranker();
  assert.deepEqual(await r.rerank("anything", [], 3), []);
});

// ---------- HTTP rerankers (use a fetch stub) ----------

let originalFetch: typeof fetch;
beforeEach(() => {
  originalFetch = globalThis.fetch;
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

function stubFetch(handler: (url: string, init: RequestInit) => Response) {
  globalThis.fetch = (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    return Promise.resolve(handler(url, init ?? {}));
  };
}

test("CohereReranker maps Cohere response to ordered results", async () => {
  stubFetch((_url, init) => {
    const body = JSON.parse(init.body as string);
    assert.equal(body.query, "redis eviction");
    assert.equal(body.documents.length, 3);
    return new Response(
      JSON.stringify({
        results: [
          { index: 2, relevance_score: 0.95 },
          { index: 0, relevance_score: 0.40 },
        ],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  });
  const r = new CohereReranker({ apiKey: "test-key" });
  const out = await r.rerank("redis eviction", candidates, 2);
  assert.equal(out.length, 2);
  assert.equal(out[0]!.chunk.id, "c:0"); // index 2 mapped back
  assert.equal(out[0]!.score, 0.95);
  assert.equal(out[0]!.rawScores?.original, 0.5);
  assert.equal(out[1]!.chunk.id, "a:0");
});

test("CohereReranker throws on non-OK response", async () => {
  stubFetch(() => new Response("rate-limited", { status: 429 }));
  const r = new CohereReranker({ apiKey: "test-key" });
  await assert.rejects(() => r.rerank("q", candidates, 3), /Cohere rerank failed.*429/);
});

test("CohereReranker constructor errors when no apiKey or env var", () => {
  const orig = process.env.COHERE_API_KEY;
  delete process.env.COHERE_API_KEY;
  try {
    assert.throws(() => new CohereReranker(), /apiKey not provided/);
  } finally {
    if (orig !== undefined) process.env.COHERE_API_KEY = orig;
  }
});

test("JinaReranker hits the configured endpoint with auth header", async () => {
  let seenUrl = "";
  let seenAuth = "";
  stubFetch((url, init) => {
    seenUrl = url;
    seenAuth = (init.headers as Record<string, string>)["Authorization"] ?? "";
    return new Response(
      JSON.stringify({
        results: [{ index: 1, relevance_score: 0.9 }],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  });
  const r = new JinaReranker({ apiKey: "jina-test" });
  const out = await r.rerank("k8s probes", candidates, 1);
  assert.ok(seenUrl.includes("api.jina.ai"));
  assert.equal(seenAuth, "Bearer jina-test");
  assert.equal(out[0]!.chunk.id, "b:0");
});

// ---------- CascadedReranker ----------

test("CascadedReranker: chains stages with declining topK", async () => {
  // Stage 1: heuristic — narrows to top-N1
  // Stage 2: stub cross-encoder — narrows to caller's topK
  let stage1SeenK = 0;
  let stage2SeenK = 0;
  const stage1: import("./reranker.js").Reranker = {
    name: "stage1",
    async rerank(_q, results, topK) {
      stage1SeenK = topK;
      return results.slice(0, topK);
    },
  };
  const stage2: import("./reranker.js").Reranker = {
    name: "stage2",
    async rerank(_q, results, topK) {
      stage2SeenK = topK;
      return results.slice(0, topK);
    },
  };
  const cascade = new CascadedReranker([
    [stage1, 5], // narrow to 5 (intermediate)
    [stage2, 99], // ignored — final stage uses caller's topK
  ]);
  const out = await cascade.rerank("q", candidates, 2);
  assert.equal(stage1SeenK, 5);
  assert.equal(stage2SeenK, 2); // final stage uses caller's topK
  assert.equal(out.length, 2);
});

test("CascadedReranker: throws when given no stages", () => {
  assert.throws(() => new CascadedReranker([]), /at least one stage/);
});

test("CascadedReranker: name reflects pipeline", () => {
  const c = new CascadedReranker([
    [new HeuristicReranker(), 50],
    [new HeuristicReranker(), 10],
  ]);
  assert.ok(c.name.includes("heuristic-reranker"));
  assert.ok(c.name.includes("→"));
});
