import { test } from "node:test";
import assert from "node:assert/strict";
import { CascadedReranker, HeuristicReranker } from "./reranker.js";
import type { Reranker } from "./reranker.js";
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

test("HeuristicReranker boosts results with high token overlap", async () => {
  const r = new HeuristicReranker();
  const reordered = await r.rerank("postgres connection pooling", candidates, 3);
  assert.equal(reordered[0]!.chunk.id, "a:0");
});

test("HeuristicReranker handles empty input", async () => {
  const r = new HeuristicReranker();
  assert.deepEqual(await r.rerank("anything", [], 3), []);
});

test("CascadedReranker: chains stages with declining topK", async () => {
  let stage1SeenK = 0;
  let stage2SeenK = 0;
  const stage1: Reranker = {
    name: "stage1",
    async rerank(_q, results, topK) {
      stage1SeenK = topK;
      return results.slice(0, topK);
    },
  };
  const stage2: Reranker = {
    name: "stage2",
    async rerank(_q, results, topK) {
      stage2SeenK = topK;
      return results.slice(0, topK);
    },
  };
  const cascade = new CascadedReranker([
    [stage1, 5],
    [stage2, 99],
  ]);
  const out = await cascade.rerank("q", candidates, 2);
  assert.equal(stage1SeenK, 5);
  assert.equal(stage2SeenK, 2);
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
