import { test } from "node:test";
import assert from "node:assert/strict";
import { MMRReranker } from "./mmr-reranker.js";
import type { SearchResult } from "../types.js";

function r(id: string, content: string, score: number): SearchResult {
  return {
    chunk: { id, documentId: id, content, index: 0 },
    score,
  };
}

test("MMRReranker: lambda=1 preserves input order (pure relevance)", async () => {
  const m = new MMRReranker({ lambda: 1.0 });
  const input = [
    r("a", "redis cache eviction policy", 0.9),
    r("b", "redis cache eviction lru", 0.85),
    r("c", "kubernetes liveness probes", 0.5),
  ];
  const out = await m.rerank("redis cache", input, 3);
  assert.deepEqual(
    out.map((x) => x.chunk.id),
    ["a", "b", "c"]
  );
});

test("MMRReranker: lambda<1 promotes diverse second result", async () => {
  const m = new MMRReranker({ lambda: 0.3 });
  const input = [
    r("a", "redis cache eviction policy allkeys lru noeviction", 0.9),
    r("b", "redis cache eviction allkeys lru policy noeviction", 0.85),
    r("c", "kubernetes liveness probes restart policy", 0.5),
  ];
  const out = await m.rerank("policy", input, 3);
  assert.equal(out[0]!.chunk.id, "a");
  assert.equal(out[1]!.chunk.id, "c");
  assert.equal(out[2]!.chunk.id, "b");
});

test("MMRReranker: empty input returns empty", async () => {
  const m = new MMRReranker();
  assert.deepEqual(await m.rerank("anything", [], 5), []);
});

test("MMRReranker: respects topK", async () => {
  const m = new MMRReranker({ lambda: 0.5 });
  const input = [
    r("a", "alpha beta gamma", 0.9),
    r("b", "delta epsilon zeta", 0.8),
    r("c", "eta theta iota", 0.7),
  ];
  const out = await m.rerank("anything", input, 2);
  assert.equal(out.length, 2);
});

test("MMRReranker: name reflects lambda", () => {
  const m = new MMRReranker({ lambda: 0.4 });
  assert.equal(m.name, "mmr(λ=0.4)");
});
