import { test } from "node:test";
import assert from "node:assert/strict";
import { dcgAt, ndcgAt, reciprocalRank, recallAt, mean } from "./metrics.js";

test("dcgAt: empty list and zero k produce 0", () => {
  assert.equal(dcgAt([], 10), 0);
  assert.equal(dcgAt([1, 1, 1], 0), 0);
});

test("dcgAt: perfect top-1 binary relevance equals 1", () => {
  // dcg = (2^1 - 1) / log2(2) = 1
  assert.equal(dcgAt([1], 1), 1);
});

test("ndcgAt: perfect ranking is 1.0", () => {
  // 3 relevant docs, all retrieved at top — should be 1.
  assert.equal(ndcgAt([1, 1, 1], 3, 10), 1);
});

test("ndcgAt: relevant doc demoted scores < 1", () => {
  const ranked = ndcgAt([0, 0, 1], 1, 10);
  const ideal = ndcgAt([1, 0, 0], 1, 10);
  assert.ok(ranked < ideal);
  assert.ok(ranked > 0);
});

test("ndcgAt: zero relevant docs returns 0", () => {
  assert.equal(ndcgAt([0, 0, 0], 0, 10), 0);
});

test("reciprocalRank: first relevant at rank 1 → 1.0", () => {
  assert.equal(reciprocalRank([1, 0, 1]), 1);
});

test("reciprocalRank: first relevant at rank 3 → 1/3", () => {
  assert.ok(Math.abs(reciprocalRank([0, 0, 1]) - 1 / 3) < 1e-9);
});

test("reciprocalRank: no relevant docs → 0", () => {
  assert.equal(reciprocalRank([0, 0, 0]), 0);
});

test("recallAt: half of relevant docs found in top-k", () => {
  assert.equal(recallAt([1, 0, 1, 0], 4, 10), 0.5);
});

test("recallAt: caps at k", () => {
  // Only first 2 retrieved positions count when k=2.
  assert.equal(recallAt([0, 0, 1, 1], 2, 2), 0);
});

test("mean: empty list returns 0", () => {
  assert.equal(mean([]), 0);
});

test("mean: simple average", () => {
  assert.equal(mean([1, 2, 3, 4]), 2.5);
});
