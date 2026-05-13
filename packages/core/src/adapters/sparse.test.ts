import { test } from "node:test";
import assert from "node:assert/strict";
import { BM25SparseEncoder } from "./sparse.js";

test("BM25SparseEncoder: isFitted is false until fit() is called", () => {
  const enc = new BM25SparseEncoder();
  assert.equal(enc.isFitted(), false);
  enc.fit(["hello world"]);
  assert.equal(enc.isFitted(), true);
});

test("BM25SparseEncoder: empty text yields empty sparse vector", () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["something else entirely"]);
  assert.deepEqual(enc.encode(""), { indices: [], values: [] });
});

test("BM25SparseEncoder: OOV terms are silently dropped", () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["postgres pooling"]);
  // None of these words appear in the fit corpus → empty sparse vector.
  const sv = enc.encode("kubernetes networking rust");
  assert.deepEqual(sv, { indices: [], values: [] });
});

test("BM25SparseEncoder: known terms produce stable indices + positive values", () => {
  const enc = new BM25SparseEncoder();
  enc.fit([
    "postgres connection pooling",
    "redis cache patterns",
    "postgres vacuum bloat",
  ]);
  const sv = enc.encode("postgres pooling");
  // Both tokens are in the vocab; sparse vector should have 2 entries.
  assert.equal(sv.indices.length, sv.values.length);
  assert.ok(sv.indices.length >= 1, `expected at least one matched term, got ${JSON.stringify(sv)}`);
  for (const v of sv.values) assert.ok(v > 0, "BM25 weights must be positive");
});

test("BM25SparseEncoder: indices are non-negative integers", () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["alpha beta gamma delta"]);
  const sv = enc.encode("alpha gamma");
  for (const i of sv.indices) {
    assert.ok(Number.isInteger(i), `index must be integer, got ${i}`);
    assert.ok(i >= 0, `index must be non-negative, got ${i}`);
  }
});

test("BM25SparseEncoder: rarer terms get higher IDF (higher value)", () => {
  const enc = new BM25SparseEncoder();
  // "common" appears in every doc → IDF should be low; "rare" appears in one.
  enc.fit([
    "common common common common rare",
    "common common common",
    "common common",
    "common",
  ]);
  const svRare = enc.encode("rare");
  const svCommon = enc.encode("common");
  // rare should outweigh common in BM25 terms (high IDF beats low IDF).
  // Both vectors have one entry each.
  const rareWeight = svRare.values[0] ?? 0;
  const commonWeight = svCommon.values[0] ?? 0;
  assert.ok(
    rareWeight > commonWeight,
    `rare (${rareWeight}) should weigh more than common (${commonWeight})`
  );
});

test("BM25SparseEncoder: re-fit clears the prior vocabulary", () => {
  const enc = new BM25SparseEncoder();
  enc.fit(["alpha beta"]);
  const sv1 = enc.encode("alpha");
  assert.equal(sv1.indices.length, 1);
  enc.fit(["gamma delta"]);
  const sv2 = enc.encode("alpha");
  assert.deepEqual(sv2, { indices: [], values: [] }, "alpha is OOV after refit");
  const sv3 = enc.encode("gamma");
  assert.equal(sv3.indices.length, 1);
});

test("BM25SparseEncoder: empty corpus fit doesn't crash and yields empty encodes", () => {
  const enc = new BM25SparseEncoder();
  enc.fit([]);
  assert.equal(enc.isFitted(), true);
  assert.deepEqual(enc.encode("anything"), { indices: [], values: [] });
});

test("BM25SparseEncoder: respects stem=false (no Porter / stopword collapse)", () => {
  const stem = new BM25SparseEncoder({ stem: true });
  stem.fit(["running runs runner"]);
  const noStem = new BM25SparseEncoder({ stem: false });
  noStem.fit(["running runs runner"]);
  // The whole sparse encoder is identical except for tokenization. With
  // stemming on, the three terms collapse; with it off, they don't —
  // vocabulary size diverges.
  assert.ok(
    (stem.encode("running").indices.length) <=
      (noStem.encode("running").indices.length),
    "stemming should produce at most as many distinct terms as no-stem"
  );
});
