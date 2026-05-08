import { test } from "node:test";
import assert from "node:assert/strict";
import {
  adaptWeightByConfidence,
  clamp,
  composeFilter,
  pickVectorWeight,
  topGapNormalized,
  weightedRrfFuse,
} from "./fusion.js";
import type { QuerySignals, SearchResult } from "./types.js";

// Minimal helper: build a fake SearchResult from id+score so tests stay
// readable and don't have to redeclare Chunk shape on every line.
function r(id: string, score: number): SearchResult {
  return {
    chunk: { id, documentId: id, content: "", index: 0, metadata: {} },
    score,
  };
}

// Default signals; tests override only the fields they care about.
function sig(overrides: Partial<QuerySignals> = {}): QuerySignals {
  return {
    wordCount: 4,
    avgWordLen: 5,
    isQuestion: false,
    questionType: null,
    hasQuotedPhrase: false,
    hasSpecificTokens: false,
    hasCodeLike: false,
    hasNamedEntity: false,
    hasDateOrVersion: false,
    hasNegation: false,
    ambiguity: 0,
    language: "en",
    ...overrides,
  };
}

// ---------- clamp ----------

test("clamp: in-range value passes through", () => {
  assert.equal(clamp(0.5, 0, 1), 0.5);
});

test("clamp: below-range clamps to lo", () => {
  assert.equal(clamp(-1, 0, 1), 0);
});

test("clamp: above-range clamps to hi", () => {
  assert.equal(clamp(2, 0, 1), 1);
});

// ---------- composeFilter ----------

test("composeFilter: no autoLang returns userFilter unchanged", () => {
  assert.deepEqual(composeFilter({ topic: "k8s" }, null), { topic: "k8s" });
});

test("composeFilter: no autoLang and no userFilter returns undefined", () => {
  assert.equal(composeFilter(undefined, null), undefined);
});

test("composeFilter: autoLang only", () => {
  assert.deepEqual(composeFilter(undefined, "ja"), { lang: "ja" });
});

test("composeFilter: autoLang merges with userFilter", () => {
  assert.deepEqual(composeFilter({ topic: "k8s" }, "ja"), {
    lang: "ja",
    topic: "k8s",
  });
});

test("composeFilter: explicit user lang wins over autoLang", () => {
  // Spread order matters: autoLang first, userFilter second; user overrides.
  assert.deepEqual(composeFilter({ lang: "fr" }, "ja"), { lang: "fr" });
});

// ---------- pickVectorWeight ----------

test("pickVectorWeight: quoted phrase → 0.3 (lexical wins)", () => {
  assert.equal(pickVectorWeight(sig({ hasQuotedPhrase: true })), 0.3);
});

test("pickVectorWeight: specific tokens → 0.3", () => {
  assert.equal(pickVectorWeight(sig({ hasSpecificTokens: true })), 0.3);
});

test("pickVectorWeight: code-like → 0.3", () => {
  assert.equal(pickVectorWeight(sig({ hasCodeLike: true })), 0.3);
});

test("pickVectorWeight: very short query → 0.4", () => {
  assert.equal(pickVectorWeight(sig({ wordCount: 2 })), 0.4);
});

test("pickVectorWeight: long natural-language question → 0.7 (semantic wins)", () => {
  assert.equal(
    pickVectorWeight(sig({ wordCount: 8, isQuestion: true })),
    0.7
  );
});

test("pickVectorWeight: medium query with no signals → 0.5 (neutral)", () => {
  assert.equal(pickVectorWeight(sig({ wordCount: 5 })), 0.5);
});

test("pickVectorWeight: lexical signals beat short-query rule", () => {
  // Both signals present; quoted-phrase branch fires first.
  assert.equal(
    pickVectorWeight(sig({ wordCount: 1, hasQuotedPhrase: true })),
    0.3
  );
});

// ---------- weightedRrfFuse ----------

test("weightedRrfFuse: pure vector side (weight=1) returns vec order", () => {
  const vec = [r("a", 0.9), r("b", 0.8), r("c", 0.7)];
  const kw = [r("z", 0.5)];
  const fused = weightedRrfFuse(vec, kw, 1);
  // 'a' should still rank above 'b' since vec ranking dominates.
  assert.equal(fused[0]!.chunk.id, "a");
  assert.equal(fused[1]!.chunk.id, "b");
  assert.equal(fused[2]!.chunk.id, "c");
});

test("weightedRrfFuse: pure keyword side (weight=0) returns kw order", () => {
  const vec = [r("z", 0.5)];
  const kw = [r("x", 0.9), r("y", 0.8)];
  const fused = weightedRrfFuse(vec, kw, 0);
  assert.equal(fused[0]!.chunk.id, "x");
  assert.equal(fused[1]!.chunk.id, "y");
});

test("weightedRrfFuse: shared id sums contributions from both sides", () => {
  const vec = [r("a", 0.9), r("b", 0.8)];
  const kw = [r("a", 0.9), r("c", 0.7)];
  const fused = weightedRrfFuse(vec, kw, 0.5);
  // 'a' appears top in both → must be top in fused.
  assert.equal(fused[0]!.chunk.id, "a");
  // Expected score: 0.5 * 1/(60+1) + 0.5 * 1/(60+1) = 1/(60+1).
  assert.ok(Math.abs(fused[0]!.score - 1 / 61) < 1e-9);
});

test("weightedRrfFuse: out-of-range weight clamps defensively", () => {
  const vec = [r("a", 0.9)];
  const kw = [r("b", 0.8)];
  // Weight=2 should clamp to 1; vec dominates.
  const fused = weightedRrfFuse(vec, kw, 2);
  assert.equal(fused[0]!.chunk.id, "a");
});

test("weightedRrfFuse: empty inputs return empty", () => {
  assert.deepEqual(weightedRrfFuse([], [], 0.5), []);
});

test("weightedRrfFuse: deduplicates within a single fused list", () => {
  const vec = [r("a", 0.9), r("b", 0.8)];
  const kw = [r("a", 0.9), r("b", 0.8)];
  const fused = weightedRrfFuse(vec, kw, 0.5);
  assert.equal(fused.length, 2);
});

// ---------- topGapNormalized ----------

test("topGapNormalized: empty list → 0", () => {
  assert.equal(topGapNormalized([]), 0);
});

test("topGapNormalized: single-item list → 0", () => {
  assert.equal(topGapNormalized([r("a", 0.9)]), 0);
});

test("topGapNormalized: standout #1 produces high confidence", () => {
  // Big gap from #1 to #2, small range to #10.
  const results = [
    r("a", 1.0),
    r("b", 0.5),
    r("c", 0.49),
    r("d", 0.48),
    r("e", 0.47),
    r("f", 0.46),
    r("g", 0.45),
    r("h", 0.44),
    r("i", 0.43),
    r("j", 0.42),
  ];
  const conf = topGapNormalized(results);
  // (1.0 - 0.5) / (1.0 - 0.42) ≈ 0.86
  assert.ok(conf > 0.8, `expected high confidence, got ${conf}`);
});

test("topGapNormalized: flat top-10 → near 0", () => {
  const flat = Array.from({ length: 10 }, (_, i) => r(`x${i}`, 0.5));
  assert.equal(topGapNormalized(flat), 0);
});

test("topGapNormalized: clamps to [0,1]", () => {
  const results = [r("a", 1.0), r("b", 0.0), r("c", 0.5)];
  const conf = topGapNormalized(results);
  assert.ok(conf >= 0 && conf <= 1);
});

// ---------- adaptWeightByConfidence ----------

test("adaptWeightByConfidence: vector confidence shifts toward vector", () => {
  // Vec has standout #1; kw is flat. Should push baseWeight UP.
  const vec = [
    r("a", 1.0),
    r("b", 0.5),
    r("c", 0.49),
    r("d", 0.48),
    r("e", 0.47),
    r("f", 0.46),
    r("g", 0.45),
    r("h", 0.44),
    r("i", 0.43),
    r("j", 0.42),
  ];
  const kw = Array.from({ length: 10 }, (_, i) => r(`x${i}`, 0.5));
  const adapted = adaptWeightByConfidence(0.5, vec, kw);
  assert.ok(adapted > 0.5, `expected shift up, got ${adapted}`);
});

test("adaptWeightByConfidence: keyword confidence shifts toward keyword", () => {
  const vec = Array.from({ length: 10 }, (_, i) => r(`x${i}`, 0.5));
  const kw = [
    r("a", 1.0),
    r("b", 0.5),
    r("c", 0.49),
    r("d", 0.48),
    r("e", 0.47),
    r("f", 0.46),
    r("g", 0.45),
    r("h", 0.44),
    r("i", 0.43),
    r("j", 0.42),
  ];
  const adapted = adaptWeightByConfidence(0.5, vec, kw);
  assert.ok(adapted < 0.5, `expected shift down, got ${adapted}`);
});

test("adaptWeightByConfidence: shift bounded to ±0.20", () => {
  // Maximum confidence delta (vec=1, kw=0). Cap at +0.20.
  const vec = [r("a", 1), r("b", 0)];
  const kw = [r("z", 1), r("y", 1)];
  const adapted = adaptWeightByConfidence(0.5, vec, kw);
  assert.ok(adapted <= 0.5 + 0.20 + 1e-9);
});

test("adaptWeightByConfidence: final weight clamped to [0.10, 0.90]", () => {
  // Even with max prior=0.9 and max upward shift, must not exceed 0.9.
  const vec = [
    r("a", 1.0),
    r("b", 0.0),
    r("c", 0.0),
    r("d", 0.0),
    r("e", 0.0),
    r("f", 0.0),
    r("g", 0.0),
    r("h", 0.0),
    r("i", 0.0),
    r("j", 0.0),
  ];
  const kw = Array.from({ length: 10 }, (_, i) => r(`x${i}`, 0.5));
  const adapted = adaptWeightByConfidence(0.9, vec, kw);
  assert.ok(adapted <= 0.9 + 1e-9);
  assert.ok(adapted >= 0.1 - 1e-9);
});

test("adaptWeightByConfidence: equal confidence preserves baseWeight", () => {
  const vec = [r("a", 1.0), r("b", 0.5)];
  const kw = [r("x", 1.0), r("y", 0.5)];
  const adapted = adaptWeightByConfidence(0.5, vec, kw);
  assert.ok(Math.abs(adapted - 0.5) < 1e-9);
});
