import test from "node:test";
import assert from "node:assert/strict";
import { Tracer, TraceStore } from "./tracer.js";
import type { SearchTrace } from "../types.js";

function makeTrace(id: string, query = "q"): SearchTrace {
  return {
    id,
    query,
    startedAt: new Date().toISOString(),
    totalMs: 1,
    candidates: 0,
    adapter: "in-memory",
    decision: {
      strategy: "vector",
      reasons: [],
      reranked: false,
      signals: {
        wordCount: 1,
        avgWordLen: 1,
        hasQuotedPhrase: false,
        hasSpecificTokens: false,
        isQuestion: false,
        ambiguity: 0,
        hasNamedEntity: false,
        hasCodeLike: false,
        hasDateOrVersion: false,
        questionType: null,
        hasNegation: false,
        language: "en",
      },
    },
    spans: [],
  };
}

test("Tracer: span records duration and attributes", async () => {
  const t = new Tracer("hello");
  await t.span("step", async () => 1, { key: "value" });
  const trace = t.finish({
    candidates: 0,
    adapter: "in-memory",
    decision: makeTrace("x").decision,
  });
  assert.equal(trace.spans.length, 1);
  assert.equal(trace.spans[0]!.name, "step");
  assert.equal(trace.spans[0]!.attributes?.key, "value");
});

test("TraceStore: rejects invalid capacity", () => {
  assert.throws(() => new TraceStore(0));
  assert.throws(() => new TraceStore(-1));
  assert.throws(() => new TraceStore(1.5));
});

test("TraceStore: list returns newest first", () => {
  const s = new TraceStore(10);
  for (let i = 0; i < 5; i++) s.push(makeTrace(`t${i}`));
  const got = s.list().map((t) => t.id);
  assert.deepEqual(got, ["t4", "t3", "t2", "t1", "t0"]);
});

test("TraceStore: list respects limit", () => {
  const s = new TraceStore(10);
  for (let i = 0; i < 5; i++) s.push(makeTrace(`t${i}`));
  const got = s.list(3).map((t) => t.id);
  assert.deepEqual(got, ["t4", "t3", "t2"]);
});

test("TraceStore: ring buffer evicts oldest at capacity", () => {
  const s = new TraceStore(3);
  for (let i = 0; i < 5; i++) s.push(makeTrace(`t${i}`));
  // After 5 pushes into a capacity-3 store: t0 and t1 are evicted.
  assert.equal(s.size(), 3);
  const got = s.list().map((t) => t.id);
  assert.deepEqual(got, ["t4", "t3", "t2"]);
});

test("TraceStore: ring buffer correctness after many evictions", () => {
  // Push 1000 traces into a capacity-50 store. The newest 50 must remain
  // and they must be ordered newest-first.
  const s = new TraceStore(50);
  for (let i = 0; i < 1000; i++) s.push(makeTrace(`t${i}`));
  const got = s.list().map((t) => t.id);
  assert.equal(got.length, 50);
  assert.equal(got[0], "t999");
  assert.equal(got[49], "t950");
});

test("TraceStore: get retrieves by id", () => {
  const s = new TraceStore(5);
  s.push(makeTrace("a"));
  s.push(makeTrace("b"));
  s.push(makeTrace("c"));
  assert.equal(s.get("b")?.id, "b");
  assert.equal(s.get("missing"), undefined);
});

test("TraceStore: get returns undefined for evicted traces", () => {
  const s = new TraceStore(2);
  s.push(makeTrace("a"));
  s.push(makeTrace("b"));
  s.push(makeTrace("c")); // evicts a
  assert.equal(s.get("a"), undefined);
  assert.equal(s.get("b")?.id, "b");
  assert.equal(s.get("c")?.id, "c");
});

test("TraceStore: clear resets size and wraps from zero", () => {
  const s = new TraceStore(3);
  for (let i = 0; i < 5; i++) s.push(makeTrace(`t${i}`));
  s.clear();
  assert.equal(s.size(), 0);
  assert.deepEqual(s.list(), []);
  s.push(makeTrace("new"));
  assert.equal(s.size(), 1);
  assert.equal(s.list()[0]!.id, "new");
});

test("TraceStore: push performance — O(1) per push at capacity", () => {
  // Catch a future regression to Array.shift() by measuring time per push
  // at capacity. With the old shift() implementation, 50k pushes into a
  // 2000-cap store took ~2-3s on a laptop. With the ring buffer, it's
  // under 100ms. Loose 1s bound to avoid CI flakiness.
  const s = new TraceStore(2000);
  const ITERATIONS = 50_000;
  const start = performance.now();
  for (let i = 0; i < ITERATIONS; i++) s.push(makeTrace(`t${i}`));
  const elapsed = performance.now() - start;
  assert.ok(
    elapsed < 1000,
    `50k pushes into a 2000-cap store took ${elapsed.toFixed(0)}ms (expected <1000ms)`
  );
});
