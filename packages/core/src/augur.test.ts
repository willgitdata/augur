import { test } from "node:test";
import assert from "node:assert/strict";
import { Augur, TraceStore } from "./index.js";
import { StubEmbedder } from "./test-fixtures.js";

const embedder = new StubEmbedder();

test("end-to-end: index + search returns relevant results", async () => {
  const augr = new Augur({ embedder });
  await augr.index([
    { id: "1", content: "The Postgres database supports vector indexing via pgvector." },
    { id: "2", content: "Pinecone is a managed vector database service." },
    { id: "3", content: "Espresso machines require regular descaling." },
  ]);
  const { results, trace } = await augr.search({
    query: "How do I store vectors in Postgres?",
    topK: 2,
  });
  assert.equal(results.length, 2);
  // Top result should be doc 1.
  assert.equal(results[0]!.chunk.documentId, "1");
  assert.ok(trace.totalMs >= 0);
  assert.ok(trace.spans.length > 0);
  assert.ok(trace.decision.reasons.length > 0);
});

test("ad-hoc search with inline documents requires no prior indexing", async () => {
  const augr = new Augur({ embedder });
  const { results } = await augr.search({
    query: "kubernetes pod restart",
    documents: [
      { id: "a", content: "Kubernetes pods restart based on liveness probes." },
      { id: "b", content: "Cooking recipes for pasta." },
    ],
    topK: 1,
  });
  assert.equal(results.length, 1);
  assert.equal(results[0]!.chunk.documentId, "a");
});

test("trace store captures every search", async () => {
  const store = new TraceStore();
  const augr = new Augur({ embedder, traceStore: store });
  await augr.index([{ id: "1", content: "hello world" }]);
  await augr.search({ query: "hello" });
  await augr.search({ query: "world" });
  assert.equal(store.size(), 2);
});

test("forced strategy overrides router", async () => {
  const augr = new Augur({ embedder });
  await augr.index([{ id: "1", content: "alpha beta gamma delta" }]);
  const { trace } = await augr.search({ query: "alpha", forceStrategy: "keyword" });
  assert.equal(trace.decision.strategy, "keyword");
});

test("constructor throws helpful error when embedder is missing", () => {
  assert.throws(
    // @ts-expect-error — intentionally omitting required field
    () => new Augur({}),
    /embedder.*required/
  );
});

test("ad-hoc cache: repeat searches over same docs reuse the scratch adapter", async () => {
  const augr = new Augur({ embedder });
  const docs = [
    { id: "a", content: "Kubernetes pods restart based on liveness probes." },
    { id: "b", content: "Cooking recipes for pasta." },
  ];
  const first = await augr.search({ query: "pod restart", documents: docs, topK: 1 });
  const second = await augr.search({ query: "pasta recipe", documents: docs, topK: 1 });
  assert.equal(first.trace.adHoc, true);
  assert.equal(first.trace.adHocCacheHit, undefined);
  assert.equal(second.trace.adHoc, true);
  assert.equal(second.trace.adHocCacheHit, true);
});

test("ad-hoc cache: different documents produce a fresh scratch adapter (no false hits)", async () => {
  const augr = new Augur({ embedder });
  const docsA = [{ id: "a", content: "alpha content here" }];
  const docsB = [{ id: "a", content: "beta content here" }]; // same id, diff content
  const first = await augr.search({ query: "alpha", documents: docsA, topK: 1 });
  const second = await augr.search({ query: "beta", documents: docsB, topK: 1 });
  assert.equal(first.trace.adHoc, true);
  assert.equal(first.trace.adHocCacheHit, undefined);
  assert.equal(second.trace.adHoc, true);
  assert.equal(second.trace.adHocCacheHit, undefined);
});

test("ad-hoc cache: adHocCacheSize=0 disables caching", async () => {
  const augr = new Augur({ embedder, adHocCacheSize: 0 });
  const docs = [{ id: "a", content: "alpha content here" }];
  const first = await augr.search({ query: "alpha", documents: docs, topK: 1 });
  const second = await augr.search({ query: "alpha", documents: docs, topK: 1 });
  assert.equal(first.trace.adHoc, true);
  assert.equal(first.trace.adHocCacheHit, undefined);
  assert.equal(second.trace.adHoc, true);
  assert.equal(second.trace.adHocCacheHit, undefined);
});
