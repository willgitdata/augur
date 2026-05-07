import { test } from "node:test";
import assert from "node:assert/strict";
import { Augur, TraceStore } from "./index.js";

test("end-to-end: index + search returns relevant results", async () => {
  const augr = new Augur();
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
  const augr = new Augur();
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
  const augr = new Augur({ traceStore: store });
  await augr.index([{ id: "1", content: "hello world" }]);
  await augr.search({ query: "hello" });
  await augr.search({ query: "world" });
  assert.equal(store.size(), 2);
});

test("forced strategy overrides router", async () => {
  const augr = new Augur();
  await augr.index([{ id: "1", content: "alpha beta gamma delta" }]);
  const { trace } = await augr.search({ query: "alpha", forceStrategy: "keyword" });
  assert.equal(trace.decision.strategy, "keyword");
});
