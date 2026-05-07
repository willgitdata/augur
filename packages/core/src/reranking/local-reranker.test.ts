import { test } from "node:test";
import assert from "node:assert/strict";
import { LocalReranker } from "./local-reranker.js";

test("LocalReranker: default model is ms-marco-MiniLM-L-6-v2", () => {
  const r = new LocalReranker();
  assert.equal(r.name, "local-reranker:Xenova/ms-marco-MiniLM-L-6-v2");
});

test("LocalReranker: empty results return empty without loading model", async () => {
  const r = new LocalReranker();
  const out = await r.rerank("query", [], 5);
  assert.deepEqual(out, []);
});

test("LocalReranker: custom model surfaces in name", () => {
  const r = new LocalReranker({ model: "Xenova/bge-reranker-base" });
  assert.equal(r.name, "local-reranker:Xenova/bge-reranker-base");
});
