import { test } from "node:test";
import assert from "node:assert/strict";
import { Doc2QueryChunker } from "./doc2query-chunker.js";
import { SentenceChunker } from "./chunker.js";

// Tests cover construction + sync-call rejection + name. The model-load
// path runs through the eval harness so CI doesn't download ~24MB of
// weights for unit tests.

test("Doc2QueryChunker: default model is Xenova/LaMini-T5-61M", () => {
  const c = new Doc2QueryChunker({ base: new SentenceChunker() });
  assert.ok(c.name.includes("Xenova/LaMini-T5-61M"));
});

test("Doc2QueryChunker: throws on sync chunk() — must use chunkAsync via chunkDocument", () => {
  const c = new Doc2QueryChunker({ base: new SentenceChunker() });
  assert.throws(
    () => c.chunk({ id: "x", content: "hi" }),
    /async; use chunkAsync/
  );
});

test("Doc2QueryChunker: empty doc returns empty without invoking the model", async () => {
  const c = new Doc2QueryChunker({ base: new SentenceChunker() });
  const out = await c.chunkAsync({ id: "x", content: "" });
  assert.deepEqual(out, []);
});

test("Doc2QueryChunker: name reflects model and base chunker", () => {
  const c = new Doc2QueryChunker({
    base: new SentenceChunker(),
    model: "Xenova/flan-t5-small",
  });
  assert.ok(c.name.includes("Xenova/flan-t5-small"));
  assert.ok(c.name.includes("sentence"));
});
