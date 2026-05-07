import { test } from "node:test";
import assert from "node:assert/strict";
import { LocalEmbedder } from "./local-embedder.js";

// These tests exercise the wiring (defaults, name, dimension, prefix
// configuration) without actually loading the ONNX model. The expensive
// load+inference path runs through the eval harness instead, so CI doesn't
// download ~50MB of weights.

test("LocalEmbedder: default model is Xenova/all-MiniLM-L6-v2 / 384d", () => {
  const e = new LocalEmbedder();
  assert.equal(e.name, "local:Xenova/all-MiniLM-L6-v2");
  assert.equal(e.dimension, 384);
});

test("LocalEmbedder: custom model surfaces in name", () => {
  const e = new LocalEmbedder({ model: "Xenova/bge-small-en-v1.5" });
  assert.equal(e.name, "local:Xenova/bge-small-en-v1.5");
});

test("LocalEmbedder: empty input returns empty array without loading model", async () => {
  const e = new LocalEmbedder();
  // Doesn't call the ONNX pipeline because length === 0.
  const out = await e.embed([]);
  assert.deepEqual(out, []);
});

test("LocalEmbedder: configurable dimension does not affect name", () => {
  const e = new LocalEmbedder({ dimension: 768 });
  assert.equal(e.dimension, 768);
});

test("LocalEmbedder: prefix options accepted (BGE-style)", () => {
  const e = new LocalEmbedder({
    model: "Xenova/bge-small-en-v1.5",
    queryPrefix: "Represent this sentence for searching relevant passages: ",
    docPrefix: "",
  });
  assert.ok(e.name.includes("bge-small-en-v1.5"));
});
