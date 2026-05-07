import { test } from "node:test";
import assert from "node:assert/strict";
import { HashEmbedder, TfIdfEmbedder } from "./embedder.js";

test("HashEmbedder: deterministic and L2-normalized", async () => {
  const e = new HashEmbedder(64);
  const [a1] = await e.embed(["hello world"]);
  const [a2] = await e.embed(["hello world"]);
  assert.deepEqual(a1, a2);
  let norm = 0;
  for (const v of a1!) norm += v * v;
  assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-6);
});

test("TfIdfEmbedder: produces L2-normalized vectors", async () => {
  const e = new TfIdfEmbedder({ dimension: 256 });
  e.fit(["postgres connection pooling", "kubernetes liveness probes", "redis cache eviction"]);
  const [v] = await e.embed(["postgres pooling"]);
  let norm = 0;
  for (const x of v!) norm += x * x;
  assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-6);
});

test("TfIdfEmbedder: rare tokens dominate the vector via IDF", async () => {
  const e = new TfIdfEmbedder({ dimension: 256 });
  e.fit([
    "common word common word common",
    "common word",
    "common",
    "common word rare-token",
  ]);
  // Rare token in only 1/4 docs → high IDF.
  const [vRare] = await e.embed(["rare-token"]);
  const [vCommon] = await e.embed(["common"]);
  // Both vectors are L2-normalized; magnitude isn't comparable that way.
  // But the dot product of vRare with itself should be 1, and the dot of
  // vRare with vCommon should be near 0 (different active dimensions).
  let dot = 0;
  for (let i = 0; i < vRare!.length; i++) dot += vRare![i]! * vCommon![i]!;
  assert.ok(Math.abs(dot) < 0.6, `expected near-orthogonal vectors, got dot=${dot}`);
});

test("TfIdfEmbedder: same input → same output", async () => {
  const e = new TfIdfEmbedder({ dimension: 256, corpus: ["alpha beta gamma"] });
  const [v1] = await e.embed(["alpha beta"]);
  const [v2] = await e.embed(["alpha beta"]);
  assert.deepEqual(v1, v2);
});

test("TfIdfEmbedder: stemming makes inflections cluster", async () => {
  const e = new TfIdfEmbedder({ dimension: 1024, useStemming: true });
  e.fit([
    "running runs run runner",
    "connection connections connect connecting",
    "deploy deploys deployed deploying",
  ]);
  const [vRunning] = await e.embed(["running"]);
  const [vRuns] = await e.embed(["runs"]);
  // Both stem to "run" → vectors should be near-identical.
  let dot = 0;
  for (let i = 0; i < vRunning!.length; i++) dot += vRunning![i]! * vRuns![i]!;
  assert.ok(dot > 0.99, `expected near-identical vectors after stemming, got dot=${dot}`);
});

test("TfIdfEmbedder: empty input yields zero vector", async () => {
  const e = new TfIdfEmbedder({ dimension: 64 });
  const [v] = await e.embed([""]);
  assert.equal(v!.length, 64);
  for (const x of v!) assert.equal(x, 0);
});
