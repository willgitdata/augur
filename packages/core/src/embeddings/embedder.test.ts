import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { GeminiEmbedder, HashEmbedder, TfIdfEmbedder } from "./embedder.js";

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

// ---------- GeminiEmbedder (mocked fetch) ----------

let originalFetch: typeof fetch;
beforeEach(() => {
  originalFetch = globalThis.fetch;
});
afterEach(() => {
  globalThis.fetch = originalFetch;
});

function stubFetch(handler: (url: string, init: RequestInit) => Response) {
  globalThis.fetch = (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    return Promise.resolve(handler(url, init ?? {}));
  };
}

test("GeminiEmbedder: constructor errors when no key", () => {
  const orig = process.env.GEMINI_API_KEY;
  const orig2 = process.env.GOOGLE_API_KEY;
  delete process.env.GEMINI_API_KEY;
  delete process.env.GOOGLE_API_KEY;
  try {
    assert.throws(() => new GeminiEmbedder(), /apiKey not provided/);
  } finally {
    if (orig !== undefined) process.env.GEMINI_API_KEY = orig;
    if (orig2 !== undefined) process.env.GOOGLE_API_KEY = orig2;
  }
});

test("GeminiEmbedder: embedDocuments tags taskType=RETRIEVAL_DOCUMENT", async () => {
  let seenBody: any = null;
  stubFetch((_url, init) => {
    seenBody = JSON.parse(init.body as string);
    return new Response(
      JSON.stringify({
        embeddings: [{ values: [0.1, 0.2, 0.3] }],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  });
  const e = new GeminiEmbedder({ apiKey: "stub-key" });
  await e.embedDocuments(["hello"]);
  assert.equal(seenBody.requests[0].taskType, "RETRIEVAL_DOCUMENT");
  assert.equal(seenBody.requests[0].content.parts[0].text, "hello");
  assert.equal(seenBody.requests[0].model, "models/gemini-embedding-001");
  assert.equal(seenBody.requests[0].outputDimensionality, 768);
});

test("GeminiEmbedder: embedQuery tags taskType=RETRIEVAL_QUERY", async () => {
  let seenBody: any = null;
  stubFetch((_url, init) => {
    seenBody = JSON.parse(init.body as string);
    return new Response(
      JSON.stringify({ embeddings: [{ values: [0.5, 0.6] }] }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  });
  const e = new GeminiEmbedder({ apiKey: "stub-key" });
  const v = await e.embedQuery("how do I deploy");
  // Embedder L2-normalizes [0.5, 0.6] → magnitude sqrt(0.25+0.36) = sqrt(0.61).
  assert.equal(v.length, 2);
  const norm = Math.sqrt(v[0]! ** 2 + v[1]! ** 2);
  assert.ok(Math.abs(norm - 1) < 1e-6, `expected unit norm, got ${norm}`);
  assert.equal(seenBody.requests[0].taskType, "RETRIEVAL_QUERY");
});

test("GeminiEmbedder: chunks large input batches by batchSize", async () => {
  let callCount = 0;
  stubFetch((_url, init) => {
    callCount++;
    const body = JSON.parse(init.body as string);
    return new Response(
      JSON.stringify({
        embeddings: body.requests.map((_: unknown) => ({ values: [1, 2, 3] })),
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  });
  const e = new GeminiEmbedder({ apiKey: "stub-key", batchSize: 5 });
  const out = await e.embed(new Array(13).fill("text"));
  assert.equal(out.length, 13);
  assert.equal(callCount, 3); // ceil(13/5)
});

test("GeminiEmbedder: throws on non-OK without leaking key in message", async () => {
  stubFetch(() => new Response("bad request", { status: 400, statusText: "Bad Request" }));
  const e = new GeminiEmbedder({ apiKey: "secret-please-do-not-leak", maxRetries: 0 });
  await assert.rejects(
    () => e.embed(["hi"]),
    (err: Error) => {
      assert.match(err.message, /Gemini embed failed.*400/);
      assert.ok(!err.message.includes("secret-please-do-not-leak"));
      return true;
    }
  );
});

test("GeminiEmbedder: retries on 429 then succeeds", async () => {
  let calls = 0;
  stubFetch(() => {
    calls += 1;
    if (calls < 2) {
      return new Response("rate", {
        status: 429,
        statusText: "Too Many Requests",
        headers: { "retry-after": "0" },
      });
    }
    return new Response(
      JSON.stringify({ embeddings: [{ values: [1, 0] }] }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  });
  const e = new GeminiEmbedder({ apiKey: "stub", maxRetries: 3 });
  const [v] = await e.embed(["hi"]);
  assert.equal(v!.length, 2);
  assert.equal(calls, 2); // one retry, one success
});

test("GeminiEmbedder: gemini-embedding-001 sets outputDimensionality", async () => {
  let seenBody: any = null;
  stubFetch((_url, init) => {
    seenBody = JSON.parse(init.body as string);
    return new Response(
      JSON.stringify({ embeddings: [{ values: new Array(1024).fill(0) }] }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  });
  const e = new GeminiEmbedder({
    apiKey: "stub",
    model: "gemini-embedding-001",
    dimension: 1024,
  });
  await e.embed(["hello"]);
  assert.equal(seenBody.requests[0].outputDimensionality, 1024);
});
