import test from "node:test";
import assert from "node:assert/strict";
import { InMemoryAdapter, type Embedder } from "@augur-rag/core";
import { buildServer, type ServerOptions } from "./server.js";

/**
 * Minimal embedder for HTTP-layer tests. Deterministic, no network, no
 * model load. We're testing the auth/CORS/routing surface — embedding
 * quality isn't in scope.
 */
class StubEmbedder implements Embedder {
  readonly name = "stub";
  readonly dimension = 4;
  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => {
      const v = [0, 0, 0, 0];
      for (let i = 0; i < t.length; i++) v[i % 4]! += t.charCodeAt(i);
      const norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
      return v.map((x) => x / norm);
    });
  }
}

function build(extra: Partial<ServerOptions> = {}) {
  return buildServer({
    adapter: new InMemoryAdapter(),
    embedder: new StubEmbedder(),
    ...extra,
  });
}

// =========================================================================
// Public routes are always reachable
// =========================================================================

test("public: /health is reachable without an api key", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({ method: "GET", url: "/health" });
  assert.equal(res.statusCode, 200);
  assert.equal(res.json().status, "ok");
  await app.close();
});

test("public: /openapi.json is reachable without an api key", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({ method: "GET", url: "/openapi.json" });
  assert.equal(res.statusCode, 200);
  assert.equal(res.json().openapi, "3.1.0");
  await app.close();
});

test("public: /docs is reachable without an api key", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({ method: "GET", url: "/docs" });
  assert.equal(res.statusCode, 200);
  assert.match(res.headers["content-type"] as string, /text\/html/);
  await app.close();
});

// =========================================================================
// API key required when configured
// =========================================================================

test("auth: /search requires api key when configured (no header → 401)", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({
    method: "POST",
    url: "/search",
    payload: { query: "hello" },
  });
  assert.equal(res.statusCode, 401);
  await app.close();
});

test("auth: /search rejects wrong api key", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({
    method: "POST",
    url: "/search",
    headers: { "x-api-key": "wrong" },
    payload: { query: "hello" },
  });
  assert.equal(res.statusCode, 401);
  await app.close();
});

test("auth: /search accepts correct api key", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({
    method: "POST",
    url: "/search",
    headers: { "x-api-key": "secret" },
    payload: { query: "hello", documents: [{ id: "1", content: "hello world" }] },
  });
  assert.equal(res.statusCode, 200);
  await app.close();
});

test("auth: /index requires api key when configured", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({
    method: "POST",
    url: "/index",
    payload: { documents: [] },
  });
  assert.equal(res.statusCode, 401);
  await app.close();
});

test("auth: /traces GET requires api key when configured", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({ method: "GET", url: "/traces" });
  assert.equal(res.statusCode, 401);
  await app.close();
});

// =========================================================================
// Destructive routes always require apiKey configuration
// =========================================================================

test("admin: POST /admin/clear returns 503 when apiKey is unset", async () => {
  const app = build({});
  const res = await app.inject({ method: "POST", url: "/admin/clear" });
  assert.equal(res.statusCode, 503);
  assert.match(res.json().error, /disabled/);
  await app.close();
});

test("admin: GET /admin/stats returns 503 when apiKey is unset", async () => {
  const app = build({});
  const res = await app.inject({ method: "GET", url: "/admin/stats" });
  assert.equal(res.statusCode, 503);
  await app.close();
});

test("admin: DELETE /traces returns 503 when apiKey is unset", async () => {
  const app = build({});
  const res = await app.inject({ method: "DELETE", url: "/traces" });
  assert.equal(res.statusCode, 503);
  await app.close();
});

test("admin: POST /admin/clear requires header even when apiKey is set", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({ method: "POST", url: "/admin/clear" });
  assert.equal(res.statusCode, 401);
  await app.close();
});

test("admin: POST /admin/clear accepts correct api key", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({
    method: "POST",
    url: "/admin/clear",
    headers: { "x-api-key": "secret" },
  });
  assert.equal(res.statusCode, 200);
  assert.deepEqual(res.json(), { ok: true });
  await app.close();
});

test("admin: DELETE /traces requires header even when apiKey is set", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({ method: "DELETE", url: "/traces" });
  assert.equal(res.statusCode, 401);
  await app.close();
});

test("admin: DELETE /traces accepts correct api key", async () => {
  const app = build({ apiKey: "secret" });
  const res = await app.inject({
    method: "DELETE",
    url: "/traces",
    headers: { "x-api-key": "secret" },
  });
  assert.equal(res.statusCode, 200);
  await app.close();
});

// =========================================================================
// No apiKey configured: read endpoints still open
// =========================================================================

test("no-key: /search works without auth when apiKey is unset", async () => {
  const app = build({});
  const res = await app.inject({
    method: "POST",
    url: "/search",
    payload: { query: "hello", documents: [{ id: "1", content: "hello world" }] },
  });
  assert.equal(res.statusCode, 200);
  await app.close();
});

test("no-key: query-string variants don't bypass the destructive check", async () => {
  // `/admin/clear?foo=1` should still be treated as destructive.
  const app = build({});
  const res = await app.inject({ method: "POST", url: "/admin/clear?nonce=x" });
  assert.equal(res.statusCode, 503);
  await app.close();
});

// =========================================================================
// CORS defaults
// =========================================================================

test("cors: defaults to no Access-Control-Allow-Origin", async () => {
  const app = build({});
  const res = await app.inject({
    method: "OPTIONS",
    url: "/search",
    headers: {
      origin: "https://evil.example",
      "access-control-request-method": "POST",
    },
  });
  // With cors:false the plugin doesn't add ACAO at all.
  assert.equal(res.headers["access-control-allow-origin"], undefined);
  await app.close();
});

test("cors: explicit string origin is allowed", async () => {
  const app = build({ cors: "https://app.example" });
  const res = await app.inject({
    method: "OPTIONS",
    url: "/search",
    headers: {
      origin: "https://app.example",
      "access-control-request-method": "POST",
    },
  });
  assert.equal(res.headers["access-control-allow-origin"], "https://app.example");
  await app.close();
});

test("cors: explicit origin rejects other origins", async () => {
  const app = build({ cors: "https://app.example" });
  const res = await app.inject({
    method: "OPTIONS",
    url: "/search",
    headers: {
      origin: "https://evil.example",
      "access-control-request-method": "POST",
    },
  });
  assert.notEqual(res.headers["access-control-allow-origin"], "https://evil.example");
  await app.close();
});

// =========================================================================
// Input validation
// =========================================================================

test("validation: /search rejects missing query", async () => {
  const app = build({});
  const res = await app.inject({ method: "POST", url: "/search", payload: {} });
  assert.equal(res.statusCode, 400);
  await app.close();
});

test("validation: /index rejects non-array documents", async () => {
  const app = build({});
  const res = await app.inject({
    method: "POST",
    url: "/index",
    payload: { documents: "not an array" },
  });
  assert.equal(res.statusCode, 400);
  await app.close();
});
