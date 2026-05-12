import cors from "@fastify/cors";
import {
  Augur,
  type AugurOptions,
  TraceStore,
  type Document,
  type SearchRequest,
} from "@augur-rag/core";
import Fastify, { type FastifyInstance, type FastifyRequest } from "fastify";
import { openApiSpec } from "./openapi.js";

/**
 * Build a Fastify server wrapping a Augur instance.
 *
 * Why we wrap rather than register routes onto a user-supplied app:
 * - Keeps the boot sequence one line for users who want a turnkey server.
 * - Lets the server own the TraceStore lifecycle so /traces always has
 *   somewhere to read from.
 * - Users who want to mount on their own Fastify app can still do so via
 *   `app.register(augurPlugin)` — see plugin.ts for that path.
 *
 * Security defaults:
 * - `cors` defaults to `false` (no cross-origin). Set explicitly to opt in.
 * - Destructive endpoints (`/admin/*`, `DELETE /traces`) require `apiKey`
 *   to be configured. When `apiKey` is unset they return 503 rather than
 *   running unauthenticated. This makes "forgot to set the env var" a
 *   loud failure instead of an open door.
 * - When `apiKey` is set, every endpoint except `/health`, `/openapi.json`,
 *   and `/docs` requires the `x-api-key` header to match.
 */
export interface ServerOptions extends AugurOptions {
  /**
   * Optional API key. When set, every endpoint except `/health`,
   * `/openapi.json`, and `/docs` requires `x-api-key: <key>`.
   * Required to enable destructive endpoints (`/admin/*`, `DELETE /traces`);
   * those return 503 if `apiKey` is unset.
   */
  apiKey?: string;
  /**
   * CORS origins. Defaults to `false` (no cross-origin allowed). Pass a
   * concrete origin or array of origins to opt in; pass `true` to reflect
   * any origin (only safe behind authentication on a trusted network).
   */
  cors?: string | string[] | boolean;
  /** Request size limit. Defaults to 10MB. */
  bodyLimit?: number;
}

/** Routes that never require auth — health, schema, and the docs UI. */
const PUBLIC_ROUTES = new Set(["/health", "/openapi.json", "/docs"]);

/** Strip a `?…` suffix; `req.url` always has at least one segment so this is safe. */
function pathOf(req: FastifyRequest): string {
  return req.url.split("?")[0] ?? "";
}

/** True for routes that perform writes the user can't easily undo. */
function isDestructive(req: FastifyRequest): boolean {
  const url = pathOf(req);
  if (url.startsWith("/admin/")) return true;
  if (req.method === "DELETE" && url === "/traces") return true;
  return false;
}

export function buildServer(options: ServerOptions): FastifyInstance {
  const traceStore = options.traceStore ?? new TraceStore(2000);
  const augr = new Augur({ ...options, traceStore });

  const app = Fastify({
    logger: { level: process.env.LOG_LEVEL ?? "info" },
    bodyLimit: options.bodyLimit ?? 10 * 1024 * 1024,
  });

  app.register(cors, {
    origin: options.cors ?? false,
    methods: ["GET", "POST", "DELETE", "OPTIONS"],
  });

  const apiKey = options.apiKey;

  app.addHook("onRequest", async (req, reply) => {
    const url = pathOf(req);
    if (PUBLIC_ROUTES.has(url)) return;

    const destructive = isDestructive(req);

    // Destructive endpoints are disabled outright when no apiKey is
    // configured. This makes "I forgot to set AUGUR_API_KEY" a loud 503
    // failure instead of a silent open `/admin/clear`.
    if (destructive && !apiKey) {
      return reply.code(503).send({
        error: "admin endpoints disabled",
        hint: "set AUGUR_API_KEY (or pass apiKey to buildServer) to enable",
      });
    }

    // When apiKey is set, every non-public endpoint requires the header.
    if (apiKey) {
      const provided = req.headers["x-api-key"];
      if (provided !== apiKey) {
        return reply.code(401).send({ error: "unauthorized" });
      }
    }
  });

  // ---------- Health & meta ----------

  app.get("/health", async () => ({
    status: "ok",
    adapter: augr.adapter.name,
    embedder: augr.embedder.name,
    chunker: augr.chunker.name,
    router: augr.router.name,
    reranker: augr.reranker?.name ?? null,
    capabilities: augr.adapter.capabilities,
  }));

  app.get("/openapi.json", async () => openApiSpec);

  app.get("/docs", async (_req, reply) => {
    reply.type("text/html").send(SWAGGER_HTML);
  });

  // ---------- Index ----------

  app.post<{ Body: { documents: Document[] } }>("/index", async (req, reply) => {
    const { documents } = req.body;
    if (!Array.isArray(documents)) {
      reply.code(400).send({ error: "body.documents must be an array of Document" });
      return;
    }
    const result = await augr.index(documents);
    return result;
  });

  // ---------- Search ----------

  app.post<{ Body: SearchRequest }>("/search", async (req, reply) => {
    const body = req.body;
    if (!body || typeof body.query !== "string" || body.query.length === 0) {
      reply.code(400).send({ error: "body.query is required" });
      return;
    }
    const result = await augr.search(body);
    return result;
  });

  // ---------- Traces ----------

  app.get<{ Querystring: { limit?: string } }>("/traces", async (req) => {
    const limit = req.query.limit ? Math.min(500, parseInt(req.query.limit, 10) || 100) : 100;
    return { traces: traceStore.list(limit) };
  });

  app.get<{ Params: { id: string } }>("/traces/:id", async (req, reply) => {
    const trace = traceStore.get(req.params.id);
    if (!trace) {
      reply.code(404).send({ error: "trace not found" });
      return;
    }
    return trace;
  });

  app.delete("/traces", async () => {
    traceStore.clear();
    return { ok: true };
  });

  // ---------- Admin ----------

  app.post("/admin/clear", async () => {
    await augr.clear();
    return { ok: true };
  });

  app.get("/admin/stats", async () => {
    return {
      chunks: await augr.adapter.count(),
      traces: traceStore.size(),
    };
  });

  return app;
}

const SWAGGER_HTML = `<!doctype html>
<html><head>
  <title>Augur API</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head><body>
  <div id="swagger"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger" });
  </script>
</body></html>`;
