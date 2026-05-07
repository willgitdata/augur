import cors from "@fastify/cors";
import {
  Augur,
  type AugurOptions,
  TraceStore,
  type Document,
  type SearchRequest,
} from "@augur/core";
import Fastify, { type FastifyInstance } from "fastify";
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
 */
export interface ServerOptions extends AugurOptions {
  /** Optional API key. Requests must include `x-api-key: <key>` if set. */
  apiKey?: string;
  /** CORS origins. Defaults to "*". */
  cors?: string | string[] | true;
  /** Request size limit. Defaults to 10MB. */
  bodyLimit?: number;
}

export function buildServer(options: ServerOptions = {}): FastifyInstance {
  const traceStore = options.traceStore ?? new TraceStore(2000);
  const qb = new Augur({ ...options, traceStore });

  const app = Fastify({
    logger: { level: process.env.LOG_LEVEL ?? "info" },
    bodyLimit: options.bodyLimit ?? 10 * 1024 * 1024,
  });

  app.register(cors, {
    origin: options.cors ?? true,
    methods: ["GET", "POST", "DELETE", "OPTIONS"],
  });

  // Optional API-key auth.
  if (options.apiKey) {
    app.addHook("onRequest", async (req, reply) => {
      // Allow health and docs without auth.
      const url = req.url.split("?")[0];
      if (url === "/health" || url === "/openapi.json" || url === "/docs") return;
      const provided = req.headers["x-api-key"];
      if (provided !== options.apiKey) {
        reply.code(401).send({ error: "unauthorized" });
      }
    });
  }

  // ---------- Health & meta ----------

  app.get("/health", async () => ({
    status: "ok",
    adapter: qb.adapter.name,
    embedder: qb.embedder.name,
    chunker: qb.chunker.name,
    router: qb.router.name,
    reranker: qb.reranker.name,
    capabilities: qb.adapter.capabilities,
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
    const result = await qb.index(documents);
    return result;
  });

  // ---------- Search ----------

  app.post<{ Body: SearchRequest }>("/search", async (req, reply) => {
    const body = req.body;
    if (!body || typeof body.query !== "string" || body.query.length === 0) {
      reply.code(400).send({ error: "body.query is required" });
      return;
    }
    const result = await qb.search(body);
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
    await qb.clear();
    return { ok: true };
  });

  app.get("/admin/stats", async () => {
    return {
      chunks: await qb.adapter.count(),
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
