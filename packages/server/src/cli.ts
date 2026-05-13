#!/usr/bin/env node
/**
 * CLI entry point — `npx augur-server`.
 *
 * Reads minimal env vars and starts the server. We deliberately keep config
 * to env vars (no flags, no config files) for the MVP. Docker-friendly,
 * twelve-factor, and matches the "Vercel-feel" goal.
 */
import {
  InMemoryAdapter,
  LocalEmbedder,
  PgVectorAdapter,
  PineconeAdapter,
  TurbopufferAdapter,
  type Embedder,
  type VectorAdapter,
} from "@augur-rag/core";
import { buildServer } from "./server.js";

/**
 * Local shim for the bits of the `pg` package this CLI uses. `pg` is not a
 * declared dep so we can't import its types; this is the minimum surface
 * we need (`Client` constructor + `connect` / `query`). Lets the
 * dynamic-import path be `Promise<PgModule>` instead of `Promise<any>`.
 */
interface PgModule {
  default: {
    Client: new (opts: { connectionString: string }) => {
      connect(): Promise<void>;
      query<T = unknown>(
        sql: string,
        params?: unknown[]
      ): Promise<{ rows: T[] }>;
      end(): Promise<void>;
    };
  };
}

async function main() {
  const port = parseInt(process.env.PORT ?? "3001", 10);
  // Default to loopback so a fresh `docker compose up` or `npx augur-server`
  // doesn't expose admin endpoints on the LAN before the operator has chosen
  // an auth posture. Set HOST=0.0.0.0 explicitly to bind all interfaces.
  const host = process.env.HOST ?? "127.0.0.1";
  const apiKey = process.env.AUGUR_API_KEY;
  const corsEnv = process.env.AUGUR_CORS;

  const adapter = await pickAdapter();
  const embedder = pickEmbedder();

  const app = buildServer({
    adapter,
    embedder,
    apiKey,
    cors: parseCors(corsEnv),
    // Opt into language-aware filtering via env var. See AugurOptions
    // JSDoc for when this is the right call (localized canonical answers
    // per language) vs the wrong call (cross-language canonical content).
    autoLanguageFilter: process.env.AUGUR_AUTO_LANGUAGE_FILTER === "1",
  });

  // Loud warnings for the two configurations that have bitten users in the
  // past: binding all interfaces without auth, or with permissive CORS.
  if (!isLoopback(host) && !apiKey) {
    app.log.warn(
      `Augur is binding ${host}:${port} with no AUGUR_API_KEY set. ` +
        "Destructive endpoints (/admin/*, DELETE /traces) are disabled, " +
        "but /search, /index, and /traces are exposed. Set AUGUR_API_KEY " +
        "or bind to 127.0.0.1 to silence this warning."
    );
  }
  if (corsEnv === "*" || corsEnv === "true") {
    app.log.warn(
      "AUGUR_CORS reflects every origin. Combined with an apiKey in a " +
        "header this still permits XHR-style abuse from any page; only use " +
        "this on trusted networks."
    );
  }

  try {
    await app.listen({ port, host });
    app.log.info(
      `Augur ready at http://${host}:${port}  (adapter=${adapter.name}, embedder=${embedder.name})`
    );
    app.log.info(`OpenAPI docs: http://${host}:${port}/docs`);
  } catch (e) {
    app.log.error(e);
    process.exit(1);
  }
}

/** Loopback addresses we don't warn about — single host, no public exposure. */
function isLoopback(host: string): boolean {
  return host === "127.0.0.1" || host === "::1" || host === "localhost";
}

/**
 * AUGUR_CORS parsing. Defaults to `false` (no cross-origin) to match the
 * library default. `*` and `true` map to "reflect any origin" (see warning
 * above). Anything else is treated as a comma-separated allowlist.
 */
function parseCors(value: string | undefined): boolean | string | string[] {
  if (value === undefined || value === "" || value === "false") return false;
  if (value === "*" || value === "true") return true;
  if (value.includes(",")) return value.split(",").map((s) => s.trim()).filter(Boolean);
  return value;
}

function pickEmbedder(): Embedder {
  // Local ONNX sentence-transformer is the only built-in embedder. For hosted
  // providers (OpenAI, Cohere, Voyage, etc) implement the Embedder interface
  // and run your own server build that wires it in — see docs/examples.md.
  const opts: { model?: string } = {};
  if (process.env.AUGUR_LOCAL_MODEL) opts.model = process.env.AUGUR_LOCAL_MODEL;
  return new LocalEmbedder(opts);
}

async function pickAdapter(): Promise<VectorAdapter> {
  const kind = process.env.AUGUR_ADAPTER ?? "in-memory";
  if (kind === "pinecone") {
    return new PineconeAdapter({
      indexHost: requireEnv("PINECONE_INDEX_HOST"),
      apiKey: requireEnv("PINECONE_API_KEY"),
      namespace: process.env.PINECONE_NAMESPACE ?? "default",
    });
  }
  if (kind === "turbopuffer") {
    return new TurbopufferAdapter({
      apiKey: requireEnv("TURBOPUFFER_API_KEY"),
      namespace: requireEnv("TURBOPUFFER_NAMESPACE"),
    });
  }
  if (kind === "pgvector") {
    // `pg` is loaded dynamically so it's not a runtime dep of the server
    // package — users who pick the pgvector adapter install pg themselves.
    // The `as string` literal-widening dodges TS's static
    // module-resolution check (pg isn't declared anywhere); the
    // `as Promise<PgModule>` gives us real types instead of `any`.
    const pg = await (import("pg" as string) as Promise<PgModule>).catch(() => null);
    if (!pg) {
      throw new Error(
        "Install 'pg' to use the pgvector adapter: npm install pg"
      );
    }
    const client = new pg.default.Client({ connectionString: requireEnv("DATABASE_URL") });
    await client.connect();
    return new PgVectorAdapter({
      // Method form (vs arrow) so the `<T>` generic on PgClient.query
      // flows through to pg.Client.query — an arrow would lock T at
      // declaration time and force `unknown` rows downstream.
      client: {
        query<T = unknown>(sql: string, params?: unknown[]) {
          return client.query<T>(sql, params);
        },
      },
      table: process.env.PGVECTOR_TABLE ?? "chunks",
      dimension: parseInt(process.env.PGVECTOR_DIMENSION ?? "1536", 10),
    });
  }
  return new InMemoryAdapter();
}

function requireEnv(key: string): string {
  const v = process.env[key];
  if (!v) throw new Error(`Required env var ${key} is not set`);
  return v;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
