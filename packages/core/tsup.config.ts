import { defineConfig } from "tsup";

/**
 * tsup multi-target build for @augur-rag/core.
 *
 * Two pipelines run side-by-side:
 *
 * 1. **Node ESM + CJS** (the package's primary distribution). Four
 *    entry points (`.`, `./adapters`, `./chunking`, `./routing`),
 *    each emitted as both `.js` (ESM) and `.cjs` (CJS), with type
 *    declarations split into `.d.ts` and `.d.cts`. `target: node20`
 *    matches `engines.node`.
 *
 * 2. **Browser / edge bundle.** A single ESM file at
 *    `dist/browser/index.js` targeted at browsers + edge runtimes
 *    (Cloudflare Workers, Vercel Edge, Deno, Bun). No more `node:*`
 *    runtime imports anywhere in the core source — the previous
 *    `node:crypto` usages were refactored to Web Crypto, available
 *    in every runtime we target.
 *
 *    The browser bundle is picked up via the `"browser"` field +
 *    a `"./browser"` subpath in `exports`. `@huggingface/transformers`
 *    is still externalised — users wiring `LocalEmbedder` /
 *    `LocalReranker` in browsers install the package themselves
 *    (transformers.js works in browsers; we just don't bundle a
 *    100 MB peer dep into our 100 KB core).
 *
 * tsup's DTS pass overrides `incremental: false` because tsup spawns
 * a fresh tsc that doesn't tolerate the `--incremental` flag from
 * `tsconfig.base.json` (kept for the dev typecheck path).
 */
export default defineConfig([
  // ----- Node target (ESM + CJS, multi-entry) -----
  {
    entry: {
      index: "src/index.ts",
      "adapters/index": "src/adapters/index.ts",
      "chunking/index": "src/chunking/index.ts",
      "routing/index": "src/routing/index.ts",
      "integrations/langchain": "src/integrations/langchain.ts",
      "integrations/llamaindex": "src/integrations/llamaindex.ts",
      "integrations/vercel-ai": "src/integrations/vercel-ai.ts",
    },
    format: ["esm", "cjs"],
    dts: {
      compilerOptions: {
        incremental: false,
        composite: false,
        declarationMap: false,
      },
    },
    splitting: false,
    sourcemap: true,
    clean: true,
    external: ["@huggingface/transformers"],
    target: "node20",
  },
  // ----- Browser / edge target (ESM only, single entry) -----
  {
    entry: { "browser/index": "src/index.ts" },
    format: ["esm"],
    dts: {
      compilerOptions: {
        incremental: false,
        composite: false,
        declarationMap: false,
      },
    },
    splitting: false,
    sourcemap: true,
    clean: false, // keep the Node-target output of the first config
    external: ["@huggingface/transformers"],
    platform: "browser",
    target: "es2022",
  },
]);
