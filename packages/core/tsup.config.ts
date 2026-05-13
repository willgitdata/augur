import { defineConfig } from "tsup";

/**
 * Dual ESM + CJS build via tsup, retaining the four sub-path exports
 * the package shipped with under plain tsc:
 *
 *   "."          — root index
 *   "./adapters" — adapter exports
 *   "./chunking" — chunker exports
 *   "./routing"  — routing exports
 *
 * tsup emits `.js` for ESM and `.cjs` for CJS. Type declarations are
 * generated per entry; tsup adds a matching `.d.cts` for the CJS side
 * so each `"require"` exports map line points at real `.d.cts` files.
 *
 * `sourcemap` and `splitting: false` mirror what most production
 * libraries do at this scale — splitting would chunk shared
 * dependencies across multiple files and break the simple
 * `dist/index.cjs` / `dist/adapters/index.cjs` layout the package
 * exports map references.
 */
export default defineConfig({
  entry: {
    index: "src/index.ts",
    "adapters/index": "src/adapters/index.ts",
    "chunking/index": "src/chunking/index.ts",
    "routing/index": "src/routing/index.ts",
  },
  format: ["esm", "cjs"],
  // tsup's DTS pass spawns a fresh tsc that doesn't tolerate the
  // `--incremental` flag tsconfig.base.json sets for the development
  // `tsc --noEmit` typecheck path. Override here rather than rip it
  // out of the base — incremental is real value for the watch path.
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
  // `@huggingface/transformers` is an optional peer; never bundle it.
  external: ["@huggingface/transformers"],
  target: "node20",
});
