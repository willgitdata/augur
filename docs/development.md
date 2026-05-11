# Development Guide

Everything you need to run, hack on, and ship Augur.

## Prerequisites

- Node.js ≥ 20
- pnpm ≥ 9 (`corepack enable && corepack prepare pnpm@9 --activate`)
- Optional: Docker, for `docker compose up`

We use pnpm because:

- Workspace dependencies are first-class, no symlink shenanigans.
- Disk usage is dramatically lower than npm or yarn for monorepos.
- Lockfile drift is easier to review.

## First-time setup

```bash
git clone https://github.com/willgitdata/augur.git
cd augur
pnpm install
pnpm build           # builds packages/core then packages/server
```

Two publishable packages, three runnable examples. A trace-explorer dashboard (Next.js) and the eval harness (BEIR runner) used to live in this repo and are kept out of tree now; both remain available in git history.

## Common workflows

### Run the example

```bash
pnpm --filter example-basic-search start
```

You should see output showing each query, the chosen strategy, the reasons, and the top results.

### Run the server

```bash
pnpm dev:server
# → http://localhost:3001
# → http://localhost:3001/docs    (Swagger UI)
# → http://localhost:3001/health  (config dump)
```

The server runs with `InMemoryAdapter` and `LocalEmbedder` (Xenova/all-MiniLM-L6-v2, ~22MB) by default. To swap the storage adapter or embedder model:

```bash
AUGUR_ADAPTER=pgvector \
DATABASE_URL=postgres://localhost/qb \
AUGUR_LOCAL_MODEL=Xenova/bge-small-en-v1.5 \
pnpm dev:server
```

Supported adapter env values: `in-memory` (default), `pinecone`, `turbopuffer`, `pgvector`.

For hosted embedders, implement the `Embedder` interface and import a custom `buildServer({ embedder })`. See examples.md §5.

### Run tests

```bash
pnpm test                              # all packages
pnpm --filter @augur-rag/core test     # just core
```

Tests use Node's built-in test runner. No Jest, no vitest. The test files live next to their subjects (`router.ts` and `router.test.ts`).

### Typecheck without building

```bash
pnpm typecheck
```

Useful in CI as a fast gate before the slower `build`.

### Format

```bash
pnpm format
```

Prettier with default config. We don't lint with ESLint; TypeScript + Prettier covers 90% of the value at this size.

## Adding a new component

### A new adapter

1. Create `packages/core/src/adapters/myadapter.ts`.
2. Either implement `VectorAdapter` directly, or extend `BaseAdapter` to inherit RRF-based hybrid for free.
3. Add a `capabilities` block that is honest.
4. Export it from `packages/core/src/adapters/index.ts` and `packages/core/src/index.ts`.
5. Optional: add an env-var path in `packages/server/src/cli.ts` so the server CLI can use it.
6. Write a test that round-trips upsert + searchVector at minimum.

### A new chunker

1. Create `packages/core/src/chunking/mychunker.ts`.
2. Export from `chunking/index.ts` and the root `index.ts`.
3. Two interfaces: implement `Chunker` (one method, `chunk(doc): Chunk[]`) for sync chunkers, or `AsyncChunker` (one method, `chunkAsync(doc): Promise<Chunk[]>`) for chunkers that need an embedder, an LLM call, or any other async dependency. They are separate interfaces; async chunkers do not implement `Chunker`. `chunkDocument()` is the polymorphic helper the orchestrator uses internally.

### A new router

The whole point of the `Router` interface is that you can do this:

```ts
class MyRouter implements Router {
  readonly name = "my-v1";
  decide(req, caps) {
    // ... your logic ...
    return { strategy: "vector", reasons: [...], signals, reranked: false };
  }
}

const augr = new Augur({ router: new MyRouter() });
```

Two principles:

- Always populate `reasons`. If a trace consumer can't see why, you're hiding bugs.
- Respect `caps`. Don't return `keyword` if `caps.keyword === false`; degrade gracefully.

### A new reranker

Implement `Reranker.rerank(query, results, topK)`. Examples:

- A hosted cross-encoder (Cohere, Voyage, Jina) over HTTP: implement the `Reranker` interface directly against the provider's SDK. See examples.md §5.
- A local ONNX model via `onnxruntime-node`: same shape, different transport.

## Project conventions

- TypeScript strict. `noUncheckedIndexedAccess` is on. The explicit `!` on array access is intentional: we want you to think about empty arrays.
- No barrel files inside subdirectories unless they're public API surface. Direct imports are easier to grep.
- Files explain themselves at the top. Every non-trivial file starts with a comment explaining what it is and why.
- Inline reasons for non-obvious choices. If a future contributor would ask "why did they do that?", the answer is in the comment.
- Prefer composition over configuration. A boolean flag should be an object you pass in.

## Releasing

The full publish runbook lives in [releasing.md](./releasing.md). High-level: bump the version, build, publish core first, server second, tag, GitHub Release.

## Troubleshooting

**`Cannot find module '@augur-rag/core'`**: run `pnpm build` first. The core package compiles to `dist/`; consumers import from there.

**Tests pass locally but fail in CI**: the most common cause is timing-sensitive trace assertions. Use `assert.ok(trace.totalMs >= 0)`, not `> 0`.

**The server returns 500 on `/search`**: check `/health`. The `capabilities` block tells you what the configured adapter supports. If you forced a strategy the adapter can't do, that's the cause.

**`LocalEmbedder` first request is slow (~5-10s)**: that's the one-time ONNX model download to `~/.cache/huggingface/hub`. Subsequent requests use the cached model and are sub-second.

## Deployment

### Server (Docker)

```bash
docker build -t augur-server .
docker run -p 3001:3001 \
  -e AUGUR_ADAPTER=pgvector \
  -e DATABASE_URL=... \
  -e AUGUR_API_KEY=$(openssl rand -hex 32) \
  augur-server
```

### Behind a reverse proxy

Most teams will route `/api/augur/*` (or whatever prefix) at `@augur-rag/server` and handle HTTPS, auth, and rate limits at the proxy. The server has an optional `AUGUR_API_KEY` for shared-secret auth. For anything more sophisticated (per-user keys, OAuth) use your reverse proxy or wrap the Fastify app.

## Contributing

1. Fork, branch, PR.
2. Keep PRs small. One adapter, one router, one bug fix per PR.
3. Tests required for new code paths. We aim for "high confidence the happy path works", not 100% line coverage.
4. Match the file-header convention: every new file gets a 1–2 paragraph "what is this and why" block.
