# Contributing to Augur

Thanks for your interest in Augur. This document covers what you need to get a working development environment and how to land a change.

## Prerequisites

- Node.js 20 or later
- pnpm 9 or later (`corepack enable && corepack prepare pnpm@9 --activate` if you don't have it)

## Setup

```bash
git clone https://github.com/willgitdata/augur.git
cd augur
pnpm install
pnpm -r --filter './packages/*' build
```

## Layout

This is a pnpm workspace.

- `packages/core`: `@augur-rag/core`, the SDK. Adapter interface, router, chunkers, embedders, rerankers, trace store.
- `packages/server`: `@augur-rag/server`, a Fastify HTTP wrapper around the SDK with `/index`, `/search`, `/traces`, `/health`, `/docs` endpoints.
- `examples/`: runnable examples covering the SDK's main paths (basic search, chunking strategies, custom adapters).

The dashboard (Next.js trace explorer) and eval harness (BEIR runner + bundled corpus) used to live in this repo. They're kept out of tree now to keep the install lean; both remain available in git history.

## Running things

```bash
pnpm dev:server                              # @augur-rag/server on :3001
pnpm --filter example-basic-search start     # one-shot SDK demo
pnpm --filter example-chunking start
pnpm --filter example-custom-adapter start
```

## Testing

```bash
pnpm test                                  # all packages; currently 192 tests in @augur-rag/core
pnpm --filter @augur-rag/core test         # core only

pnpm typecheck                             # typecheck the whole workspace
pnpm build                                 # compile both publishable packages
```

Tests use Node's built-in test runner (no Jest, no vitest). Test files live next to their subjects (e.g. `router.ts` and `router.test.ts`). CI runs `typecheck`, `build`, and `test` on every PR.

## Making a change

1. Open an issue first for anything bigger than a typo or a one-line bugfix. We'd rather agree on the design than have you build something we can't merge.
2. Fork, branch, and make the change. Keep PRs focused; one logical change per PR.
3. Add or update tests for any behavior change. Bare minimum: if the router decision flow changes, add a router test that exercises the new branch.
4. Run `pnpm typecheck` and the core test suite locally before pushing.
5. Open a PR. Describe what changed and why; if it's a routing change, include a sentence about how it would have routed before vs. after.

## Coding conventions

- TypeScript everywhere; strict mode is on.
- Module style is ESM (`"type": "module"`); imports use `.js` extensions even for `.ts` source files (NodeNext-style).
- Comments explain why, not what. The router (`packages/core/src/routing/router.ts`) is a good reference for what useful comments look like.
- Public API additions should land with a JSDoc block on the exported symbol.

## Reporting bugs

Use the bug-report template under "Issues". The most useful bug reports include:

- Reproduction steps (or even better, a minimal failing example)
- The trace from the search call (or the routing decision string)
- Adapter, embedder, and router used
- Node and pnpm versions

## Reporting security issues

Please don't file a public GitHub issue. See [SECURITY.md](./SECURITY.md) for the disclosure process.
