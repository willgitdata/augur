# Contributing

Thanks for poking at Augur.

## Setup

```bash
git clone https://github.com/willgitdata/augur.git
cd augur
pnpm install
pnpm build
```

Node 20+ and pnpm 9+. `corepack enable && corepack prepare pnpm@9 --activate` if you don't have pnpm.

## Run things

```bash
pnpm dev:server                              # @augur-rag/server on :3001
pnpm --filter example-basic-search start
pnpm --filter example-chunking start
pnpm --filter example-custom-adapter start
```

## Test

```bash
pnpm test          # all packages
pnpm typecheck
pnpm build
```

Tests use Node's built-in test runner. Test files live next to their subjects (`router.ts` next to `router.test.ts`).

## PRs

Open an issue first for anything bigger than a small fix; happy to talk through the design before you build. Keep PRs focused on one thing. If the routing flow changes, add a router test that exercises the new branch.

For releases, the publish workflow runs on `v*` tag push; see [`.github/workflows/release.yml`](./workflows/release.yml).

## Security

Don't file public issues for security stuff. See [SECURITY.md](./SECURITY.md).
