# Releasing

Augur ships as two npm packages from a single monorepo:

- [`@augur-rag/core`](https://www.npmjs.com/package/@augur-rag/core) — the SDK
- [`@augur-rag/server`](https://www.npmjs.com/package/@augur-rag/server) — optional Fastify HTTP wrapper

This file documents the exact sequence to release a new version. There is no release automation yet; expect that once we ship every few weeks instead of once.

## Why order matters

`@augur-rag/server` depends on `@augur-rag/core` at the matching version. When `pnpm publish` runs, the `workspace:*` spec in `packages/server/package.json` is rewritten to the literal version (e.g. `"@augur-rag/core": "0.1.0"`). **If you publish `@augur-rag/server` before `@augur-rag/core`, the install on a fresh machine fails:** npm cannot resolve `@augur-rag/core@0.1.0` because it does not exist yet.

Core publishes first, server publishes second. Always.

## Steps

```bash
# 0. Make sure main is green and you're on it.
git checkout main && git pull
pnpm install --frozen-lockfile

# 1. Bump versions in BOTH package.json files. They share a version.
#    Example for a patch release:
pnpm -r --filter './packages/*' exec npm version patch --no-git-tag-version

# 2. Refresh the lockfile and run the full prepublish gate locally.
pnpm install
pnpm build && pnpm typecheck && pnpm test

# 3. Update CHANGELOG.md
#    - Move [Unreleased] → [<new-version>] - YYYY-MM-DD
#    - Add a fresh empty [Unreleased] header
#    - Update the bottom-of-file link refs
#    Commit the version bump + CHANGELOG.

git add packages/core/package.json packages/server/package.json CHANGELOG.md
git commit -m "release: v<new-version>"
git push origin main

# 4. Tag the release commit.
git tag v<new-version>
git push --tags

# 5. Publish, IN ORDER. Each package has publishConfig.access=public
#    baked in, so --access flags are not needed.
pnpm --filter @augur-rag/core publish
pnpm --filter @augur-rag/server publish

# 6. Create a GitHub Release for the tag, body = the CHANGELOG entry
#    you just wrote in step 3.
```

The root `pnpm release` script wires steps 5+6 together for you:

```bash
pnpm release           # publishes core, then server, refusing if dirty
```

It does NOT bump versions or tag; that's still manual so a clueless future-me can't accidentally publish the wrong version.

## What gets published

Both packages have a strict `files` allowlist in their `package.json`. The published tarball contains only:

- `dist/` — built JS + type declarations + sourcemaps
- `README.md`
- `LICENSE`
- `package.json`

No tests, no fixtures, no eval data, no `tsconfig.json`, no source `.ts` files. Verify yourself before publish:

```bash
cd packages/core && pnpm pack --pack-destination /tmp
tar -tzf /tmp/augur-core-<new-version>.tgz | sort
```

## Post-publish smoke check

```bash
# In a fresh tmp dir, install from the registry and verify the API surface.
mkdir -p /tmp/augur-smoke && cd /tmp/augur-smoke
npm init -y
npm install @augur-rag/core @huggingface/transformers
node -e "import('@augur-rag/core').then(m => console.log(Object.keys(m).sort()))"
```

You should see the full export list (`Augur`, `LocalEmbedder`, `LocalReranker`, `HeuristicRouter`, all the adapters, etc.). If anything is missing, the build or `files` allowlist is wrong.

## What can go wrong

| Symptom | Likely cause |
| --- | --- |
| `402 Payment Required` on first publish | `publishConfig.access=public` missing. Both packages have it; check you didn't delete it. |
| `npm install @augur-rag/server` fails to resolve `@augur-rag/core@X` | You published server before core. Re-publish core at the same version; npm will then find it. |
| `Cannot find module '@huggingface/transformers'` for users | Peer dep is *optional* but required for `LocalEmbedder` / `LocalReranker`. README documents this; the user has to `npm install @huggingface/transformers` separately. |
| Tarball contains source `.ts` files | Build wasn't run before publish, or `files` allowlist was changed. `prepublishOnly` should catch this. |
