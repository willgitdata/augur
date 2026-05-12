# Releasing

Augur ships as two npm packages from a single monorepo:

- [`@augur-rag/core`](https://www.npmjs.com/package/@augur-rag/core): the SDK
- [`@augur-rag/server`](https://www.npmjs.com/package/@augur-rag/server): optional Fastify HTTP wrapper

Releases are automated via the [release workflow](../.github/workflows/release.yml): pushing a `v*` tag publishes both packages to npm with provenance attestations. The manual flow below is kept as a fallback when CI is unavailable.

## Why order matters

`@augur-rag/server` depends on `@augur-rag/core` at the matching version. When `pnpm publish` runs, the `workspace:*` spec in `packages/server/package.json` is rewritten to the literal version (e.g. `"@augur-rag/core": "0.1.1"`). If you publish `@augur-rag/server` before `@augur-rag/core`, a fresh install fails because npm can't resolve the matching core version.

Core publishes first, server publishes second. The release workflow enforces this order.

## One-time setup

Pick one auth mode. **Trusted publishing is recommended** — no long-lived secret stored anywhere.

### Option A — Trusted publishing (recommended)

Binds each npm package to this repo + workflow. No `NPM_TOKEN` needed. Configure on npmjs.com once per package:

1. Sign in to https://www.npmjs.com as the publisher account.
2. Open the package page (e.g. https://www.npmjs.com/package/@augur-rag/core).
3. **Settings → Publishing access → Trusted publishers → Add GitHub Actions**.
4. Fill in:
   - **Organization or user:** `willgitdata`
   - **Repository:** `augur`
   - **Workflow filename:** `release.yml`
   - **Environment name:** `npm-publish`
5. Repeat for `@augur-rag/server`.

Reference: [npm docs on trusted publishing](https://docs.npmjs.com/trusted-publishers).

### Option B — Classic `NPM_TOKEN` secret

Fallback when trusted publishing isn't an option (e.g. you don't own the package on npm yet).

1. On npmjs.com → **Account → Access Tokens → Generate New Token → Granular Access Token**.
2. Scope: read+write on `@augur-rag/core` and `@augur-rag/server`. Set an expiry you'll actually rotate.
3. On GitHub: **Repo settings → Secrets and variables → Actions → New repository secret** named `NPM_TOKEN`.

The workflow reads either auth source via `NODE_AUTH_TOKEN`. Once trusted publishing is configured, the token isn't consulted and can be deleted from the secrets.

### Optional — manual approval gate

The workflow runs in a GitHub Environment named `npm-publish`. To require a human approval before publish, add required reviewers under **Repo settings → Environments → npm-publish**.

## Cutting a release (automated)

```bash
# 0. Make sure main is green and you're on it.
git checkout main && git pull --ff-only origin main

# 1. Bump versions in BOTH packages. They share a version.
pnpm -r --filter './packages/*' exec npm version patch --no-git-tag-version

# 2. Update CHANGELOG.md
#    - Move [Unreleased] → [<new-version>] - YYYY-MM-DD
#    - Add a fresh empty [Unreleased] header
#    - Update bottom-of-file link refs

# 3. Commit and tag.
git add packages/core/package.json packages/server/package.json CHANGELOG.md
git commit -m "release: v<new-version>"
git tag -a v<new-version> -m "v<new-version>"

# 4. Push the tag — this triggers the release workflow.
git push origin main
git push origin v<new-version>
```

What the workflow does:

1. Checks out the tag.
2. Verifies the tag (`v0.1.2`) matches both packages' `version` fields. Mismatch fails loud.
3. `pnpm install --frozen-lockfile && pnpm build && pnpm typecheck && pnpm test`.
4. `pnpm --filter @augur-rag/core publish --provenance --access public`.
5. `pnpm --filter @augur-rag/server publish --provenance --access public`.
6. Posts a step-summary link to both npm pages and the commit SHA.

The `--provenance` flag attaches an OIDC-signed attestation. The published package page on npm displays a "Built and signed on GitHub Actions" badge linking to the exact workflow run and commit. See [npm docs on provenance](https://docs.npmjs.com/generating-provenance-statements).

## Dry runs

To rehearse the workflow without uploading anything, run it from the **Actions** tab via **Run workflow** with `dry_run: true`. Each `pnpm publish` step runs with `--dry-run`.

## Manual release (fallback)

Use this only if the workflow is unavailable. Everything below mirrors what the workflow does, but on your laptop.

```bash
# 0–3. Same as the automated flow above, through tagging.

# 4. Run the full prepublish gate locally.
pnpm install --frozen-lockfile
pnpm build && pnpm typecheck && pnpm test

# 5. Verify the tarballs are not empty BEFORE publishing.
#    The 0.1.0 release shipped with an empty dist/ because of a stale-cache
#    issue; this gate catches a repeat.
(cd packages/core   && pnpm pack --pack-destination /tmp)
(cd packages/server && pnpm pack --pack-destination /tmp)
tar -tzf /tmp/augur-rag-core-<new-version>.tgz   | grep -c 'dist/.*\.js$'   # expect 25+
tar -tzf /tmp/augur-rag-server-<new-version>.tgz | grep -c 'dist/.*\.js$'   # expect 3+

# 6. Publish, IN ORDER. Each package has publishConfig.access=public baked in.
pnpm --filter @augur-rag/core   publish --ignore-scripts --no-git-checks
pnpm --filter @augur-rag/server publish --ignore-scripts --no-git-checks

# 7. Push the tag if you haven't already.
git push origin v<new-version>

# 8. Create a GitHub Release for the tag, body = the CHANGELOG entry from step 2.
```

Note: manual publishes do **not** get the provenance attestation. The npm package page will not show the "Built and signed" badge for those versions. Prefer the automated workflow.

## What gets published

Both packages have a strict `files` allowlist in their `package.json`. The published tarball contains only:

- `dist/`: built JS, type declarations, and sourcemaps
- `README.md`
- `LICENSE`
- `package.json`

No tests, no fixtures, no eval data, no `tsconfig.json`, no source `.ts` files. Verify yourself before publishing:

```bash
cd packages/core && pnpm pack --pack-destination /tmp
tar -tzf /tmp/augur-rag-core-<new-version>.tgz | sort
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
| Workflow fails at "Verify tag matches package versions" | You forgot to bump one of the package.json `version` fields before tagging. |
| `402 Payment Required` on first publish | `publishConfig.access=public` missing. Both packages have it; check you didn't delete it. |
| `npm install @augur-rag/server` fails to resolve `@augur-rag/core@X` | Server published before core. The workflow always publishes in the right order — only happens with the manual fallback. |
| `Cannot find module '@huggingface/transformers'` for users | Peer dep is optional but required for `LocalEmbedder` and `LocalReranker`. README documents this; users must `npm install @huggingface/transformers` separately. |
| Tarball contains source `.ts` files | Build wasn't run before publish, or `files` allowlist was changed. `prepublishOnly` and the workflow should catch this. |
| Tarball has only `package.json`, `README.md`, `LICENSE` (no `dist/`) | Stale `tsconfig.tsbuildinfo` made tsc skip emitting. The `clean` scripts now remove it. |
| `npm ERR! 403 Forbidden ... unable to authenticate` | Trusted publishing isn't configured on the package yet, and there's no `NPM_TOKEN` secret. Pick one of the auth modes above. |
| Provenance badge doesn't appear on npm | Workflow ran without `id-token: write` permission, or `--provenance` was dropped. Check the workflow file. |
