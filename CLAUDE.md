# CLAUDE.md — internal context for AI sessions

This file is written by and for Claude Code sessions working in this repo.
Everything below reflects deliberate decisions, not casual notes — read it
before assuming anything is missing.

If you're a human and you stumbled in here, you can safely read this too.
The user-facing entry points are `README.md`, `SETUP.md` (in the
`tools/dashboard-and-large-corpus` branch), and `MIGRATING.md`.

---

## Repo shape — what's in main and what isn't

The main branch ships **only** the SDK and the optional server wrapper:

```
packages/core/      @augur/core    — the SDK
packages/server/    @augur/server  — Fastify wrapper (optional)
examples/           reference snippets
```

These were intentionally extracted in commit `feffc73` ("chore: extract
dashboard + eval harness; lean SDK-focused repo"):

| Removed from main | Lives in git history at | Purpose |
|---|---|---|
| `apps/dashboard/` | `feffc73^` | Next.js trace-explorer UI for testing routing |
| `evaluations/` | `feffc73^` | BEIR runner + 182-doc/504-query bundled eval. Produced the README's headline numbers. |

The published packages were measured against `evaluations/` exactly as
they ship — when someone asks for the "real eval," that's where it is.

**Don't try to re-add the dashboard or eval harness to main.** That
ship has sailed; they belong as separate concerns. Use the worktree
flow below instead.

---

## How to launch the dashboard locally

A dedicated branch — `tools/dashboard-and-large-corpus`, on origin —
sits on top of `feffc73^` with everything restored plus the testing
QoL upgrades the user asked for. Materialize it with a worktree:

```bash
# From the main repo root:
git fetch origin
git worktree add ../augur-tools tools/dashboard-and-large-corpus

# Or to recreate the local path used previously:
git worktree add /path/to/repo \
    tools/dashboard-and-large-corpus

cd /path/to/repo
pnpm install
pnpm build
```

### Start the dashboard against a seeded in-memory adapter

```bash
# Terminal 1 — server with the diverse 324-doc corpus
cd /path/to/repo
pnpm --filter @augur/evaluations build:large    # one-time
AUGUR_SEED_DEMO=large pnpm dev:server           # serves on :3001

# Terminal 2 — dashboard
cd /path/to/repo
NEXT_PUBLIC_AUGUR_URL=http://localhost:3001 pnpm dev:dashboard
# open http://localhost:3000
```

### Start the dashboard against the user's existing backend

The dashboard talks HTTP to the server, not the database directly,
so any adapter works. Config is via env vars on the server side:

```bash
# Pinecone
AUGUR_ADAPTER=pinecone \
PINECONE_API_KEY=pcsk_xxx \
PINECONE_INDEX_HOST=https://your-index.svc.us-east-1.pinecone.io \
pnpm dev:server

# Turbopuffer (vector + keyword + hybrid)
AUGUR_ADAPTER=turbopuffer \
TURBOPUFFER_API_KEY=tpuf_xxx \
TURBOPUFFER_NAMESPACE=ns \
pnpm dev:server

# pgvector
AUGUR_ADAPTER=pgvector \
DATABASE_URL=postgres://... \
PGVECTOR_TABLE=chunks \
PGVECTOR_DIMENSION=384 \
pnpm dev:server
```

Skip `AUGUR_SEED_*` when pointing at a backend that already has data
— the server uses what's there. See `SETUP.md` on the tools branch
for the full matrix and schema requirements (pgvector needs
`CREATE EXTENSION vector` + a specific table layout).

### Run the dashboard against the **latest** SDK code

The dashboard is locked to its older `@augur/core` (workspace dep at
`feffc73^`), but the protocol is just HTTP — point it at the latest
server from main:

```bash
# Terminal 1 — latest server from main
cd /path/to/repo
pnpm --filter @augur/server dev

# Terminal 2 — dashboard from the tools worktree
cd /path/to/repo
NEXT_PUBLIC_AUGUR_URL=http://localhost:3001 pnpm dev:dashboard
```

This gives the visual UI against the current routing logic.

---

## Corpus options the server understands

The server's seed step (in `packages/server/src/cli.ts` on the tools
branch) reads three env vars in priority order:

| Env var | What it loads |
|---|---|
| `AUGUR_SEED_CORPUS=<path>` | Any user-supplied `.json` (array) or `.jsonl` (one Document per line). |
| `AUGUR_SEED_DEMO=large` | `evaluations/corpus-large.jsonl` — 324 docs across 60+ topics, multi-language entries, RFC/CVE references. |
| `AUGUR_SEED_DEMO=1` | `evaluations/corpus.json` — bundled 182-doc curated corpus. |

Document shape:
```jsonc
{
  "id": "stable-unique-id",
  "content": "indexed and searchable text",
  "metadata": {
    "topic": "rust",       // arbitrary; usable in filter: { topic: "rust" }
    "lang": "ja"           // BCP-47; drives autoLanguageFilter when ON
  }
}
```

To extend the diverse corpus, edit `evaluations/build-large-corpus.ts`
on the tools branch and re-run `pnpm --filter @augur/evaluations
build:large`. The script combines the bundled corpus with hand-written
templates; ID collisions favor bundled.

---

## What's on the tools branch that's NOT in main

Beyond just restoring `apps/dashboard/` and `evaluations/`, the tools
branch adds:

1. **`AUGUR_SEED_CORPUS=<path>`** — server-cli env var that accepts any
   user JSON / JSONL corpus. (Built on top of the existing
   `AUGUR_SEED_DEMO=1` flag.)

2. **`AUGUR_SEED_DEMO=large`** — opts into the 324-doc diverse corpus
   if `corpus-large.jsonl` has been generated.

3. **`evaluations/build-large-corpus.ts`** — the generator that
   produces `corpus-large.jsonl` by combining the bundled 182 docs
   with templated entries across 60+ topics. Includes Japanese,
   Chinese, French, German, Spanish, Korean entries to exercise the
   language filter.

4. **`apps/dashboard/components/StatusBar.tsx`** — auto-refreshing
   status bar that polls `/health` and `/admin/stats` every 5s and
   surfaces the active adapter, embedder, reranker, indexed chunk
   count, and capability flags. Helps users see exactly which
   backend their queries hit before reading any results.

5. **Playground example chips** — six pre-loaded queries that
   demonstrate each routing rule (quoted phrase, specific tokens,
   procedural Q, definitional Q, RFC token, non-English).

6. **`NEXT_PUBLIC_AUGUR_URL`** — the dashboard's browser-side fetch
   target. The original `AUGUR_URL` was a server-only env var that
   wouldn't reach client components; this is the fix.

7. **`SETUP.md`** — the canonical launch + adapter-config doc on the
   tools branch.

---

## Main-branch breaking-change notes (0.2.x)

If a user asks why `something` doesn't compile after upgrading from
0.1.x, the breaking changes are documented in
[MIGRATING.md](MIGRATING.md). The short list:

- `Augur` requires explicit `embedder` (placeholder embedders gone).
- `Augur` no longer defaults to `HeuristicReranker` — pass
  `new LocalReranker()` (or any provider) explicitly. Default is
  `null` now.
- `signals.tokens` → `signals.wordCount`,
  `signals.avgTokenLen` → `signals.avgWordLen`.
- `signals.language` widened from `"en" | "non-en"` to BCP-47 string.
- `HeuristicRouter` defaults `alwaysRerank: true`.
- `trace.adapter` is always the bare adapter name; ad-hoc / cache
  state on `trace.adHoc` and `trace.adHocCacheHit` (typed booleans).
- Async chunkers (`SemanticChunker`, `Doc2QueryChunker`,
  `ContextualChunker`) implement a new `AsyncChunker` interface
  instead of `Chunker`. Compile-time check replaces a runtime trap.
- `Router.decide` gained an optional `hasReranker?: boolean` parameter
  (default `true`) so routing decisions don't claim `reranked: true`
  when no reranker is configured.

---

## Test count + build commands (main branch)

- `pnpm install && pnpm build` — both packages compile clean.
- `pnpm test` — 191 tests across `@augur/core` (was 100 pre-review,
  163 after round 1).
- `pnpm --filter @augur/core test` — core only.
- The eval-smoke regression test asserts `NDCG@10 > 0.65` against a
  synthetic 16-doc/12-query fixture; it's a structural net for the
  routing pipeline. The real eval is in git history at `feffc73^`.

---

## Things that will trip up future sessions

- **The worktree sits on a detached commit (`feffc73^`)** — its
  `@augur/core` is OLD. If the dashboard or eval imports a `@augur/core`
  symbol that only exists on main (e.g., `AsyncChunker`), expect a
  build error in the worktree but not on main. Fix by either
  cherry-picking the missing piece into the tools branch or pointing
  the dashboard at a main-built server (HTTP-only, no SDK coupling).
- **The user's main repo has no `apps/dashboard` or `evaluations`** —
  if you go looking, you'll come up empty. They're on the tools branch.
- **The dashboard polls every 5s** — if it goes silent, that's the
  StatusBar's auto-refresh failing because the server is down. Check
  the server's terminal first, not the dashboard.
- **Port :3001 collisions** — the user runs servers from both repos
  at times. `lsof -i :3001` and kill before starting a fresh one.
- **`@huggingface/transformers` is an optional peer dep** — `pnpm
  install` in the tools worktree always pulls it because the workspace
  has it as a dev dep. A user copying example code out of the repo
  needs to install it explicitly.

---

## File locations cheat sheet

| Topic | Path |
|---|---|
| The SDK (main) | `packages/core/src/` |
| The server (main) | `packages/server/src/` |
| Tests (main) | `packages/core/src/**/*.test.ts` |
| Migration guide | `MIGRATING.md` (main) |
| Changelog | `CHANGELOG.md` (main) |
| Dashboard app | `apps/dashboard/` (tools branch) |
| Dashboard StatusBar | `apps/dashboard/components/StatusBar.tsx` (tools) |
| Server CLI w/ seed support | `packages/server/src/cli.ts` (tools) |
| Eval harness | `evaluations/cli.ts` (tools) |
| Diverse corpus generator | `evaluations/build-large-corpus.ts` (tools) |
| Adapter setup matrix | `SETUP.md` (tools) |

---

## Don't lose this

This file lives on `main`. Future sessions reading the repo's root
will find it. Keep it updated when:

- A new adapter is added → update the SETUP.md matrix and the env-var
  table here.
- The diverse corpus generator changes → update the doc-count + topic
  list above.
- Dashboard launch flow changes → update the "How to launch" section.
- A new optional env var is added to the server CLI → document it in
  the corpus-options table.
