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

## Good first issues

If you're new to the codebase and want a self-contained chunk of work to chew on, any of these are scoped to a single PR and don't require a deep tour of the rest of the system. Open an issue or PR with the title prefix in brackets to claim one.

- **[adapter] Weaviate adapter.** Hybrid is native (Weaviate has BM25 + vector + `nearText` in one query). Mirror the shape of [`PgVectorAdapter`](../packages/core/src/adapters/pgvector.ts); five methods + the capabilities block. Look at [`TurbopufferAdapter`](../packages/core/src/adapters/turbopuffer.ts) for the "native hybrid override" pattern.
- **[adapter] LanceDB adapter.** Local-first columnar vector store. Useful for desktop/CLI RAG. `@lancedb/lancedb` exposes a clean Node API.
- **[adapter] MongoDB Atlas Vector Search adapter.** Aggregation-pipeline-based `$vectorSearch` stage. Vector only (Atlas has no built-in BM25). Set `capabilities.keyword = false` and let the router fall through.
- **[adapter] OpenSearch / Elasticsearch adapter.** Both have native vector + native BM25 + native hybrid via `rrf`. One adapter can cover both with a vendor flag.
- **[adapter] Redis (RediSearch with vector indexing).** Hybrid via HYBRID + KNN syntax. Useful for teams already running Redis.
- **[adapter] Vespa adapter.** Native hybrid via `nearestNeighbor` + WAND/BM25. Niche but production-grade.
- **[binding] Mastra retriever binding.** Mastra ships its own retriever interface; we should expose `AugurRetriever` for the agent-framework crowd.
- **[binding] AWS Bedrock Knowledge Base hosted-retriever binding.** Wrap Bedrock's KB API behind the `VectorAdapter` interface for users who don't want to run their own vector DB.
- **[chunker] Markdown-aware chunker.** Don't split inside fenced code blocks; respect heading boundaries. Strict superset of `SentenceChunker` for prose-heavy docs.
- **[chunker] Code-aware chunker.** Use [`tree-sitter`](https://github.com/tree-sitter/node-tree-sitter) to chunk at function / class boundaries. Worth a separate package because tree-sitter is a native dep.
- **[reranker] BGE / Cohere / Voyage / Jina hosted reranker reference impls.** Each is ~20 lines against the provider SDK. Worth their own files under `packages/core/src/reranking/` so the docs link is a real file, not a snippet.
- **[router] Domain-specific routers.** A `CodeRouter` (forces keyword + BM25 on snake_case / dotted-path / `()` tokens) or a `ConversationalRouter` (always-vector for chatbot-style queries) would let users skip the heuristic-tuning step for known query shapes.
- **[observability] OpenTelemetry exporter.** Wrap `Tracer` so spans land in an OTel collector. ~30 lines per ARCHITECTURE.md.
- **[observability] LangSmith / Phoenix / Langfuse bridges.** Same shape — wrap `TraceStore.push` to also forward to the target backend.

Smaller drive-bys are also welcome:

- **[docs] Tracer screenshot or GIF.** The trace decision + reasons rendered in a terminal would be a stronger README artifact than another paragraph of prose.
- **[docs] StackBlitz / CodeSandbox embed.** A one-click "open it and try a query" link from the README beats any install instruction.

## Security

Don't file public issues for security stuff. See [SECURITY.md](./SECURITY.md).

## Maintainer-only: repo settings

The following are GitHub-side, not in the repo, so they need to be applied via the GitHub UI or the `gh` CLI. Captured here so they're not lost.

```bash
# Enable Discussions so people have somewhere to ask questions
# that aren't issues. Indexes in search results; reduces issue noise.
gh api -X PATCH repos/willgitdata/augur --field has_discussions=true

# Topics make the repo show up in GitHub's topic-page traffic and on
# the npm package page. Keep them tight — overlapping topics are penalised.
gh api -X PUT repos/willgitdata/augur/topics --input - <<'JSON'
{
  "names": [
    "rag", "retrieval-augmented-generation", "rag-pipeline",
    "vector-search", "semantic-search", "bm25", "hybrid-search",
    "reranking", "cross-encoder", "embeddings",
    "pgvector", "pinecone", "turbopuffer",
    "typescript", "onnx", "huggingface",
    "contextual-retrieval", "mteb", "beir", "information-retrieval"
  ]
}
JSON
```
