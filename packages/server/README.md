# @augur-rag/server

The HTTP API for Augur: adaptive retrieval orchestration for RAG and semantic search. A Fastify wrapper around [@augur-rag/core](https://www.npmjs.com/package/@augur-rag/core) with OpenAPI docs at `/docs` and a search trace returned with every response.

## Install and run

```bash
npm install -g @augur-rag/server
# also install the peer dep, since augur-server uses LocalEmbedder by default:
npm install -g @huggingface/transformers
augur-server
```

`@huggingface/transformers` is an optional peer of `@augur-rag/core`. The `augur-server` CLI uses `LocalEmbedder` by default (on-device ONNX, no API keys), which needs the peer dep. If you wire your own `buildServer({ embedder })` against a hosted provider (OpenAI, Cohere, Voyage, etc.), you can skip it.

## Or as a library inside your own service

```ts
import { buildServer } from "@augur-rag/server";
import { LocalEmbedder } from "@augur-rag/core";

const app = buildServer({
  embedder: new LocalEmbedder(),
  // adapter, reranker, chunker, router, autoLanguageFilter: all optional
});

await app.listen({ port: 3001 });
```

## What you get

- `POST /search`: runs the auto-routing pipeline, returns `{ results, trace }`
- `POST /index`: indexes documents in batch
- `GET /traces`: recent search traces (for trace explorers / observability backends)
- `GET /health`: capability dump
- `GET /docs`: interactive Swagger UI
- `GET /openapi.json`: OpenAPI 3 spec

## Configuration via environment variables

| Var | Default | Purpose |
| --- | --- | --- |
| `PORT` | `3001` | HTTP listen port |
| `HOST` | `127.0.0.1` | Bind address. Set to `0.0.0.0` to expose on all interfaces (the server logs a warning if you do this without `AUGUR_API_KEY`). |
| `AUGUR_ADAPTER` | `in-memory` | `in-memory` \| `pinecone` \| `turbopuffer` \| `pgvector` |
| `AUGUR_API_KEY` | *(unset)* | When set, all endpoints except `/health`, `/openapi.json`, `/docs` require `x-api-key: <key>`. Destructive endpoints (`/admin/*`, `DELETE /traces`) return **503** until this is set — they never run unauthenticated. |
| `AUGUR_CORS` | `false` | Cross-origin allowlist. Use a single origin (`https://app.example`), a comma-separated list, or `*` to reflect any origin (only safe behind `AUGUR_API_KEY` on a trusted network). |
| `AUGUR_LOCAL_MODEL` | `Xenova/all-MiniLM-L6-v2` | Override the default `LocalEmbedder` model |
| `AUGUR_AUTO_LANGUAGE_FILTER` | `0` | Set `1` to filter results to the query's detected language (with soft fallback) |
| `PINECONE_INDEX_HOST` / `PINECONE_API_KEY` / `PINECONE_NAMESPACE` | unset | When `AUGUR_ADAPTER=pinecone` |
| `TURBOPUFFER_API_KEY` / `TURBOPUFFER_NAMESPACE` | unset | When `AUGUR_ADAPTER=turbopuffer` |
| `DATABASE_URL` / `PGVECTOR_TABLE` / `PGVECTOR_DIMENSION` | unset | When `AUGUR_ADAPTER=pgvector` (also `npm i pg` in your project) |

### Security defaults

The server ships with conservative defaults so a fresh `npx augur-server` or `docker compose up` isn't a foot-gun:

- Binds to `127.0.0.1` unless `HOST` is set explicitly. No LAN exposure without intent.
- CORS is `false` (no cross-origin). Opt in via `AUGUR_CORS`.
- `/admin/clear`, `/admin/stats`, and `DELETE /traces` return `503 admin endpoints disabled` until `AUGUR_API_KEY` is set. They never run unauthenticated, even on loopback.
- When `AUGUR_API_KEY` is set, every endpoint except `/health`, `/openapi.json`, and `/docs` requires the header.

## Learn more

- [Project README](https://github.com/willgitdata/augur#readme): pitch, BEIR comparison table, and quick start
- [Architecture](https://github.com/willgitdata/augur/blob/main/docs/architecture.md): how the orchestrator, router, adapters, chunkers, and rerankers fit together
- [API reference](https://github.com/willgitdata/augur/blob/main/docs/api-reference.md): SDK and HTTP API reference

## License

MIT. See [LICENSE](./LICENSE).
