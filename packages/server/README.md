# @augur/server

The HTTP API for Augur — adaptive retrieval orchestration for RAG.

```bash
npm install -g @augur/server
augur-server
```

Or as a dependency in your own service:

```ts
import { buildServer } from "@augur/server";
import { LocalEmbedder } from "@augur/core";

const app = buildServer({
  embedder: new LocalEmbedder(),
  // adapter, reranker, chunker, router, autoLanguageFilter — all optional
});

await app.listen({ port: 3001 });
```

## What you get

- `POST /search` — runs the auto-routing pipeline, returns `{ results, trace }`
- `POST /index` — indexes documents in batch
- `GET /traces` — recent search traces (for the dashboard)
- `GET /health` — capability dump
- `GET /docs` — interactive Swagger UI
- `GET /openapi.json` — OpenAPI 3 spec

## Configuration via environment variables

| Var | Default | Purpose |
| --- | --- | --- |
| `PORT` | `3001` | HTTP listen port |
| `HOST` | `0.0.0.0` | Bind address |
| `AUGUR_ADAPTER` | `in-memory` | `in-memory` \| `pinecone` \| `turbopuffer` \| `pgvector` |
| `AUGUR_API_KEY` | *(unset)* | If set, requests must include `x-api-key: <key>` |
| `AUGUR_LOCAL_MODEL` | `Xenova/all-MiniLM-L6-v2` | Override the default `LocalEmbedder` model |
| `AUGUR_AUTO_LANGUAGE_FILTER` | `0` | Set `1` to filter results to the query's detected language (with soft fallback) |
| `PINECONE_INDEX_HOST` / `PINECONE_API_KEY` / `PINECONE_NAMESPACE` | — | When `AUGUR_ADAPTER=pinecone` |
| `TURBOPUFFER_API_KEY` / `TURBOPUFFER_NAMESPACE` | — | When `AUGUR_ADAPTER=turbopuffer` |
| `DATABASE_URL` / `PGVECTOR_TABLE` / `PGVECTOR_DIMENSION` | — | When `AUGUR_ADAPTER=pgvector` (also `npm i pg` in your project) |

## License

MIT — see [LICENSE](./LICENSE).
