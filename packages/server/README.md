# @augur/server

The HTTP API for Augur: adaptive retrieval orchestration for RAG and semantic search. A drop-in Fastify wrapper around [@augur/core](https://www.npmjs.com/package/@augur/core) with OpenAPI docs at `/docs` and search traces in every response.

## Install + run

```bash
npm install -g @augur/server
# also install the peer dep, since augur-server uses LocalEmbedder by default:
npm install -g @huggingface/transformers
augur-server
```

`@huggingface/transformers` is an *optional* peer of `@augur/core`. The `augur-server` CLI uses `LocalEmbedder` out of the box (on-device ONNX, no API keys), which needs the peer dep. If you wire your own `buildServer({ embedder })` against a hosted provider (OpenAI / Cohere / Voyage / etc.), you can skip it.

## Or as a library in your own service

```ts
import { buildServer } from "@augur/server";
import { LocalEmbedder } from "@augur/core";

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
| `HOST` | `0.0.0.0` | Bind address |
| `AUGUR_ADAPTER` | `in-memory` | `in-memory` \| `pinecone` \| `turbopuffer` \| `pgvector` |
| `AUGUR_API_KEY` | *(unset)* | If set, requests must include `x-api-key: <key>` |
| `AUGUR_LOCAL_MODEL` | `Xenova/all-MiniLM-L6-v2` | Override the default `LocalEmbedder` model |
| `AUGUR_AUTO_LANGUAGE_FILTER` | `0` | Set `1` to filter results to the query's detected language (with soft fallback) |
| `PINECONE_INDEX_HOST` / `PINECONE_API_KEY` / `PINECONE_NAMESPACE` | — | When `AUGUR_ADAPTER=pinecone` |
| `TURBOPUFFER_API_KEY` / `TURBOPUFFER_NAMESPACE` | — | When `AUGUR_ADAPTER=turbopuffer` |
| `DATABASE_URL` / `PGVECTOR_TABLE` / `PGVECTOR_DIMENSION` | — | When `AUGUR_ADAPTER=pgvector` (also `npm i pg` in your project) |

## Learn more

- [Project README](https://github.com/willgitdata/augur#readme): the headline pitch, BEIR comparison table, and quick start
- [ARCHITECTURE.md](https://github.com/willgitdata/augur/blob/main/ARCHITECTURE.md): how the orchestrator, router, adapters, chunkers, and rerankers fit together
- [API_REFERENCE.md](https://github.com/willgitdata/augur/blob/main/API_REFERENCE.md): SDK and HTTP API reference

## License

MIT. See [LICENSE](./LICENSE).
