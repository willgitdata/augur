# Examples

Each example lives in `examples/<name>/` and runs with `pnpm --filter example-<name> start`.

## Hello world

```ts
import { Augur, LocalEmbedder } from "@augur-rag/core";

const augr = new Augur({ embedder: new LocalEmbedder() });

await augr.index([
  { id: "pg",    content: "PostgreSQL supports vector search via pgvector." },
  { id: "redis", content: "Redis is an in-memory key-value store." },
  { id: "k8s",   content: "Kubernetes manages containers across hosts." },
]);

const { results, trace } = await augr.search({
  query: "best vector database for production",
});

console.log(trace.decision.strategy); // → "hybrid"
console.log(results[0].chunk.documentId);
```

Run: `pnpm --filter example-basic-search start`.

## Same docs, four query shapes

```ts
const queries = [
  "How do I configure connection pooling in Postgres?",  // → vector
  "ERR_CONNECTION_REFUSED 4101",                          // → keyword
  '"liveness probes"',                                    // → keyword (quoted)
  "kubernetes",                                           // → keyword (very short)
];
```

The user doesn't pick the strategy. The router does, and records the reasoning in the trace.

## Custom adapter

Implement `VectorAdapter` and you get the routing engine and RRF hybrid for free.

```ts
import { BaseAdapter } from "@augur-rag/core";

class JsonFileAdapter extends BaseAdapter {
  readonly name = "json-file";
  readonly capabilities = {
    vector: true, keyword: true, hybrid: true,
    computesEmbeddings: false, filtering: false,
  };
  // upsert / searchVector / searchKeyword / delete / count / clear
}
```

Full ~90-line example: `examples/custom-adapter/index.ts`.

## Hosted embedder + reranker

`@augur-rag/core` ships one offline embedder (`LocalEmbedder`) and four offline rerankers. Hosted providers are a 30-line adapter against the `Embedder` / `Reranker` interfaces.

```ts
import { Augur, PineconeAdapter, type Embedder, type Reranker } from "@augur-rag/core";
import OpenAI from "openai";
import { CohereClient } from "cohere-ai";

class OpenAIEmbedder implements Embedder {
  readonly name = "openai:text-embedding-3-small";
  readonly dimension = 1536;
  private client = new OpenAI();
  async embed(texts: string[]) {
    const r = await this.client.embeddings.create({
      model: "text-embedding-3-small",
      input: texts,
    });
    return r.data.map((d) => d.embedding);
  }
}

class CohereReranker implements Reranker {
  readonly name = "cohere:rerank-english-v3.0";
  private client = new CohereClient({ token: process.env.COHERE_API_KEY! });
  async rerank(query, results, topK) {
    const r = await this.client.rerank({
      model: "rerank-english-v3.0",
      query,
      documents: results.map((x) => x.chunk.content),
      topN: topK,
    });
    return r.results.map((x) => ({ ...results[x.index]!, score: x.relevanceScore }));
  }
}

const augr = new Augur({
  adapter: new PineconeAdapter({
    indexHost: process.env.PINECONE_INDEX_HOST!,
    apiKey: process.env.PINECONE_API_KEY!,
  }),
  embedder: new OpenAIEmbedder(),
  reranker: new CohereReranker(),
});
```

## On-device stack (no API keys)

`LocalEmbedder` and `LocalReranker` run real models on-device via `@huggingface/transformers`. First run downloads the ONNX model to `~/.cache/huggingface/hub`; subsequent runs are instant.

```ts
import { Augur, LocalEmbedder, LocalReranker } from "@augur-rag/core";

const augr = new Augur({
  embedder: new LocalEmbedder(),   // Xenova/all-MiniLM-L6-v2, 22 MB
  reranker: new LocalReranker(),   // Xenova/ms-marco-MiniLM-L-6-v2, 22 MB
});
```

Swap to a bigger model with the matching prefix:

```ts
new LocalEmbedder({
  model: "Xenova/bge-small-en-v1.5",
  queryPrefix: "Represent this sentence for searching relevant passages: ",
});
```

You need the optional peer dep: `pnpm add @huggingface/transformers`.

## Postgres + pgvector

The schema dimension MUST match your embedder. `LocalEmbedder` defaults to 384.

```sql
CREATE EXTENSION vector;
CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  content TEXT NOT NULL,
  index INT NOT NULL,
  metadata JSONB,
  embedding VECTOR(384),
  content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (content_tsv);
```

```ts
import pg from "pg";
import { Augur, PgVectorAdapter, LocalEmbedder } from "@augur-rag/core";

const client = new pg.Client({ connectionString: process.env.DATABASE_URL });
await client.connect();

const augr = new Augur({
  adapter: new PgVectorAdapter({
    client: { query: (sql, params) => client.query(sql, params).then((r) => ({ rows: r.rows })) },
    table: "chunks",
    dimension: 384,
  }),
  embedder: new LocalEmbedder(),
});
```

## Contextual retrieval (Anthropic's pattern)

For each chunk, ask a fast LLM for a one-line description that situates it. Prepend that to the chunk before embedding. Anthropic measured chunk failure rate dropping from 5.7% to 1.9%.

```ts
import {
  Augur,
  ContextualChunker,
  SentenceChunker,
  ANTHROPIC_CONTEXTUAL_PROMPT,
  sanitizeForContextualPrompt,
} from "@augur-rag/core";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const chunker = new ContextualChunker({
  base: new SentenceChunker(),
  provider: {
    name: "anthropic:claude-haiku-4-5",
    async contextualize({ chunk, document }) {
      const r = await client.messages.create({
        model: "claude-haiku-4-5",
        max_tokens: 100,
        messages: [{
          role: "user",
          content: ANTHROPIC_CONTEXTUAL_PROMPT
            .replace("{WHOLE_DOCUMENT}", sanitizeForContextualPrompt(document))
            .replace("{CHUNK_CONTENT}", sanitizeForContextualPrompt(chunk)),
        }],
      });
      const block = r.content[0];
      return block && block.type === "text" ? block.text : "";
    },
  },
});
```

With prompt caching the per-chunk cost is roughly the chunk size in tokens. Re-indexing unchanged content is a cache hit. Always sanitize document content before substituting into the prompt template; raw `</document>` in user content would otherwise let the doc author hijack the LLM call.

## Inspecting traces

```ts
const { trace } = await augr.search({ query: "how to deploy with zero downtime" });

console.log(trace.decision);
// {
//   strategy: "vector",
//   reasons: ["natural-language question → semantic search", "reranking enabled"],
//   reranked: true,
//   signals: { wordCount: 6, hasQuotedPhrase: false, language: "en", ... }
// }

console.log(trace.spans.map(s => `${s.name}: ${s.durationMs.toFixed(1)}ms`));
// ["embed:query: 1.4ms", "pool:vector: 0.8ms", "rerank: 0.3ms"]
```

If a user reports a bad result, the trace ID is enough to look up exactly what happened.

## HTTP

```bash
curl -X POST localhost:3001/index \
  -H 'Content-Type: application/json' \
  -d '{"documents":[{"id":"1","content":"hello world"}]}'

curl -X POST localhost:3001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello","topK":3}'
```

The trace ships in the response body. OpenAPI spec at `/openapi.json`, Swagger UI at `/docs`.
