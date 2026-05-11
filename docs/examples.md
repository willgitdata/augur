# Examples

Hands-on walkthroughs. Each example lives in `examples/<name>/` and is runnable with `pnpm --filter example-<name> start`.

---

## 1. Hello, Augur

The 30-line "hello world".

```ts
import { Augur, LocalEmbedder } from "@augur-rag/core";

const augr = new Augur({ embedder: new LocalEmbedder() });

await augr.index([
  { id: "pg",    content: "PostgreSQL supports vector search via pgvector." },
  { id: "redis", content: "Redis is an in-memory key-value store." },
  { id: "k8s",   content: "Kubernetes manages containers across hosts." },
]);

const { results, trace } = await augr.search({ query: "best vector database for production" });

console.log(trace.decision.strategy); // → "hybrid"
console.log(results[0].chunk.documentId);
```

Run: `pnpm --filter example-basic-search start`. The full example exercises four query archetypes (natural-language question, error code, quoted phrase, single keyword) and prints the routing decision for each.

---

## 2. The router in action

Same documents, four queries, four different strategies:

```ts
const queries = [
  "How do I configure connection pooling in Postgres?",  // → vector
  "ERR_CONNECTION_REFUSED 4101",                         // → keyword
  '"liveness probes"',                                   // → keyword (quoted)
  "kubernetes",                                          // → keyword (very short)
];
```

What you'll see in the trace:

```
=== Query: How do I configure connection pooling in Postgres?
Strategy: vector (+rerank)  · 14.3 ms · 30 candidates
Reasons:
  - natural-language question → semantic search
  - reranking enabled (latency budget allows)
```

```
=== Query: ERR_CONNECTION_REFUSED 4101
Strategy: keyword  · 1.8 ms · 1 candidates
Reasons:
  - short query with specific identifiers/codes
  - reranking skipped (latency budget)
```

The user never asked about strategies. The router did.

---

## 3. Custom adapter

Implement `VectorAdapter`, get the whole orchestrator for free.

```ts
import { BaseAdapter } from "@augur-rag/core";

class JsonFileAdapter extends BaseAdapter {
  readonly name = "json-file";
  readonly capabilities = {
    vector: true, keyword: true, hybrid: true,
    computesEmbeddings: false, filtering: false,
  };
  // ... upsert / searchVector / searchKeyword / delete / count / clear ...
}

const augr = new Augur({
  adapter: new JsonFileAdapter("./store.json"),
  embedder: new LocalEmbedder(),
});
```

The full implementation is `examples/custom-adapter/index.ts`. About 90 lines, including the BM25-ish keyword search.

Hybrid retrieval (RRF) and the routing engine come for free as soon as you implement vector + keyword. You don't write the fusion code yourself.

---

## 4. Comparing chunkers

```ts
import { FixedSizeChunker, SentenceChunker, SemanticChunker, LocalEmbedder } from "@augur-rag/core";

const fixed    = new FixedSizeChunker({ size: 200, overlap: 30 });
const sentence = new SentenceChunker({ targetSize: 200 });
const semantic = new SemanticChunker({ embedder: new LocalEmbedder(), threshold: 0.5 });
```

Run on the same document, you'll see:

- `fixed-size`: predictable count, mid-sentence cuts
- `sentence`: grammatical units, slight count variance
- `semantic`: topic-aligned cuts, fewer chunks for cohesive text

Pick based on content type, not on intuition. When in doubt, `SentenceChunker` is the default for a reason.

---

## 5. Switching to a hosted embedder and reranker

`@augur-rag/core` ships one built-in embedder (`LocalEmbedder`, on-device ONNX) and three rerankers (`HeuristicReranker`, `LocalReranker`, `MMRReranker`). Hosted providers are intentionally not in core. The `Embedder` interface is three methods, `Reranker` is one. Implement against your provider's official SDK and pass it in:

```ts
import { Augur, PineconeAdapter, type Embedder, type Reranker } from "@augur-rag/core";
import OpenAI from "openai";
import { CohereClient } from "cohere-ai";

class OpenAIEmbedder implements Embedder {
  readonly name = "openai:text-embedding-3-small";
  readonly dimension = 1536;
  private client = new OpenAI();
  async embed(texts: string[]): Promise<number[][]> {
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
  async rerank(query: string, results, topK) {
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
    namespace: "prod",
  }),
  embedder: new OpenAIEmbedder(),
  reranker: new CohereReranker(),
});
```

The router adapts to Pinecone's keyword-incapable status automatically (stops picking keyword, lets the reranker carry precision).

### Picking the built-in offline embedder (no API key needed)

`LocalEmbedder` is the only embedder shipped in core. It runs a real sentence-transformer model entirely on-device via `@huggingface/transformers` (ONNX Runtime). Default model is `Xenova/all-MiniLM-L6-v2` (~22MB, 384d). First run downloads the model to `~/.cache/huggingface/hub`; subsequent runs are instant.

```ts
import {
  Augur,
  LocalEmbedder,
  LocalReranker,
  MetadataChunker,
  SentenceChunker,
} from "@augur-rag/core";

const augr = new Augur({
  embedder: new LocalEmbedder(),                                  // 22MB, 384d
  reranker: new LocalReranker(),                                  // 22MB cross-encoder
  chunker: new MetadataChunker({ base: new SentenceChunker() }),
});
```

You'll need to install the optional peer dep:

```bash
pnpm add @huggingface/transformers
```

For higher accuracy at a slightly larger size, swap the model and supply the model's required prefixes:

```ts
// BGE-small: top of MTEB at this size; query prefix required.
new LocalEmbedder({
  model: "Xenova/bge-small-en-v1.5",
  queryPrefix: "Represent this sentence for searching relevant passages: ",
});

// E5-small: balanced; both prefixes required.
new LocalEmbedder({
  model: "Xenova/e5-small-v2",
  queryPrefix: "query: ",
  docPrefix: "passage: ",
});

// nomic-embed-text-v1.5: 768d, 137MB; instruction-tuned.
new LocalEmbedder({
  model: "nomic-ai/nomic-embed-text-v1.5",
  dimension: 768,
  queryPrefix: "search_query: ",
  docPrefix: "search_document: ",
});
```

`MetadataChunker` wraps any base chunker and prepends `[doc-id | topic | title]` to each chunk before embedding.

Stacking order matters. Each layer attacks a different failure mode: the bi-encoder gives broad recall, the cross-encoder rescues near-misses, the metadata chunker fixes "the chunk doesn't mention the doc topic," stemmed BM25 catches plural and inflectional misses on lexical queries. Numbers measured on a 504-query development eval (preserved out-of-tree at commit `feffc73^`) confirmed each layer adds incremental NDCG@10; that harness will be republished as `augur-eval` so you can re-run on your own corpus.

### Contextual Retrieval (Anthropic's pattern)

For each chunk, send the chunk and its source document to a fast LLM (Haiku, GPT-4o-mini, Gemini Flash) and ask for a one-line description that situates the chunk. Prepend that description to the chunk content before embedding. Anthropic measured chunk failure rate dropping from 5.7% to 1.9% (a 67% reduction) with this technique on their internal RAG eval, and it stacks with hybrid retrieval and reranking.

The contract is one method (`contextualize`); wire any LLM provider in ~10 lines:

```ts
import { Augur, ContextualChunker, SentenceChunker, ANTHROPIC_CONTEXTUAL_PROMPT } from "@augur-rag/core";
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
            .replace("{WHOLE_DOCUMENT}", document)
            .replace("{CHUNK_CONTENT}", chunk),
        }],
      });
      const block = r.content[0];
      return block && block.type === "text" ? block.text : "";
    },
  },
  concurrency: 4,            // tune to your provider's rate limit
});

const augr = new Augur({
  embedder: new LocalEmbedder(),
  chunker,
});

await augr.index(documents);  // one LLM call per chunk, cached on hash(doc, chunk)
```

Cost: with Anthropic's prompt caching the document portion (the bulk of the input) is amortized across all chunks of the same document, so the per-chunk cost is roughly the chunk size in tokens, a few cents per thousand chunks at Haiku rates. Re-indexing unchanged content is free (the `MemoryContextCache` default returns the prior result; swap in a persistent cache for cross-process re-use).

The same `provider` interface works with OpenAI, Gemini, or any LLM; swap the SDK call inside `contextualize`. The `ContextualChunker` itself stays the same.

This composes with `MetadataChunker` and `Doc2QueryChunker`; wrap them in any order. Common stack: `Contextual( Metadata( Sentence ) )` so chunks get both document metadata (cheap, deterministic) and LLM-generated context (expensive, semantic).

### Doc2Query: synthetic-question expansion at index time

For each chunk, generate N questions the chunk could answer using a small T5 model (`Xenova/LaMini-T5-61M`, ~24MB), then append them to the chunk's content before embedding and BM25 indexing. Cost is paid once at index; zero query-time latency. Works particularly well on conversational queries against reference-style content (and on non-English chunks indexed alongside English questions).

```ts
import { Augur, Doc2QueryChunker, SentenceChunker, MetadataChunker } from "@augur-rag/core";

const augr = new Augur({
  chunker: new Doc2QueryChunker({
    base: new MetadataChunker({ base: new SentenceChunker() }),
    numQueries: 3,             // questions per chunk; more = better recall, longer index
    model: "Xenova/LaMini-T5-61M",
  }),
});
```

Requires `@huggingface/transformers` (same dep as LocalEmbedder).

### Query-aware hybrid weights

`Augur.search()` picks the BM25-vs-vector mix per query from the router's signals: quoted phrases, specific tokens, and very short queries lean BM25 (0.3-0.4 vector weight), long natural-language questions lean vector (0.7), default is 0.5. Production hybrid systems all do some version of this; a fixed 0.5/0.5 mix under-weights whichever side is wrong for the current query shape.

No configuration needed; it's automatic when strategy = "hybrid".

### Stemmed BM25 (`InMemoryAdapter({ useStemming: true })`)

Turns on Porter stemming + English stopword filtering for the keyword path. Same pipeline Lucene and Elasticsearch use by default. The reliable win on quoted, named-entity, and short keyword queries is the recall lift from collapsing inflectional forms (running ↔ runs, connection ↔ connections all map to one stem); for any non-trivial BM25 workload it's the cheapest improvement you'll find.

```ts
import { Augur, InMemoryAdapter, LocalEmbedder, LocalReranker } from "@augur-rag/core";

const augr = new Augur({
  adapter: new InMemoryAdapter({ useStemming: true }),
  embedder: new LocalEmbedder(),
  reranker: new LocalReranker(),
});
```

### MMR (Maximal Marginal Relevance) for diverse top-K

For ambiguous queries with multiple relevant docs, pure-relevance reranking concentrates on near-duplicates. `MMRReranker` rebalances toward novelty:

```ts
import { CascadedReranker, LocalReranker, MMRReranker, Augur } from "@augur-rag/core";

// Cross-encoder narrows by relevance; MMR diversifies the survivors.
const reranker = new CascadedReranker([
  [new LocalReranker(), 50],
  [new MMRReranker({ lambda: 0.7 }), 10],
]);

const augr = new Augur({ reranker, /* ... */ });
```

`λ = 1.0` is pure relevance; `λ = 0.7` is the standard "relevance with diversity boost". Not enabled by default; on QA-style queries (one relevant doc) MMR pushes hits out in favor of variety, which hurts. Reach for it on multi-aspect queries, search-results pages, and RAG pipelines where the LLM benefits from non-redundant context.

### Picking a reranker

`@augur-rag/core` ships four offline rerankers:

```ts
import {
  HeuristicReranker,    // zero-dep, sub-ms; weak baseline (token overlap + proximity)
  LocalReranker,        // local ONNX cross-encoder (~22MB, ms-marco-MiniLM-L-6-v2)
  MMRReranker,          // diversity-aware reranking (no model)
  CascadedReranker,     // chain rerankers: cheap-broad to expensive-narrow
} from "@augur-rag/core";

// Cascaded rerank: heuristic narrows 100 → 50, cross-encoder narrows 50 → 10:
const cascade = new CascadedReranker([
  [new HeuristicReranker(), 50],
  [new LocalReranker(), 10],
]);
```

For Cohere, Jina, Voyage, or any hosted cross-encoder, implement the one-method `Reranker` interface directly against the provider's SDK; see §5 above for a Cohere snippet. `HeuristicReranker` is fine for a smoke test; cross-encoder rerankers are typically the single biggest accuracy lever once embeddings are decent.

---

## 6. Postgres with pgvector

The recommended adapter for most teams: you probably already have Postgres.

The schema dimension MUST match your embedder. The example below uses `LocalEmbedder` (384d). For a hosted embedder, swap both numbers in lockstep: `text-embedding-3-small` is 1536d, `text-embedding-3-large` is 3072d, Cohere `embed-english-v3.0` is 1024d.

```sql
CREATE EXTENSION vector;
CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  content TEXT NOT NULL,
  index INT NOT NULL,
  metadata JSONB,
  embedding VECTOR(384),  -- match your embedder's dimension
  content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (content_tsv);
CREATE INDEX ON chunks (document_id);
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
    dimension: 384,                  // matches LocalEmbedder + the VECTOR(384) above
  }),
  embedder: new LocalEmbedder(),     // or your hosted embedder per §5
});
```

The adapter validates `chunk.embedding.length === dimension` at upsert time and throws on mismatch, so you'll catch this on the first `index()` call rather than silently corrupting the table.

Vector + keyword + hybrid all in one place, no extra services.

> Filter keys must be plain identifiers. `PgVectorAdapter` rejects filter keys that don't match `^[a-zA-Z_][a-zA-Z0-9_]*$` (the same rule it applies to the table name). Postgres can't parameter-bind inside `metadata->>'key'`, so we whitelist instead of escaping.

---

## 7. Trace introspection

Every search produces a trace; every trace is enough to explain (or debug) the result.

```ts
const { trace } = await augr.search({ query: "how to deploy with zero downtime" });

console.log(trace.decision);
// {
//   strategy: "vector",
//   reasons: [
//     "natural-language question → semantic search",
//     "reranking enabled (latency budget allows)"
//   ],
//   reranked: true,
//   signals: { wordCount: 6, avgWordLen: 4.5, hasQuotedPhrase: false, language: "en", ... }
// }

console.log(trace.spans.map(s => `${s.name}: ${s.durationMs.toFixed(1)}ms`));
// ["embed:query: 1.4ms", "pool:vector: 0.8ms", "pool:keyword: 0.4ms", "rerank: 0.3ms"]
```

Save the trace ID; if a user reports a bad result, you can look up exactly what happened.

---

## 8. The HTTP API

```bash
curl -X POST localhost:3001/index \
  -H 'Content-Type: application/json' \
  -d '{"documents":[{"id":"1","content":"hello world"}]}'

curl -X POST localhost:3001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello","topK":3}'
```

The trace is in the response: log it, ship it to your observability backend, or render it in your own UI.

---

## 9. Wiring into LangChain (sketch)

```ts
import { BaseRetriever } from "@langchain/core/retrievers";
import { Augur } from "@augur-rag/core";

class AugurRetriever extends BaseRetriever {
  augr = new Augur({ embedder: new LocalEmbedder() /* + adapter, reranker, ... */ });
  lc_namespace = ["custom", "augur"];
  async _getRelevantDocuments(query: string) {
    const { results } = await this.augr.search({ query });
    return results.map((r) => ({
      pageContent: r.chunk.content,
      metadata: { ...r.chunk.metadata, score: r.score, chunkId: r.chunk.id },
    }));
  }
}
```

About 20 lines. The same pattern works for LlamaIndex; we'll publish official bindings as `@augur-rag/langchain` and `@augur-rag/llamaindex` once the SDK API stabilizes.

---

## 10. A/B testing two routers

```ts
const embedder = new LocalEmbedder();
const a = new Augur({ embedder, router: new HeuristicRouter() });
const b = new Augur({ embedder, router: new MyMLRouter() });

const which = userId.charCodeAt(0) % 2 === 0 ? a : b;
const result = await which.search({ query, context: { ab: which === a ? "A" : "B" } });
```

The `context.ab` propagates into the trace, so every search is analyzable by bucket.
