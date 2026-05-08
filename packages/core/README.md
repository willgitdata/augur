# @augur/core

The Augur SDK — adaptive retrieval orchestration for RAG.

```bash
npm install @augur/core
```

```ts
import { Augur, LocalEmbedder } from "@augur/core";

const augr = new Augur({ embedder: new LocalEmbedder() });

await augr.index([
  { id: "1", content: "Postgres supports vector search via pgvector." },
  { id: "2", content: "Pinecone is a managed vector database." },
]);

const { results, trace } = await augr.search({
  query: "How do I store vectors in Postgres?",
  topK: 5,
});

console.log(trace.decision.strategy);  // "vector"
console.log(trace.decision.reasons);   // ["natural-language question → semantic search", ...]
```

`LocalEmbedder` is a fully on-device sentence-transformer (ONNX, ~22MB, no API keys). Swap to OpenAI or any hosted provider by implementing the three-method `Embedder` interface — see the root [README](../../README.md) and [ARCHITECTURE.md](../../ARCHITECTURE.md) for the full story.
