# @augur-rag/core

The Augur SDK: adaptive retrieval orchestration for RAG and semantic search. Per-query routing across vector, BM25, hybrid, and cross-encoder reranking. A trace is returned with every search response.

## Install

```bash
npm install @augur-rag/core
# also install the peer dep if you use LocalEmbedder or LocalReranker (on-device ONNX models):
npm install @huggingface/transformers
```

`@huggingface/transformers` is an optional peer dep. Only `LocalEmbedder` and `LocalReranker` need it. If you wire in OpenAI / Cohere / Voyage / Anthropic or any other provider via the `Embedder` / `Reranker` interface, you can skip it.

## Hello world

```ts
import { Augur, LocalEmbedder } from "@augur-rag/core";

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

`LocalEmbedder` is an on-device sentence-transformer (`Xenova/all-MiniLM-L6-v2` by default, ~22 MB ONNX, no API keys). Swap to a hosted provider by implementing the three-method `Embedder` interface.

## Learn more

- [Project README](https://github.com/willgitdata/augur#readme): pitch, BEIR comparison table, and quick start
- [ARCHITECTURE.md](https://github.com/willgitdata/augur/blob/main/ARCHITECTURE.md): how the orchestrator, router, adapters, chunkers, and rerankers fit together
- [EXAMPLES.md](https://github.com/willgitdata/augur/blob/main/EXAMPLES.md): hosted embedders, contextual retrieval, BGE / E5 model swaps, MMR diversity, pgvector setup
- [API_REFERENCE.md](https://github.com/willgitdata/augur/blob/main/API_REFERENCE.md): SDK and HTTP API reference

## License

MIT. See [LICENSE](./LICENSE).
