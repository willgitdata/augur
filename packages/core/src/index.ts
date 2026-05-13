/**
 * @augur-rag/core — public API surface.
 *
 * This file is the single entry point users import from. Sub-paths
 * (`@augur-rag/core/adapters`, `/chunking`, `/routing`) exist for
 * tree-shaking / explicitness but everything is also re-exported here
 * for the simple `import { Augur } from "@augur-rag/core"` case.
 */

export { Augur, type AugurOptions } from "./augur.js";

// Types
export type {
  Document,
  Chunk,
  SearchRequest,
  SearchResponse,
  SearchResult,
  SearchTrace,
  TraceSpan,
  RoutingDecision,
  RetrievalStrategy,
  QuerySignals,
  IndexResponse,
} from "./types.js";

// Adapters
export {
  type VectorAdapter,
  type AdapterCapabilities,
  type VectorSearchOpts,
  type KeywordSearchOpts,
  type HybridSearchOpts,
  BaseAdapter,
  InMemoryAdapter,
  type InMemoryAdapterOptions,
  PineconeAdapter,
  type PineconeAdapterOptions,
  TurbopufferAdapter,
  PgVectorAdapter,
  type PgClient,
  type SparseVector,
  type SparseEncoder,
  BM25SparseEncoder,
  type BM25SparseEncoderOptions,
} from "./adapters/index.js";

// Chunking
export {
  type Chunker,
  type AsyncChunker,
  FixedSizeChunker,
  SentenceChunker,
  SemanticChunker,
  MetadataChunker,
  Doc2QueryChunker,
  type Doc2QueryChunkerOptions,
  ContextualChunker,
  type ContextualChunkerOptions,
  type ContextProvider,
  type ContextCache,
  MemoryContextCache,
  ANTHROPIC_CONTEXTUAL_PROMPT,
  sanitizeForContextualPrompt,
  chunkDocument,
} from "./chunking/index.js";

// Embeddings
export {
  type Embedder,
  tokenize,
  tokenizeAdvanced,
  stem,
  STOPWORDS,
} from "./embeddings/embedder.js";
export { LocalEmbedder } from "./embeddings/local-embedder.js";

// Routing
export {
  type Router,
  HeuristicRouter,
  type HeuristicRouterOptions,
  computeSignals,
} from "./routing/index.js";

// Reranking
export {
  type Reranker,
  HeuristicReranker,
  CascadedReranker,
} from "./reranking/reranker.js";
export { LocalReranker } from "./reranking/local-reranker.js";
export { MMRReranker } from "./reranking/mmr-reranker.js";

// Question taxonomy from signals (re-exported for routers consuming the type).
export type { QuestionType } from "./types.js";

// Observability
export { Tracer, TraceStore } from "./observability/tracer.js";
