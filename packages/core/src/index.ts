/**
 * @augur/core — public API surface.
 *
 * This file is the single entry point users import from. Sub-paths
 * (`@augur/core/adapters`, `/chunking`, `/routing`) exist for
 * tree-shaking / explicitness but everything is also re-exported here
 * for the simple `import { Augur } from "@augur/core"` case.
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
  BaseAdapter,
  InMemoryAdapter,
  PineconeAdapter,
  TurbopufferAdapter,
  PgVectorAdapter,
  type PgClient,
} from "./adapters/index.js";

// Chunking
export {
  type Chunker,
  FixedSizeChunker,
  SentenceChunker,
  SemanticChunker,
  chunkDocument,
} from "./chunking/index.js";

// Embeddings
export {
  type Embedder,
  HashEmbedder,
  OpenAIEmbedder,
  tokenize,
} from "./embeddings/embedder.js";

// Routing
export { type Router, HeuristicRouter, computeSignals } from "./routing/index.js";

// Reranking
export {
  type Reranker,
  HeuristicReranker,
  CohereReranker,
  JinaReranker,
  HttpCrossEncoderReranker,
} from "./reranking/reranker.js";

// Question taxonomy from signals (re-exported for routers consuming the type).
export type { QuestionType } from "./types.js";

// Observability
export { Tracer, TraceStore } from "./observability/tracer.js";
