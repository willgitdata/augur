/**
 * Core types for Augur.
 *
 * Design notes:
 * - Documents are the unit of indexing; chunks are derived.
 * - Adapters operate on chunks (post-chunking), not raw documents.
 * - Every search returns both results and a Trace — the trace is what makes
 *   Augur debuggable. Every routing decision is recorded.
 */

/** A document submitted by the user for indexing. */
export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

/** A chunk is a sub-section of a document, produced by a Chunker. */
export interface Chunk {
  id: string;
  documentId: string;
  content: string;
  /** Position of this chunk within the document, used for ordering & overlap. */
  index: number;
  /** Optional pre-computed embedding. Adapters may compute lazily. */
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

/** A single result returned from search. */
export interface SearchResult {
  chunk: Chunk;
  /** Higher = more relevant. Backends should normalize to 0..1 where possible. */
  score: number;
  /** Per-strategy raw scores, useful for debugging hybrid retrieval. */
  rawScores?: Record<string, number>;
}

/** The four retrieval strategies Augur can route between. */
export type RetrievalStrategy = "vector" | "keyword" | "hybrid" | "rerank";

/** User-facing search request. */
export interface SearchRequest {
  query: string;
  /** Optional pre-loaded documents for ad-hoc/in-memory mode. */
  documents?: Document[];
  /** Top-K results to return. Defaults to 10. */
  topK?: number;
  /** Override the router and force a strategy. */
  forceStrategy?: RetrievalStrategy;
  /** Soft latency budget in ms. The router uses this to decide whether reranking is worth it. */
  latencyBudgetMs?: number;
  /** Optional metadata filter passed through to the adapter. */
  filter?: Record<string, unknown>;
  /** Free-form context the router can use (e.g. user id, session id, ab-test bucket). */
  context?: Record<string, unknown>;
}

/** A routing decision recorded for observability. */
export interface RoutingDecision {
  strategy: RetrievalStrategy;
  reasons: string[];
  /** Signals computed about the query (length, specificity, ambiguity, …). */
  signals: QuerySignals;
  /** Whether reranking was applied on top of the base strategy. */
  reranked: boolean;
}

/** Signals derived from the query. */
export interface QuerySignals {
  /** Token count (whitespace-split, lowercased). */
  tokens: number;
  /** Average token length — proxy for technical/long-form queries. */
  avgTokenLen: number;
  /** Has quoted phrases — strong signal for keyword search. */
  hasQuotedPhrase: boolean;
  /** Has rare/specific tokens (numbers, identifiers, code-like patterns). */
  hasSpecificTokens: boolean;
  /** Has natural-language question structure. */
  isQuestion: boolean;
  /** Heuristic ambiguity score 0..1. Higher = more ambiguous. */
  ambiguity: number;
}

/** A timing span recorded during search. */
export interface TraceSpan {
  name: string;
  startMs: number;
  endMs: number;
  durationMs: number;
  attributes?: Record<string, unknown>;
}

/** Full execution trace — returned alongside every search. */
export interface SearchTrace {
  id: string;
  query: string;
  startedAt: string; // ISO
  totalMs: number;
  decision: RoutingDecision;
  spans: TraceSpan[];
  /** Number of candidates considered before reranking & top-K cutoff. */
  candidates: number;
  /** Adapter implementation name, e.g. "in-memory" or "pinecone". */
  adapter: string;
  /** Embedding model name, when applicable. */
  embeddingModel?: string;
}

/** Result of a search call. */
export interface SearchResponse {
  results: SearchResult[];
  trace: SearchTrace;
}

/** Result of an indexing call. */
export interface IndexResponse {
  documents: number;
  chunks: number;
  trace: {
    chunkingMs: number;
    embeddingMs: number;
    upsertMs: number;
    totalMs: number;
  };
}
