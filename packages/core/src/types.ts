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
  /**
   * Confidence floor — drop any result whose final score is below this
   * value. Useful when "no answer" is a better signal to the LLM than a
   * noisy low-relevance answer. Score units depend on the active reranker:
   * cross-encoders typically emit calibrated [0, 1] sigmoid scores, so
   * `minScore: 0.4` is a reasonable starting point.
   */
  minScore?: number;
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

/** Refined question type. `null` means the query isn't a question. */
export type QuestionType = "factoid" | "procedural" | "definitional";

/** Signals derived from the query. */
export interface QuerySignals {
  /**
   * Whitespace-split word count. NOT the embedding model's subword token
   * count — these signals are used by the router's rule decisions
   * ("≥6 words AND isQuestion → vector"), not the embedder. The embedder
   * has its own WordPiece/BPE tokenization that's independent of this.
   */
  wordCount: number;
  /** Average word length in characters — proxy for technical/long-form queries. */
  avgWordLen: number;
  /** Has quoted phrases — strong signal for keyword search. */
  hasQuotedPhrase: boolean;
  /** Has rare/specific tokens (numbers, identifiers, code-like patterns). */
  hasSpecificTokens: boolean;
  /** Has natural-language question structure. */
  isQuestion: boolean;
  /** Heuristic ambiguity score 0..1. Higher = more ambiguous. */
  ambiguity: number;
  /**
   * Has a probable named entity — capitalized non-stopword token outside the
   * very first position. Catches "PgBouncer", "Redis Streams", "TLS handshake".
   */
  hasNamedEntity: boolean;
  /**
   * Has code-like syntax — camelCase identifiers, dotted paths, snake_case,
   * `::` scope, function-call parens. Stronger keyword signal than prose.
   */
  hasCodeLike: boolean;
  /**
   * Has a date, semver, RFC, or CVE token. Specific identifiers users want
   * matched literally.
   */
  hasDateOrVersion: boolean;
  /**
   * Refined question taxonomy. `factoid` → who/when/where/which (entity match
   * matters); `procedural` → how/why (semantic); `definitional` → "what is X"
   * (semantic + rerank). `null` if not a question.
   */
  questionType: QuestionType | null;
  /**
   * Has a negation token — "not", "without", "except", "vs", "never". Bi-encoders
   * famously confuse "X without Y" with "X with Y", so this forces rerank on.
   */
  hasNegation: boolean;
  /**
   * BCP-47-style language code derived from Unicode-script analysis.
   * `"en"` for Latin script (default), or `"ja" | "zh" | "ko" | "ru" |
   * "ar" | "hi" | "th" | "he" | "el"` for the corresponding non-Latin
   * scripts. Drives two things: the router's vector-bias rule for
   * non-English queries, and Augur's automatic language-aware filter
   * at search time (see `Augur.search`).
   */
  language: string;
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
