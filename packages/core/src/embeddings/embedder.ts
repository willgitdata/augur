/**
 * Embedding provider interface.
 *
 * Tradeoff: this interface is small on purpose. Real-world embedders all share
 * one operation: text-in, vector-out. Anything more complex (caching, batching,
 * rate limiting) is a wrapper concern.
 *
 * Embedders that benefit from corpus statistics (TF-IDF, learned-IDF, etc) can
 * implement the optional `fit(docs)` method. The orchestrator calls fit() during
 * `index()` so the embedder can update its internal state before embedding.
 *
 * Embedders that distinguish documents-to-be-indexed vs search-queries (Gemini's
 * task types, Cohere v3's input_type, BGE's instruct prefixes) implement the
 * optional `embedDocuments` / `embedQuery` methods. Augur prefers them over
 * the generic `embed()` when present.
 */
export interface Embedder {
  /** Stable identifier — surfaced in traces. */
  readonly name: string;
  /** Output vector dimension. Must be stable across calls. */
  readonly dimension: number;
  /** Embed a batch of texts. Returns one vector per input, in order. */
  embed(texts: string[]): Promise<number[][]>;
  /** Optional: ingest documents to update embedder state. No-op by default. */
  fit?(texts: string[]): void;
  /** Optional: embed text(s) explicitly tagged as documents-to-be-indexed. */
  embedDocuments?(texts: string[]): Promise<number[][]>;
  /** Optional: embed a single text explicitly tagged as a search query. */
  embedQuery?(text: string): Promise<number[]>;
}

/** Whitespace + punctuation tokenizer, lowercased. */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/u)
    .filter((t) => t.length > 0);
}

// Re-export advanced tokenizer + stemmer + stopwords from text-utils for
// users who want to plug them into custom embedders or rerankers.
export { tokenizeAdvanced, stem, STOPWORDS } from "./text-utils.js";
