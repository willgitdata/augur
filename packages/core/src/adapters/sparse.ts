/**
 * Sparse vector representation + encoder interface, used by adapters that
 * support sparse-dense hybrid search (today: PineconeAdapter).
 *
 * Why this lives in core (not in pinecone.ts): the shape — `{indices,
 * values}` — is the de-facto wire format across Pinecone, Qdrant
 * (`sparse_vector`), and Vespa (`sparseVector`). Any future adapter that
 * adds hybrid support will reuse it. Splitting it into its own file
 * keeps the Pinecone adapter from being a barrel.
 *
 * `BM25SparseEncoder` is shipped as a reference implementation so users
 * with no sparse-encoder library installed can still flip Pinecone into
 * hybrid mode. Users with Pinecone's own `pinecone-text` SDK, with a
 * trained SPLADE model, or with a Cohere-tokenizer sparse encoder
 * implement the one-method `SparseEncoder` interface against their
 * provider and pass it in — same pattern as `Embedder` and `Reranker`.
 */

import { tokenize } from "../embeddings/embedder.js";
import { tokenizeAdvanced } from "../embeddings/text-utils.js";

export interface SparseVector {
  /** Non-negative integer indices into the vocabulary / hash space. */
  indices: number[];
  /** Per-index weights, same length as `indices`. */
  values: number[];
}

export interface SparseEncoder {
  /** Stable identifier surfaced in adapter / trace metadata. */
  readonly name: string;
  /**
   * Optional one-time corpus fit. Called by adapters before the first
   * `encode()` if the encoder hasn't been pre-fit. Encoders that compute
   * IDF / vocabulary statistics use this; encoders that don't (e.g. a
   * pre-trained SPLADE model) can omit it.
   */
  fit?(corpus: string[]): void;
  /**
   * Optional fitness check. Adapters call this to decide whether to
   * auto-fit at upsert time. Encoders that don't need fitting (e.g.
   * pre-trained SPLADE) can omit it (treated as always fit). Encoders
   * that do need fitting and report `false` here will be auto-fit by
   * the adapter on the first upsert.
   */
  isFitted?(): boolean;
  /** Encode one piece of text to a sparse vector. Must be deterministic. */
  encode(text: string): SparseVector;
}

/**
 * BM25-weighted sparse vector encoder. Vocabulary is built lazily in
 * `fit()`; each unique stemmed term gets a stable integer index.
 *
 * The encoded values are the per-term BM25 contributions to the score,
 * so dot product of two such vectors approximates the BM25 score
 * function. Compatible with any vector DB whose sparse interface
 * accepts `{indices, values}` — Pinecone, Qdrant, Vespa, OpenSearch
 * (with knn-sparse plugin), and most newer hybrid backends.
 *
 * Out-of-vocabulary terms in `encode()` are silently dropped — they
 * carry no IDF anyway, so omitting them is correct. Re-fitting on a
 * new corpus resets the vocabulary and IDF table.
 */
export interface BM25SparseEncoderOptions {
  /** BM25 k1 (term-frequency saturation). Default 1.5. */
  k1?: number;
  /** BM25 b (length-normalization strength). Default 0.75. */
  b?: number;
  /**
   * Apply Porter stemming + English stopword filtering before encoding.
   * Default true — sparse encoders are almost always paired with a BM25
   * keyword side, and the recall lift from stemming is the same here.
   */
  stem?: boolean;
}

export class BM25SparseEncoder implements SparseEncoder {
  readonly name = "bm25-sparse";
  private k1: number;
  private b: number;
  private tokenize: (s: string) => string[];
  private vocab = new Map<string, number>();
  private idf = new Map<string, number>();
  private avgDocLen = 0;
  private fitted = false;

  constructor(opts: BM25SparseEncoderOptions = {}) {
    this.k1 = opts.k1 ?? 1.5;
    this.b = opts.b ?? 0.75;
    this.tokenize = (opts.stem ?? true)
      ? (s) => tokenizeAdvanced(s, { stem: true, dropStopwords: true })
      : tokenize;
  }

  fit(corpus: string[]): void {
    this.vocab.clear();
    this.idf.clear();
    const N = corpus.length;
    if (N === 0) {
      this.avgDocLen = 0;
      this.fitted = true;
      return;
    }

    const docFreq = new Map<string, number>();
    let totalLen = 0;
    for (const doc of corpus) {
      const tokens = this.tokenize(doc);
      totalLen += tokens.length;
      const seen = new Set<string>();
      for (const tok of tokens) {
        if (!this.vocab.has(tok)) this.vocab.set(tok, this.vocab.size);
        if (!seen.has(tok)) {
          docFreq.set(tok, (docFreq.get(tok) ?? 0) + 1);
          seen.add(tok);
        }
      }
    }
    this.avgDocLen = totalLen / N;
    for (const [tok, df] of docFreq) {
      // BM25 IDF with +1 smoothing — same variant InMemoryAdapter uses.
      this.idf.set(tok, Math.log(1 + (N - df + 0.5) / (df + 0.5)));
    }
    this.fitted = true;
  }

  /** Has `fit()` been called at least once? Used by adapters to gate auto-fit. */
  isFitted(): boolean {
    return this.fitted;
  }

  encode(text: string): SparseVector {
    const tokens = this.tokenize(text);
    if (tokens.length === 0) return { indices: [], values: [] };

    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    const len = tokens.length;
    const norm = 1 - this.b + this.b * (len / (this.avgDocLen || 1));

    const indices: number[] = [];
    const values: number[] = [];
    for (const [tok, f] of tf) {
      const idx = this.vocab.get(tok);
      if (idx === undefined) continue; // OOV — silently dropped
      const termIdf = this.idf.get(tok) ?? 0;
      const val = termIdf * ((f * (this.k1 + 1)) / (f + this.k1 * norm));
      if (val > 0) {
        indices.push(idx);
        values.push(val);
      }
    }
    return { indices, values };
  }
}
