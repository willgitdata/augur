import { tokenize } from "../embeddings/embedder.js";
import { tokenizeAdvanced, STOPWORDS } from "../embeddings/text-utils.js";
import type { Chunk, SearchResult } from "../types.js";
import {
  BaseAdapter,
  type AdapterCapabilities,
  type KeywordSearchOpts,
  type VectorSearchOpts,
} from "./adapter.js";

/**
 * InMemoryAdapter — zero-dependency, fully-featured local adapter.
 *
 * Why we ship this in core:
 * - Lets `npm install augur` work end-to-end with no external services.
 * - The reference implementation for the adapter contract — easy to read.
 * - Useful in production for test fixtures and small datasets (< ~50k chunks).
 *
 * Implementation notes:
 * - Vector search is brute-force cosine. Fine up to ~50k chunks; for more,
 *   plug in a real adapter.
 * - Keyword search is BM25 with standard parameters (k1, b configurable;
 *   defaults k1=1.5, b=0.75). The IDF table is invalidated on every
 *   `upsert` / `delete` call and rebuilt lazily on the next
 *   `searchKeyword` (cost: O(unique terms in corpus)). The average doc
 *   length is recomputed eagerly inside the write path itself
 *   (cost: O(chunks in corpus)). For interleaved write-heavy / read-
 *   heavy workloads this rebuild cost is the bottleneck — swap in a
 *   real adapter (pgvector, Turbopuffer) when corpus size or write
 *   rate makes this measurable.
 * - With `useStemming: true`, both indexing and query tokens go through
 *   Porter stemming + English stopword removal. This is the standard
 *   Lucene/Elasticsearch keyword pipeline and yields material recall
 *   gains on natural-language queries (running ↔ runs, connection ↔
 *   connections). Default is `false` for backwards compatibility.
 */
export interface InMemoryAdapterOptions {
  /** Stem indexed and query tokens (Porter) and drop stopwords. Default false. */
  useStemming?: boolean;
  /** BM25 k1. Default 1.5. */
  k1?: number;
  /** BM25 b. Default 0.75. */
  b?: number;
}

export class InMemoryAdapter extends BaseAdapter {
  readonly name = "in-memory";
  readonly capabilities: AdapterCapabilities = {
    vector: true,
    keyword: true,
    hybrid: true,
    computesEmbeddings: false,
    filtering: true,
  };

  private chunks = new Map<string, Chunk>();
  private invertedIndex = new Map<string, Set<string>>();
  private termFreq = new Map<string, Map<string, number>>();
  private docLen = new Map<string, number>();
  private idfCache: Map<string, number> | null = null;
  private avgDocLen = 0;

  private k1: number;
  private b: number;
  private tokenizer: (text: string) => string[];

  constructor(opts: InMemoryAdapterOptions = {}) {
    super();
    this.k1 = opts.k1 ?? 1.5;
    this.b = opts.b ?? 0.75;
    this.tokenizer = opts.useStemming
      ? (text: string) => tokenizeAdvanced(text, { stem: true, dropStopwords: true })
      : tokenize;
  }

  async upsert(chunks: Chunk[]): Promise<void> {
    for (const chunk of chunks) {
      this.chunks.set(chunk.id, chunk);
      const tokens = this.tokenizer(chunk.content);
      this.docLen.set(chunk.id, tokens.length);

      // Reset previous postings for this chunk if it existed.
      const prevTf = this.termFreq.get(chunk.id);
      if (prevTf) {
        for (const tok of prevTf.keys()) {
          this.invertedIndex.get(tok)?.delete(chunk.id);
        }
      }

      const tf = new Map<string, number>();
      for (const tok of tokens) {
        tf.set(tok, (tf.get(tok) ?? 0) + 1);
        let postings = this.invertedIndex.get(tok);
        if (!postings) {
          postings = new Set();
          this.invertedIndex.set(tok, postings);
        }
        postings.add(chunk.id);
      }
      this.termFreq.set(chunk.id, tf);
    }
    this.recomputeAvgDocLen();
    this.idfCache = null;
  }

  async delete(ids: string[]): Promise<void> {
    for (const id of ids) {
      const tf = this.termFreq.get(id);
      if (tf) {
        for (const tok of tf.keys()) {
          this.invertedIndex.get(tok)?.delete(id);
        }
      }
      this.chunks.delete(id);
      this.termFreq.delete(id);
      this.docLen.delete(id);
    }
    this.recomputeAvgDocLen();
    this.idfCache = null;
  }

  async count(): Promise<number> {
    return this.chunks.size;
  }

  async clear(): Promise<void> {
    this.chunks.clear();
    this.invertedIndex.clear();
    this.termFreq.clear();
    this.docLen.clear();
    this.idfCache = null;
    this.avgDocLen = 0;
  }

  async searchVector(opts: VectorSearchOpts): Promise<SearchResult[]> {
    const { embedding, topK, filter } = opts;
    const results: SearchResult[] = [];
    for (const chunk of this.chunks.values()) {
      if (!chunk.embedding) continue;
      if (filter && !matchesFilter(chunk, filter)) continue;
      const score = cosine(embedding, chunk.embedding);
      results.push({ chunk, score });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  async searchKeyword(opts: KeywordSearchOpts): Promise<SearchResult[]> {
    const { query, topK, filter } = opts;
    const queryTokens = Array.from(new Set(this.tokenizer(query)));
    if (queryTokens.length === 0) return [];

    if (!this.idfCache) this.recomputeIdf();
    const idf = this.idfCache!;

    // Candidate set: union of all postings for query terms.
    const candidates = new Set<string>();
    for (const tok of queryTokens) {
      const postings = this.invertedIndex.get(tok);
      if (postings) for (const id of postings) candidates.add(id);
    }

    const { k1, b } = this;
    const results: SearchResult[] = [];

    for (const id of candidates) {
      const chunk = this.chunks.get(id)!;
      if (filter && !matchesFilter(chunk, filter)) continue;
      const tf = this.termFreq.get(id)!;
      const len = this.docLen.get(id)!;
      let score = 0;
      for (const tok of queryTokens) {
        const f = tf.get(tok) ?? 0;
        if (f === 0) continue;
        const termIdf = idf.get(tok) ?? 0;
        const norm = 1 - b + b * (len / (this.avgDocLen || 1));
        score += termIdf * ((f * (k1 + 1)) / (f + k1 * norm));
      }
      if (score > 0) results.push({ chunk, score });
    }

    // Phrase-substring boost. BM25 ignores word contiguity, so a doc with
    // the literal phrase loses to a doc that mentions the same words far
    // apart. For quoted phrases and short queries this is the dominant
    // failure mode (eval traced it to ~9 of 34 errors). One substring check
    // per candidate is sub-millisecond on 30-50 candidates.
    const boosted = boostByPhrase(results, query);
    boosted.sort((a, b) => b.score - a.score);
    return boosted.slice(0, topK);
  }

  // ---------- internals ----------

  private recomputeAvgDocLen(): void {
    if (this.docLen.size === 0) {
      this.avgDocLen = 0;
      return;
    }
    let total = 0;
    for (const len of this.docLen.values()) total += len;
    this.avgDocLen = total / this.docLen.size;
  }

  private recomputeIdf(): void {
    const N = this.chunks.size || 1;
    const idf = new Map<string, number>();
    for (const [tok, postings] of this.invertedIndex) {
      const df = postings.size;
      // BM25 IDF with the +1 smoothing variant.
      idf.set(tok, Math.log(1 + (N - df + 0.5) / (df + 0.5)));
    }
    this.idfCache = idf;
  }
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const ai = a[i]!;
    const bi = b[i]!;
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function matchesFilter(chunk: Chunk, filter: Record<string, unknown>): boolean {
  const meta = chunk.metadata;
  if (!meta) return false;
  for (const [k, v] of Object.entries(filter)) {
    if (meta[k] !== v) return false;
  }
  return true;
}

/**
 * Pull phrases worth boosting on contiguous-substring match:
 *   - every quoted span in the query (always — these are explicit phrase intents)
 *   - for queries ≤6 tokens, every 2- and 3-token contiguous window of the
 *     unquoted portion (catches things like `503 Service Unavailable`,
 *     `exit code 137`, `SSL_ERROR_SYSCALL`)
 *
 * Longer queries are skipped to avoid quadratic-feeling boosts on natural-
 * language input where contiguous trigrams aren't a meaningful intent signal.
 */
function extractPhrases(query: string): string[] {
  const phrases: string[] = [];
  const quotedRe = /["']([^"']+)["']/g;
  let m: RegExpExecArray | null;
  while ((m = quotedRe.exec(query)) !== null) {
    const inner = m[1]?.trim();
    if (inner && inner.length >= 3) phrases.push(inner.toLowerCase());
  }
  const stripped = query.replace(quotedRe, " ").trim();
  const tokens = stripped.split(/\s+/).filter(Boolean);
  if (tokens.length >= 2 && tokens.length <= 6) {
    for (let n = Math.min(3, tokens.length); n >= 2; n--) {
      for (let i = 0; i + n <= tokens.length; i++) {
        const slice = tokens.slice(i, i + n);
        // Skip windows containing a stopword. "who created" or "the
        // database" are common across the corpus and produce false-positive
        // boosts. We want selective phrases like "exit code" or "least
        // privilege" — pure content words.
        if (slice.some((t) => STOPWORDS.has(t.toLowerCase()))) continue;
        const phrase = slice.join(" ").toLowerCase();
        if (phrase.length >= 5) phrases.push(phrase);
      }
    }
  }
  return phrases;
}

/**
 * Multiplicative boost for candidates whose content contains an extracted
 * phrase as a literal substring. Bonus scales with phrase length: a 3-word
 * match is worth more than a 2-word match. The 0.15 coefficient was tuned
 * against the bundled 504-query eval — too small and the boost doesn't
 * change the rank order, too large and it overrules legitimate BM25 signal.
 */
function boostByPhrase(results: SearchResult[], query: string): SearchResult[] {
  const phrases = extractPhrases(query);
  if (phrases.length === 0) return results;
  return results.map((r) => {
    const content = r.chunk.content.toLowerCase();
    let bonus = 0;
    for (const p of phrases) {
      if (content.includes(p)) bonus += p.split(/\s+/).length;
    }
    if (bonus === 0) return r;
    return {
      ...r,
      score: r.score * (1 + bonus * 0.15),
      rawScores: { ...r.rawScores, phraseBoost: bonus },
    };
  });
}
