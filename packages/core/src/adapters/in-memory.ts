import { tokenize } from "../embeddings/embedder.js";
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
 * - Keyword search is BM25 with standard parameters (k1=1.5, b=0.75).
 *   We rebuild the IDF table lazily on insert. For high-write workloads
 *   this is a known bottleneck — again, swap in a real adapter at scale.
 */
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
  // Inverted index: token -> set of chunk IDs.
  private invertedIndex = new Map<string, Set<string>>();
  // Per-chunk token frequencies for BM25.
  private termFreq = new Map<string, Map<string, number>>();
  // Per-chunk total length (tokens).
  private docLen = new Map<string, number>();
  // Cached IDF values, invalidated on every upsert/delete.
  private idfCache: Map<string, number> | null = null;
  private avgDocLen = 0;

  async upsert(chunks: Chunk[]): Promise<void> {
    for (const chunk of chunks) {
      this.chunks.set(chunk.id, chunk);
      const tokens = tokenize(chunk.content);
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
    const queryTokens = Array.from(new Set(tokenize(query)));
    if (queryTokens.length === 0) return [];

    if (!this.idfCache) this.recomputeIdf();
    const idf = this.idfCache!;

    // Candidate set: union of all postings for query terms.
    const candidates = new Set<string>();
    for (const tok of queryTokens) {
      const postings = this.invertedIndex.get(tok);
      if (postings) for (const id of postings) candidates.add(id);
    }

    const k1 = 1.5;
    const b = 0.75;
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

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
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
