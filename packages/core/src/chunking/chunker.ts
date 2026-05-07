import type { Chunk, Document } from "../types.js";

/**
 * Chunker interface.
 *
 * Why a Chunker is a discrete component:
 * - Different content types (code, prose, transcripts, structured docs) want
 *   different chunking strategies.
 * - Chunking is the #1 quality lever in RAG that nobody tunes — making it
 *   pluggable lets users experiment without rewriting their pipeline.
 *
 * Implementations should:
 * - Be deterministic given the same input (so chunk IDs are stable)
 * - Respect the user's content boundaries when meaningful
 * - Generate stable chunk IDs of the form `${docId}:${index}`
 */
export interface Chunker {
  readonly name: string;
  chunk(doc: Document): Chunk[];
}

/**
 * FixedSizeChunker — splits on a token/character window with overlap.
 *
 * Tradeoff: fast, predictable, language-agnostic. Frequently splits mid-sentence
 * which can degrade retrieval. Use when:
 * - Content is highly structured/uniform (logs, code, transcripts)
 * - You need predictable chunk counts for cost modeling
 *
 * The overlap parameter mitigates the mid-sentence problem by ensuring context
 * straddles boundaries.
 */
export class FixedSizeChunker implements Chunker {
  readonly name = "fixed-size";
  private size: number;
  private overlap: number;

  constructor(opts: { size?: number; overlap?: number } = {}) {
    this.size = opts.size ?? 800;
    this.overlap = opts.overlap ?? 100;
    if (this.overlap >= this.size) {
      throw new Error("FixedSizeChunker: overlap must be smaller than size");
    }
  }

  chunk(doc: Document): Chunk[] {
    const chunks: Chunk[] = [];
    const text = doc.content;
    if (text.length === 0) return chunks;

    let i = 0;
    let idx = 0;
    while (i < text.length) {
      const slice = text.slice(i, i + this.size).trim();
      if (slice.length > 0) {
        chunks.push({
          id: `${doc.id}:${idx}`,
          documentId: doc.id,
          content: slice,
          index: idx,
          metadata: doc.metadata,
        });
        idx += 1;
      }
      i += this.size - this.overlap;
    }
    return chunks;
  }
}

/**
 * SentenceChunker — groups consecutive sentences up to a target length.
 *
 * Why this is usually the right default:
 * - Sentence boundaries respect grammatical units → better embeddings.
 * - Variable size, but capped, so cost is bounded.
 * - The grouping logic ("pack until target") avoids hyper-tiny chunks for
 *   short sentences and oversized ones for long ones.
 *
 * Limitation: regex-based sentence splitting is imperfect for English and
 * worse for non-Latin scripts. For production use with mixed languages,
 * swap in an Intl.Segmenter implementation or a model-based splitter.
 */
export class SentenceChunker implements Chunker {
  readonly name = "sentence";
  private targetSize: number;
  private maxSize: number;

  constructor(opts: { targetSize?: number; maxSize?: number } = {}) {
    this.targetSize = opts.targetSize ?? 600;
    this.maxSize = opts.maxSize ?? 1200;
  }

  chunk(doc: Document): Chunk[] {
    const sentences = splitSentences(doc.content);
    const chunks: Chunk[] = [];
    let buf: string[] = [];
    let bufLen = 0;
    let idx = 0;

    const flush = () => {
      if (buf.length === 0) return;
      const content = buf.join(" ").trim();
      if (content.length > 0) {
        chunks.push({
          id: `${doc.id}:${idx}`,
          documentId: doc.id,
          content,
          index: idx,
          metadata: doc.metadata,
        });
        idx += 1;
      }
      buf = [];
      bufLen = 0;
    };

    for (const s of sentences) {
      if (bufLen + s.length > this.maxSize && buf.length > 0) flush();
      buf.push(s);
      bufLen += s.length + 1;
      if (bufLen >= this.targetSize) flush();
    }
    flush();
    return chunks;
  }
}

/**
 * SemanticChunker — groups sentences whose embeddings are similar.
 *
 * The idea: instead of size, use cosine distance between adjacent sentence
 * embeddings as the boundary signal. When the topic shifts, similarity drops,
 * and that's where we cut.
 *
 * Tradeoff:
 * - Higher quality on long-form content (essays, long docs).
 * - Costs N embedding calls per document at chunking time.
 * - We use the same Embedder that the rest of the system uses, so callers
 *   can opt to use a cheap one (HashEmbedder) for chunking and a real one
 *   for indexing — that's a sound default.
 *
 * This is intentionally simple: a 1-pass scan with a similarity threshold.
 * More elaborate variants (sliding-window outliers, double-pass) are easy to
 * layer on top by implementing the Chunker interface.
 */
export class SemanticChunker implements Chunker {
  readonly name = "semantic";
  private embedder: import("../embeddings/embedder.js").Embedder;
  private threshold: number;
  private maxSize: number;

  constructor(opts: {
    embedder: import("../embeddings/embedder.js").Embedder;
    /** Cosine similarity threshold below which we cut. 0..1. Lower = more chunks. */
    threshold?: number;
    /** Hard cap on chunk size in chars to prevent runaway groupings. */
    maxSize?: number;
  }) {
    this.embedder = opts.embedder;
    this.threshold = opts.threshold ?? 0.65;
    this.maxSize = opts.maxSize ?? 1500;
  }

  chunk(_doc: Document): Chunk[] {
    throw new Error("SemanticChunker is async; use chunkAsync()");
  }

  async chunkAsync(doc: Document): Promise<Chunk[]> {
    const sentences = splitSentences(doc.content);
    if (sentences.length === 0) return [];
    const embeddings = await this.embedder.embed(sentences);

    const chunks: Chunk[] = [];
    let buf: string[] = [sentences[0]!];
    let bufLen = sentences[0]!.length;
    let idx = 0;

    for (let i = 1; i < sentences.length; i++) {
      const sim = cosine(embeddings[i - 1]!, embeddings[i]!);
      const wouldOverflow = bufLen + sentences[i]!.length > this.maxSize;
      if (sim < this.threshold || wouldOverflow) {
        chunks.push({
          id: `${doc.id}:${idx}`,
          documentId: doc.id,
          content: buf.join(" ").trim(),
          index: idx,
          metadata: doc.metadata,
        });
        idx += 1;
        buf = [];
        bufLen = 0;
      }
      buf.push(sentences[i]!);
      bufLen += sentences[i]!.length + 1;
    }
    if (buf.length > 0) {
      chunks.push({
        id: `${doc.id}:${idx}`,
        documentId: doc.id,
        content: buf.join(" ").trim(),
        index: idx,
        metadata: doc.metadata,
      });
    }
    return chunks;
  }
}

/** Compatibility helper: any Chunker, sync or async. */
export async function chunkDocument(
  chunker: Chunker | SemanticChunker,
  doc: Document
): Promise<Chunk[]> {
  if (chunker instanceof SemanticChunker) return chunker.chunkAsync(doc);
  return chunker.chunk(doc);
}

// ---------- Helpers ----------

/**
 * Naive sentence splitter. Handles common abbreviations minimally.
 * For production-grade splitting, swap with Intl.Segmenter.
 */
function splitSentences(text: string): string[] {
  // Protect a few common abbreviations from being treated as sentence ends.
  const protectedText = text
    .replace(/\b(Mr|Mrs|Ms|Dr|Sr|Jr|St|Inc|Ltd|e\.g|i\.e)\.\s/g, "$1<DOT> ")
    .replace(/\b([A-Z])\.\s/g, "$1<DOT> ");
  const parts = protectedText
    .split(/(?<=[.!?])\s+(?=[A-Z(\["'])/u)
    .map((s) => s.replace(/<DOT>/g, ".").trim())
    .filter((s) => s.length > 0);
  // If we got nothing useful, fall back to newline-split.
  if (parts.length <= 1) {
    return text.split(/\n+/).map((s) => s.trim()).filter(Boolean);
  }
  return parts;
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
