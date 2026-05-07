import type { Chunk, Document } from "../types.js";
import type { Chunker } from "./chunker.js";
import { chunkDocument, SemanticChunker } from "./chunker.js";

/**
 * MetadataChunker — wraps a base chunker, prepending document metadata to
 * each chunk's content before storage and embedding.
 *
 * This is the "Doc2Query lite" idea from production retrieval systems:
 * at index time, augment chunk content with structured signals (title,
 * section, topic) so the embedder and the BM25 index see them. Cheap,
 * deterministic, and reliably yields +5-15% recall on conversational
 * queries that use natural-language descriptions of metadata fields.
 *
 * Trade-off: the prefix appears in the chunk content returned to users.
 * Strip it on display if you don't want it visible, or keep it as a
 * compact citation header.
 *
 * Default formatter:
 *   [doc-id | topic: <topic> | title: <title>]\n<content>
 *
 * Pass a custom `formatPrefix` to control the format.
 */
export class MetadataChunker implements Chunker {
  readonly name: string;
  private base: Chunker | SemanticChunker;
  private formatPrefix: (doc: Document) => string;

  constructor(opts: {
    base: Chunker | SemanticChunker;
    formatPrefix?: (doc: Document) => string;
  }) {
    this.base = opts.base;
    this.formatPrefix = opts.formatPrefix ?? defaultPrefix;
    this.name = `metadata(${(opts.base as Chunker).name ?? "base"})`;
  }

  chunk(doc: Document): Chunk[] {
    if (this.base instanceof SemanticChunker) {
      throw new Error(
        "MetadataChunker base is async (SemanticChunker); use chunkAsync via chunkDocument()"
      );
    }
    const chunks = (this.base as Chunker).chunk(doc);
    return this.prefixChunks(chunks, doc);
  }

  async chunkAsync(doc: Document): Promise<Chunk[]> {
    const chunks = await chunkDocument(this.base, doc);
    return this.prefixChunks(chunks, doc);
  }

  private prefixChunks(chunks: Chunk[], doc: Document): Chunk[] {
    const prefix = this.formatPrefix(doc).trim();
    if (!prefix) return chunks;
    return chunks.map((c) => ({
      ...c,
      content: `${prefix}\n${c.content}`,
    }));
  }
}

function defaultPrefix(doc: Document): string {
  const parts: string[] = [doc.id];
  const meta = doc.metadata ?? {};
  for (const key of ["title", "topic", "section", "kind", "lang"]) {
    const v = meta[key];
    if (typeof v === "string" && v.length > 0) {
      parts.push(`${key}: ${v}`);
    }
  }
  return `[${parts.join(" | ")}]`;
}
