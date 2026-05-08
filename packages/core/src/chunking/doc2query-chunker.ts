/**
 * Doc2QueryChunker — at index time, generate synthetic questions each
 * chunk could answer, then append them to the chunk's content. The
 * embedder and BM25 index see the union of (real content) + (synthesized
 * questions), which closes the lexical gap between conversational queries
 * and reference-style source material.
 *
 * Why this works: production retrieval systems (msmarco, Anthropic's
 * internal RAG, doc2query/T5 paper) consistently find +5-15% recall on
 * conversational queries from this trick. The cost is paid once, at
 * indexing. Query-time latency is unchanged.
 *
 * Default generator: `Xenova/LaMini-T5-61M` (~24MB ONNX, instruction-
 * tuned for "small but coherent" tasks). Override with any sequence-to-
 * sequence model on HuggingFace that's been ONNX-converted under the
 * `Xenova/` namespace (e.g. `Xenova/flan-t5-small`, `Xenova/t5-small`).
 *
 * Like LocalEmbedder, this does a dynamic import of
 * `@huggingface/transformers` so consumers who don't use Doc2Query don't
 * need the package installed.
 */

import type { Chunk, Document } from "../types.js";
import type { Chunker } from "./chunker.js";
import { chunkDocument, SemanticChunker } from "./chunker.js";

const pipelineCache = new Map<string, Promise<unknown>>();

async function getGenerationPipeline(model: string): Promise<unknown> {
  let p = pipelineCache.get(model);
  if (p) return p;
  p = (async () => {
    const transformers = (await import("@huggingface/transformers")) as unknown as {
      pipeline: (task: string, model: string) => Promise<unknown>;
    };
    return transformers.pipeline("text2text-generation", model);
  })();
  pipelineCache.set(model, p);
  return p;
}

export interface Doc2QueryChunkerOptions {
  /** Base chunker that produces the original chunks. Required. */
  base: Chunker | SemanticChunker;
  /** ONNX text2text model used to generate synthetic queries. Default: Xenova/LaMini-T5-61M. */
  model?: string;
  /** How many synthetic queries per chunk. More = better recall, longer index time. Default 3. */
  numQueries?: number;
  /** Generation max length. Short questions are fine for retrieval. Default 32. */
  maxLength?: number;
  /**
   * Prompt template. The model sees `prompt(chunk.content)`. Default
   * is the standard doc2query phrasing: "Generate a question this
   * passage answers: <text>".
   */
  prompt?: (text: string) => string;
  /**
   * How to combine synthetic queries with the chunk's content. Default
   * appends them as a separate "Questions answered:" block so they
   * influence BM25 + embedding without obliterating the original prose.
   */
  format?: (originalContent: string, queries: string[]) => string;
}

export class Doc2QueryChunker implements Chunker {
  readonly name: string;
  private base: Chunker | SemanticChunker;
  private model: string;
  private numQueries: number;
  private maxLength: number;
  private prompt: (text: string) => string;
  private format: (original: string, queries: string[]) => string;

  constructor(opts: Doc2QueryChunkerOptions) {
    this.base = opts.base;
    this.model = opts.model ?? "Xenova/LaMini-T5-61M";
    this.numQueries = opts.numQueries ?? 3;
    this.maxLength = opts.maxLength ?? 32;
    this.prompt = opts.prompt ?? ((text) => `Generate a question this passage answers: ${text}`);
    this.format =
      opts.format ??
      ((original, queries) =>
        queries.length === 0
          ? original
          : `${original}\n\nQuestions answered:\n${queries.map((q) => `- ${q}`).join("\n")}`);
    this.name = `doc2query(${(opts.base as Chunker).name ?? "base"}, ${this.model})`;
  }

  chunk(_doc: Document): Chunk[] {
    throw new Error("Doc2QueryChunker is async; use chunkAsync() via chunkDocument()");
  }

  async chunkAsync(doc: Document): Promise<Chunk[]> {
    const baseChunks = await chunkDocument(this.base, doc);
    if (baseChunks.length === 0) return [];

    const pipe = (await getGenerationPipeline(this.model)) as (
      input: string | string[],
      opts: {
        max_length?: number;
        num_return_sequences?: number;
        do_sample?: boolean;
        top_k?: number;
        top_p?: number;
      }
    ) => Promise<Array<{ generated_text: string }> | Array<Array<{ generated_text: string }>>>;

    const augmented: Chunk[] = [];
    for (const c of baseChunks) {
      const prompted = this.prompt(c.content);
      // T5 generation: ask for N candidates with sampling for diversity.
      const raw = await pipe(prompted, {
        max_length: this.maxLength,
        num_return_sequences: this.numQueries,
        do_sample: true,
        top_k: 50,
        top_p: 0.95,
      });
      const flat = Array.isArray(raw[0]) ? (raw as Array<Array<{ generated_text: string }>>)[0]! : (raw as Array<{ generated_text: string }>);
      const queries = Array.from(new Set(flat.map((r) => r.generated_text.trim()))).filter(Boolean);
      augmented.push({
        ...c,
        content: this.format(c.content, queries),
      });
    }
    return augmented;
  }
}
