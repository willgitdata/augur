import { BoundedCache } from "../internal/bounded-cache.js";
import { sha256Hex, utf8ByteLength } from "../internal/sha256.js";
import type { Chunk, Document } from "../types.js";
import type { AsyncChunker, Chunker } from "./chunker.js";
import { chunkDocument } from "./chunker.js";

/**
 * ContextualChunker — implementation of Anthropic's
 * [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
 * pattern.
 *
 * The problem: a sentence/paragraph chunk taken out of its document loses
 * the context that disambiguates it. "The company's revenue grew 3%" is
 * useless without knowing which company, which year, which segment. Bare
 * embeddings of that chunk match poorly against queries like "ACME 2024
 * Q4 earnings."
 *
 * The fix: at indexing time, send each chunk + its source document to a
 * fast LLM (Haiku, GPT-4o-mini, Gemini Flash) and ask for a one-line
 * contextual description. Prepend that description to the chunk content
 * before embedding and indexing. Anthropic measured chunk failure rate
 * dropping from 5.7% → 1.9% (a 67% reduction) using this technique on
 * their internal eval, and the same technique stacks with hybrid
 * retrieval and reranking.
 *
 * Cost: one LLM call per chunk, indexed once. With prompt caching on
 * Anthropic's API, the document portion (the bulk of input tokens) is
 * cached across all chunks of the same doc, making the per-chunk cost
 * roughly the chunk size in tokens — typically a few cents per thousand
 * chunks at Haiku rates.
 *
 * Usage:
 *
 *   import Anthropic from "@anthropic-ai/sdk";
 *   import { sanitizeForContextualPrompt } from "@augur-rag/core";
 *   const client = new Anthropic();
 *
 *   const chunker = new ContextualChunker({
 *     base: new SentenceChunker(),
 *     provider: {
 *       name: "anthropic:claude-haiku-4-5",
 *       async contextualize({ chunk, document }) {
 *         // IMPORTANT: sanitize before substitution. A document containing
 *         // `</document>` or `</chunk>` would otherwise let an attacker
 *         // inject instructions into the LLM call (stored prompt injection).
 *         const r = await client.messages.create({
 *           model: "claude-haiku-4-5",
 *           max_tokens: 100,
 *           messages: [{
 *             role: "user",
 *             content: ANTHROPIC_CONTEXTUAL_PROMPT
 *               .replace("{WHOLE_DOCUMENT}", sanitizeForContextualPrompt(document))
 *               .replace("{CHUNK_CONTENT}", sanitizeForContextualPrompt(chunk)),
 *           }],
 *         });
 *         const block = r.content[0];
 *         return block && block.type === "text" ? block.text : "";
 *       },
 *     },
 *   });
 *
 * The provider is intentionally minimal — the LLM call lives in user
 * code so we don't pull in any vendor SDK as a dependency. Cache and
 * concurrency are handled here.
 */

/**
 * The exact prompt template Anthropic published in the
 * [Contextual Retrieval cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide).
 * Substitute `{WHOLE_DOCUMENT}` and `{CHUNK_CONTENT}` before sending.
 *
 * **Important — sanitize before substitution.** A document or chunk
 * containing literal `</document>` or `</chunk>` tags lets an attacker
 * close the template tags and inject instructions that the LLM will
 * then follow when producing the "context" string. That context is
 * prepended to every chunk and embedded into the index — a permanent
 * stored prompt-injection that contaminates retrieval forever.
 *
 * Use {@link sanitizeForContextualPrompt} on both `document` and
 * `chunk` before substituting into this template. The provider example
 * in the JSDoc above does this.
 */
export const ANTHROPIC_CONTEXTUAL_PROMPT = `<document>
{WHOLE_DOCUMENT}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.`;

/**
 * Defang the close-tag sequences `ANTHROPIC_CONTEXTUAL_PROMPT` relies on
 * so a malicious document can't escape out of the template's `<document>`
 * or `<chunk>` envelope.
 *
 * Strategy: replace `</document>` and `</chunk>` (case-insensitive, with
 * optional whitespace) with a zero-width-joiner-broken variant the LLM
 * still understands as text but doesn't parse as a closing tag. We
 * deliberately preserve the surrounding characters so the visible
 * content the LLM sees is otherwise unchanged.
 *
 * Returns the sanitized string. Idempotent — running it twice is a no-op
 * on already-sanitized content.
 */
export function sanitizeForContextualPrompt(text: string): string {
  // Match `</tag>` with optional whitespace inside, case-insensitive. Then
  // insert a zero-width joiner so the literal close-tag sequence is broken.
  return text
    .replace(/<\s*\/\s*document\s*>/gi, "<‍/document>")
    .replace(/<\s*\/\s*chunk\s*>/gi, "<‍/chunk>");
}

export interface ContextProvider {
  /** Stable identifier surfaced in trace metadata. */
  readonly name: string;
  /**
   * Return a brief contextual description for `chunk` given its source
   * document. Implementations are expected to be safe to call concurrently.
   */
  contextualize(args: { chunk: string; document: string }): Promise<string>;
}

/**
 * Pluggable cache keyed by a content hash of (document, chunk). The
 * default is in-memory; production callers should plug in a persistent
 * cache (Redis, file system, KV) so re-indexing is free for unchanged
 * content.
 */
export interface ContextCache {
  get(key: string): Promise<string | undefined> | string | undefined;
  set(key: string, value: string): Promise<void> | void;
}

/**
 * In-memory ContextCache. Bounded LRU (default 10,000 entries) so a
 * long-running indexer can't grow unbounded. Lost on restart — production
 * callers should plug in a persistent ContextCache (Redis, file system,
 * KV) instead.
 *
 * Sizing: 10k entries × ~200 bytes of context-string each ≈ 2 MB
 * worst-case. Far above any realistic per-document chunk-count working
 * set, but still safe to leave running on a server. Override via
 * `new MemoryContextCache({ capacity: ... })` if you need more.
 */
export class MemoryContextCache implements ContextCache {
  private store: BoundedCache<string, string>;
  constructor(opts: { capacity?: number } = {}) {
    this.store = new BoundedCache<string, string>(opts.capacity ?? 10_000);
  }
  get(key: string): string | undefined {
    return this.store.get(key);
  }
  set(key: string, value: string): void {
    this.store.set(key, value);
  }
  size(): number {
    return this.store.size();
  }
  clear(): void {
    this.store.clear();
  }
}

export interface ContextualChunkerOptions {
  /** Base chunker whose chunks we'll annotate. */
  base: Chunker | AsyncChunker;
  /** LLM-backed contextualizer. Called once per (cache-miss) chunk. */
  provider: ContextProvider;
  /**
   * Cache layer. Defaults to MemoryContextCache. Use a persistent cache
   * in production so re-indexing is free for unchanged docs.
   */
  cache?: ContextCache;
  /**
   * Max in-flight LLM calls. Defaults to 4. Tune to your provider's
   * rate limit. Higher = faster indexing but more risk of 429s.
   */
  concurrency?: number;
  /** Joins context and chunk content. Default: "\n\n". */
  separator?: string;
}

/**
 * Wraps a base chunker and contextualizes each chunk with an LLM-generated
 * description before storing it. The contextualized content is what gets
 * embedded by `Augur.index()`, so the embedder sees a chunk that *knows*
 * what document it came from.
 */
export class ContextualChunker implements AsyncChunker {
  readonly name: string;
  private base: Chunker | AsyncChunker;
  private provider: ContextProvider;
  private cache: ContextCache;
  private concurrency: number;
  private separator: string;

  constructor(opts: ContextualChunkerOptions) {
    this.base = opts.base;
    this.provider = opts.provider;
    this.cache = opts.cache ?? new MemoryContextCache();
    this.concurrency = opts.concurrency ?? 4;
    this.separator = opts.separator ?? "\n\n";
    this.name = `contextual(${opts.base.name})`;
  }

  async chunkAsync(doc: Document): Promise<Chunk[]> {
    const baseChunks = await chunkDocument(this.base, doc);
    if (baseChunks.length === 0) return baseChunks;

    const out: Chunk[] = new Array(baseChunks.length);
    let nextIndex = 0;

    const worker = async (): Promise<void> => {
      while (true) {
        const i = nextIndex++;
        if (i >= baseChunks.length) return;
        const chunk = baseChunks[i]!;
        out[i] = await this.contextualizeOne(chunk, doc);
      }
    };

    // Fire `concurrency` workers, each pulling from the shared queue.
    await Promise.all(
      Array.from({ length: Math.min(this.concurrency, baseChunks.length) }, () => worker())
    );

    return out;
  }

  private async contextualizeOne(chunk: Chunk, doc: Document): Promise<Chunk> {
    const key = await cacheKey(doc.content, chunk.content);
    let context = await this.cache.get(key);
    if (context === undefined) {
      context = await this.provider.contextualize({
        chunk: chunk.content,
        document: doc.content,
      });
      await this.cache.set(key, context);
    }
    const ctx = (context ?? "").trim();
    if (!ctx) return chunk;
    return { ...chunk, content: `${ctx}${this.separator}${chunk.content}` };
  }
}


/**
 * SHA-256 of (doc-content, chunk-content). Stable across runs.
 *
 * Length-prefix encoding: hash `${len(doc)}:${doc}${len(chunk)}:${chunk}`
 * so a malicious caller can't engineer a collision like
 * `(doc="A", chunk="B C")` vs `(doc="A B", chunk="C")`. The previous
 * implementation used a single NUL-byte separator, which prevented the
 * trivial space-collision but still allowed an attacker who controls
 * document content to engineer a collision by embedding NUL bytes —
 * letting one caller's context poison another's cache slot.
 *
 * Byte length, not character length, so multi-byte UTF-8 sequences
 * don't accidentally collide with shorter ASCII content.
 */
async function cacheKey(documentContent: string, chunkContent: string): Promise<string> {
  // Byte length, not character length, so multi-byte UTF-8 sequences
  // don't accidentally collide with shorter ASCII content. Web Crypto's
  // TextEncoder gives us this in every runtime (browser + edge + Node).
  const docBytes = utf8ByteLength(documentContent);
  const chunkBytes = utf8ByteLength(chunkContent);
  return sha256Hex(`${docBytes}:${documentContent}${chunkBytes}:${chunkContent}`);
}
