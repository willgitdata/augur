/**
 * `@augur-rag/langchain` — LangChain.js binding for Augur.
 *
 * The binding is intentionally minimal: a single function that runs an
 * Augur search and reshapes the results into LangChain's `Document`
 * shape. The reason we don't subclass `BaseRetriever` directly is that
 * doing so would force a hard dependency on `@langchain/core` (and
 * pinning to a particular major version of it). Instead we publish
 * the shape — `{ pageContent, metadata }` — and a 5-line subclass
 * snippet that wires it into the user's already-installed LangChain
 * version.
 *
 * Usage:
 *
 *   import { searchAsLangchainDocs } from "@augur-rag/langchain";
 *   import { BaseRetriever } from "@langchain/core/retrievers";
 *
 *   class AugurRetriever extends BaseRetriever {
 *     lc_namespace = ["augur"];
 *     constructor(private augur: Augur, private topK = 10) { super(); }
 *     async _getRelevantDocuments(query: string) {
 *       return searchAsLangchainDocs(this.augur, query, { topK: this.topK });
 *     }
 *   }
 *
 *   const retriever = new AugurRetriever(augur);
 *   const docs = await retriever.invoke("how do I X?");
 */

import type { Augur } from "@augur-rag/core";

/**
 * Mirrors `Document` from `@langchain/core/documents` without pulling
 * the package in as a hard dep. The shape is stable across LangChain
 * v0.1 → v0.3; if upstream breaks it we'll publish a new major of this
 * binding and the JSDoc snippet above will change.
 */
export interface LangchainDocLike {
  pageContent: string;
  metadata: Record<string, unknown>;
}

export interface SearchAsLangchainDocsOptions {
  /** topK passed through to `augur.search`. Default 10. */
  topK?: number;
  /** Metadata filter passed through to `augur.search`. */
  filter?: Record<string, unknown>;
  /**
   * Optional latency budget in ms — passed through to the router. Useful
   * when LangChain wraps the retriever in a streaming chain and you
   * want a soft cap on retrieval time.
   */
  latencyBudgetMs?: number;
  /**
   * Optional confidence floor. Results scored below this are dropped.
   * Sometimes useful in LangChain QA chains where "no answer" is a
   * better signal than a noisy low-relevance hit.
   */
  minScore?: number;
}

/**
 * Run an Augur search and convert results to the LangChain `Document`
 * shape. Carries the chunk id, document id, and final score through in
 * `metadata` so downstream chains (citation rendering, score-based
 * filtering, observability) have them. Original chunk metadata is
 * spread in last so user-supplied fields don't collide with the
 * binding-added ones.
 */
export async function searchAsLangchainDocs(
  augur: Augur,
  query: string,
  opts: SearchAsLangchainDocsOptions = {}
): Promise<LangchainDocLike[]> {
  const req: Parameters<Augur["search"]>[0] = {
    query,
    topK: opts.topK ?? 10,
  };
  if (opts.filter) req.filter = opts.filter;
  if (opts.latencyBudgetMs !== undefined) req.latencyBudgetMs = opts.latencyBudgetMs;
  if (opts.minScore !== undefined) req.minScore = opts.minScore;

  const { results } = await augur.search(req);
  return results.map((r) => ({
    pageContent: r.chunk.content,
    metadata: {
      ...(r.chunk.metadata ?? {}),
      chunkId: r.chunk.id,
      documentId: r.chunk.documentId,
      score: r.score,
    },
  }));
}
