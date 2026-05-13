/**
 * `@augur-rag/llamaindex` — LlamaIndex.ts binding for Augur.
 *
 * Same minimal pattern as `@augur-rag/langchain`: we don't subclass
 * LlamaIndex's `BaseRetriever` directly because that would force a
 * hard dep on `llamaindex` and pin to a particular major. Instead we
 * publish the `NodeWithScore` shape and a 5-line subclass snippet.
 *
 * Usage:
 *
 *   import { searchAsLlamaIndexNodes } from "@augur-rag/llamaindex";
 *   import { BaseRetriever, NodeWithScore } from "llamaindex";
 *
 *   class AugurRetriever extends BaseRetriever {
 *     constructor(private augur: Augur, private topK = 10) { super(); }
 *     async _retrieve({ query }: { query: string }): Promise<NodeWithScore[]> {
 *       return searchAsLlamaIndexNodes(this.augur, query, { topK: this.topK });
 *     }
 *   }
 */

import type { Augur } from "@augur-rag/core";

/**
 * Mirrors LlamaIndex.ts's `NodeWithScore` shape without the dep. The
 * `TextNode` flavour is the most common; if your pipeline expects a
 * different node type, post-process the result.
 */
export interface LlamaIndexNodeWithScoreLike {
  node: {
    id_: string;
    text: string;
    metadata: Record<string, unknown>;
  };
  score: number;
}

export interface SearchAsLlamaIndexNodesOptions {
  /** topK passed through to `augur.search`. Default 10. */
  topK?: number;
  /** Metadata filter passed through to `augur.search`. */
  filter?: Record<string, unknown>;
  /** Soft latency budget in ms. */
  latencyBudgetMs?: number;
  /** Confidence floor — drops results below this score. */
  minScore?: number;
}

/**
 * Run an Augur search and convert results to the LlamaIndex
 * `NodeWithScore` shape (with `TextNode`-like nodes inside). Carries
 * the chunk's documentId into `metadata` so LlamaIndex citation flows
 * still work.
 */
export async function searchAsLlamaIndexNodes(
  augur: Augur,
  query: string,
  opts: SearchAsLlamaIndexNodesOptions = {}
): Promise<LlamaIndexNodeWithScoreLike[]> {
  const req: Parameters<Augur["search"]>[0] = {
    query,
    topK: opts.topK ?? 10,
  };
  if (opts.filter) req.filter = opts.filter;
  if (opts.latencyBudgetMs !== undefined) req.latencyBudgetMs = opts.latencyBudgetMs;
  if (opts.minScore !== undefined) req.minScore = opts.minScore;

  const { results } = await augur.search(req);
  return results.map((r) => ({
    node: {
      id_: r.chunk.id,
      text: r.chunk.content,
      metadata: {
        ...(r.chunk.metadata ?? {}),
        documentId: r.chunk.documentId,
      },
    },
    score: r.score,
  }));
}
