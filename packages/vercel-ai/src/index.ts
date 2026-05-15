/**
 * `@augur-rag/vercel-ai` — Vercel AI SDK binding for Augur.
 *
 * Vercel's AI SDK lets language models call tools you describe with
 * `tool({ description, parameters, execute })`. This binding returns
 * a ready-made descriptor pointing at `Augur.search`, so a model can
 * pull retrieval results into its context whenever it decides to.
 *
 * Why a descriptor rather than calling `tool()` ourselves: Vercel's
 * `tool()` is a thin identity helper that exists for type inference;
 * by returning the descriptor we avoid a hard dep on the `ai` package
 * and stay version-agnostic. The user passes our return value as the
 * single argument to `tool()` (or, in older AI SDK versions, directly
 * into the `tools` map).
 *
 * Usage:
 *
 *   import { tool } from "ai";
 *   import { generateText } from "ai";
 *   import { augurToolDescriptor } from "@augur-rag/vercel-ai";
 *
 *   const augurTool = tool(augurToolDescriptor(augur, { topK: 5 }));
 *
 *   const { text } = await generateText({
 *     model,
 *     tools: { search_corpus: augurTool },
 *     prompt: "...",
 *   });
 */

import type { Augur } from "@augur-rag/core";

export interface AugurToolDescriptorOptions {
  /**
   * Description the model sees when deciding whether to call this
   * tool. Default is generic; override with a corpus-specific hint
   * for better tool selection ("Retrieve passages from the company's
   * internal Postgres handbook…").
   */
  description?: string;
  /** topK passed through to `augur.search`. Default 5 — tools are usually called for tight LLM context. */
  topK?: number;
  /**
   * Soft latency budget passed to the router. Default `undefined`
   * (no constraint). Useful when the tool is part of a streaming
   * response loop and you want a cap on per-call retrieval time.
   */
  latencyBudgetMs?: number;
}

/**
 * Shape of a single result the tool returns to the model. Trimmed to
 * the fields a model can actually reason about — full chunk metadata
 * is omitted to keep the per-call token cost low.
 */
export interface AugurToolResult {
  content: string;
  documentId: string;
  score: number;
}

/**
 * Vercel AI SDK tool descriptor for an Augur search. The shape (
 * `description`, `parameters`, `execute`) is what `tool()` from
 * the `ai` package expects.
 */
export interface AugurToolDescriptor {
  description: string;
  parameters: {
    type: "object";
    properties: {
      query: { type: "string"; description: string };
    };
    required: ["query"];
  };
  execute: (
    args: { query: string }
  ) => Promise<{ results: AugurToolResult[] }>;
}

/**
 * Build a Vercel AI SDK tool descriptor that calls Augur's search and
 * returns the matched passages. Pass the descriptor into `tool()`
 * (from the `ai` package) and register the result in the `tools` map
 * on `generateText` / `streamText` / similar.
 */
export function augurToolDescriptor(
  augur: Augur,
  opts: AugurToolDescriptorOptions = {}
): AugurToolDescriptor {
  const topK = opts.topK ?? 5;
  const latencyBudgetMs = opts.latencyBudgetMs;
  return {
    description:
      opts.description ??
      "Retrieve passages from the indexed corpus that are relevant to the query. " +
        "Returns each passage's content, its source documentId, and a relevance score.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Natural-language query that the model wants to retrieve passages for.",
        },
      },
      required: ["query"],
    },
    async execute({ query }) {
      const req: Parameters<Augur["search"]>[0] = { query, topK };
      if (latencyBudgetMs !== undefined) req.latencyBudgetMs = latencyBudgetMs;
      const { results } = await augur.search(req);
      return {
        results: results.map((r) => ({
          content: r.chunk.content,
          documentId: r.chunk.documentId,
          score: r.score,
        })),
      };
    },
  };
}
