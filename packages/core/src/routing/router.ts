import type { AdapterCapabilities } from "../adapters/adapter.js";
import type { RoutingDecision, RetrievalStrategy, SearchRequest } from "../types.js";
import { computeSignals } from "./signals.js";

/**
 * Router interface — separates "how do we decide" from "what we decide".
 *
 * Today: HeuristicRouter (rule-based).
 * Tomorrow: MLRouter (logistic regression / small classifier trained on
 *   click logs and eval datasets). Both implement this interface, so the
 *   Augur class never has to change.
 */
export interface Router {
  readonly name: string;
  decide(req: SearchRequest, caps: AdapterCapabilities): RoutingDecision;
}

/**
 * HeuristicRouter — rule-based routing, the MVP default.
 *
 * The decision flow, in plain English:
 *
 *   1. If the user forced a strategy, honor it (record in trace).
 *   2. Compute query signals.
 *   3. If the adapter only supports vector → vector (capability fallback).
 *   4. If the query has quoted phrases or is highly specific (IDs, codes,
 *      numbers, acronyms) → keyword. Pure semantic embeddings don't reliably
 *      retrieve exact tokens.
 *   5. If the query is short AND specific → keyword (small queries embed badly).
 *   6. If the query is a question or long natural prose → vector. The latency
 *      budget decides whether to tack on reranking.
 *   7. Otherwise → hybrid. The default for "I don't know" is to do both —
 *      hybrid retrieval is rarely worse than either alone, and RRF makes it
 *      cheap to combine.
 *
 * Reranking is layered on top:
 *   - If `latencyBudgetMs` is undefined or > 800ms, and the strategy is
 *     vector or hybrid, mark `reranked = true`. Reranking is the single
 *     biggest quality lever after embeddings.
 *   - For keyword-only routes we skip reranking by default — keyword scores
 *     are already lexical, and rerankers are designed for vector recalls.
 *
 * Every decision records human-readable `reasons` so the dashboard and the
 * trace explorer can show *why* a route was picked. This is the
 * "observability + explainability" requirement made concrete.
 */
export class HeuristicRouter implements Router {
  readonly name = "heuristic-v1";

  decide(req: SearchRequest, caps: AdapterCapabilities): RoutingDecision {
    const signals = computeSignals(req.query);
    const reasons: string[] = [];

    // 1. Forced strategy.
    if (req.forceStrategy) {
      reasons.push(`forceStrategy=${req.forceStrategy}`);
      return finalize(req.forceStrategy, signals, reasons, caps, req);
    }

    // 2. Capability fallback — adapter only supports vector.
    if (!caps.keyword && !caps.hybrid) {
      reasons.push("adapter is vector-only");
      return finalize("vector", signals, reasons, caps, req);
    }

    let strategy: RetrievalStrategy;

    // 3. Phrase or high-specificity → keyword.
    if (signals.hasQuotedPhrase) {
      reasons.push("query contains quoted phrase");
      strategy = "keyword";
    } else if (signals.hasSpecificTokens && signals.tokens <= 6) {
      reasons.push("short query with specific identifiers/codes");
      strategy = "keyword";
    } else if (signals.tokens <= 2) {
      // Very short queries (single keyword) — keyword if available.
      reasons.push("very short query (≤2 tokens) → prefer keyword");
      strategy = caps.keyword ? "keyword" : "vector";
    } else if (signals.isQuestion && signals.tokens >= 5) {
      reasons.push("natural-language question → semantic search");
      strategy = "vector";
    } else if (signals.ambiguity > 0.6) {
      reasons.push("ambiguous query → semantic + rerank");
      strategy = "vector";
    } else {
      reasons.push("default → hybrid (no strong signal either way)");
      strategy = caps.hybrid || (caps.vector && caps.keyword) ? "hybrid" : "vector";
    }

    // 4. If we picked hybrid but adapter can't do hybrid AND lacks keyword,
    //    degrade gracefully. (BaseAdapter provides hybrid as RRF on top of
    //    vec+kw, so as long as both are supported, we're fine.)
    if (strategy === "hybrid" && !caps.hybrid && !(caps.vector && caps.keyword)) {
      reasons.push("hybrid not supported by adapter, falling back to vector");
      strategy = "vector";
    }
    if (strategy === "keyword" && !caps.keyword) {
      reasons.push("keyword not supported by adapter, falling back to vector");
      strategy = "vector";
    }

    return finalize(strategy, signals, reasons, caps, req);
  }
}

function finalize(
  strategy: RetrievalStrategy,
  signals: ReturnType<typeof computeSignals>,
  reasons: string[],
  _caps: AdapterCapabilities,
  req: SearchRequest
): RoutingDecision {
  const reranked = shouldRerank(strategy, req);
  if (reranked) reasons.push("reranking enabled (latency budget allows)");
  else if (strategy !== "keyword") reasons.push("reranking skipped (latency budget)");
  return { strategy, reasons, signals, reranked };
}

/**
 * Rerank decision — separate from strategy so it's easy to override later.
 *
 * Heuristic: we rerank when
 *   - the strategy is vector or hybrid (rerankers add little to keyword), AND
 *   - either no latency budget was supplied, or the budget exceeds 800ms.
 *
 * 800ms is empirical: a typical cross-encoder rerank over 50 candidates is
 * ~200-500ms on commodity infra. We want enough headroom for the upstream
 * caller's own work.
 */
function shouldRerank(strategy: RetrievalStrategy, req: SearchRequest): boolean {
  if (strategy === "keyword") return false;
  if (strategy === "rerank") return true;
  const budget = req.latencyBudgetMs;
  if (budget === undefined) return true;
  return budget > 800;
}
