import type { AdapterCapabilities } from "../adapters/adapter.js";
import type {
  RoutingDecision,
  RetrievalStrategy,
  SearchRequest,
  QuerySignals,
} from "../types.js";
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
 * HeuristicRouter — rule-based per-query strategy selection. See the
 * inline comments in `decide()` for the full priority order. Every
 * decision records human-readable `reasons` so the dashboard and trace
 * explorer can show *why* a route was picked.
 */
export interface HeuristicRouterOptions {
  /**
   * Push every strategy (including pure keyword) through the multi-stage
   * gather → fuse → rerank pipeline. **Default: true** — auto routing
   * targets best NDCG, and on the bundled eval rerank-everything is
   * +0.008 NDCG@10 over the BM25-fast-path mode. Set to `false` only
   * when you have a strict <50ms latency budget and your queries are
   * mostly lexical; then keyword-routed queries skip the cross-encoder.
   */
  alwaysRerank?: boolean;
}

export class HeuristicRouter implements Router {
  readonly name: string;
  private alwaysRerank: boolean;

  constructor(opts: HeuristicRouterOptions = {}) {
    this.alwaysRerank = opts.alwaysRerank ?? true;
    this.name = this.alwaysRerank ? "heuristic-v1" : "heuristic-v1+fast-keyword";
  }

  decide(req: SearchRequest, caps: AdapterCapabilities): RoutingDecision {
    const signals = computeSignals(req.query);
    const reasons: string[] = [];

    // 1. Forced strategy.
    if (req.forceStrategy) {
      reasons.push(`forceStrategy=${req.forceStrategy}`);
      return finalize(req.forceStrategy, signals, reasons, req, this.alwaysRerank);
    }

    // 2. Capability fallback — adapter only supports vector.
    if (!caps.keyword && !caps.hybrid) {
      reasons.push("adapter is vector-only");
      return finalize("vector", signals, reasons, req, this.alwaysRerank);
    }

    // 3. Non-English → vector (English-tuned BM25 fails on CJK and similar).
    if (signals.language === "non-en") {
      reasons.push("non-English query → semantic search");
      return finalize("vector", signals, reasons, req, this.alwaysRerank);
    }

    let strategy: RetrievalStrategy;

    // 4. Quoted phrase → keyword.
    if (signals.hasQuotedPhrase) {
      reasons.push("query contains quoted phrase");
      strategy = "keyword";
    }
    // 5a. Code-like syntax → keyword. camelCase / snake_case / function calls
    // are unambiguously lexical — exact tokens, no synonyms to chase.
    else if (signals.hasCodeLike && signals.tokens <= 6) {
      reasons.push("short query with code-like syntax → keyword");
      strategy = "keyword";
    }
    // 5b. Bare identifier / numeric / date-version (no code syntax) → hybrid.
    // Eval traced cases like "503 Service Unavailable" → rfc-9110 where the
    // target doc was topical (HTTP semantics) and didn't mention the literal
    // tokens. Pure keyword can't reach those; vector + BM25 (RRF) does.
    else if (
      (signals.hasSpecificTokens || signals.hasDateOrVersion) &&
      signals.tokens <= 6
    ) {
      const why = signals.hasDateOrVersion
        ? "date/version/RFC token"
        : "specific identifiers without code syntax";
      reasons.push(`short query with ${why} → hybrid (keyword precision + vector recall)`);
      strategy = caps.hybrid || (caps.vector && caps.keyword) ? "hybrid" : "keyword";
    }
    // 6. Very short queries → keyword.
    else if (signals.tokens <= 2) {
      reasons.push("very short query (≤2 tokens) → prefer keyword");
      strategy = caps.keyword ? "keyword" : "vector";
    }
    // 7. Mid-query named entity, not a question → keyword.
    else if (signals.hasNamedEntity && signals.tokens <= 6 && !signals.isQuestion) {
      reasons.push("named entity detected → keyword");
      strategy = "keyword";
    }
    // 8. Question taxonomy.
    else if (signals.questionType === "procedural" && signals.tokens >= 5) {
      reasons.push("procedural question (how/why) → semantic search");
      strategy = "vector";
    } else if (signals.questionType === "definitional" && signals.tokens >= 3) {
      reasons.push("definitional question (what is X) → semantic search");
      strategy = "vector";
    } else if (signals.questionType === "factoid" && signals.tokens >= 3) {
      reasons.push("factoid question (who/when/where/which) → hybrid");
      strategy = "hybrid";
    } else if (signals.isQuestion && signals.tokens >= 5) {
      reasons.push("natural-language question → semantic search");
      strategy = "vector";
    }
    // 9. High ambiguity → vector + rerank.
    else if (signals.ambiguity > 0.6) {
      reasons.push("ambiguous query → semantic + rerank");
      strategy = "vector";
    }
    // 10. Default → hybrid.
    else {
      reasons.push("default → hybrid (no strong signal either way)");
      strategy = caps.hybrid || (caps.vector && caps.keyword) ? "hybrid" : "vector";
    }

    // Graceful degradation if the picked strategy isn't supported by the adapter.
    if (strategy === "hybrid" && !caps.hybrid && !(caps.vector && caps.keyword)) {
      reasons.push("hybrid not supported by adapter, falling back to vector");
      strategy = "vector";
    }
    if (strategy === "keyword" && !caps.keyword) {
      reasons.push("keyword not supported by adapter, falling back to vector");
      strategy = "vector";
    }

    return finalize(strategy, signals, reasons, req, this.alwaysRerank);
  }
}

function finalize(
  strategy: RetrievalStrategy,
  signals: QuerySignals,
  reasons: string[],
  req: SearchRequest,
  alwaysRerank: boolean
): RoutingDecision {
  const reranked = shouldRerank(strategy, signals, req, alwaysRerank);
  if (reranked) {
    if (alwaysRerank && strategy === "keyword") {
      reasons.push("alwaysRerank=true → multi-stage rerank on keyword strategy");
    } else if (signals.hasNegation && strategy !== "rerank") {
      reasons.push("negation detected → reranking forced");
    } else {
      reasons.push("reranking enabled (latency budget allows)");
    }
  } else if (strategy !== "keyword") {
    reasons.push("reranking skipped (latency budget)");
  }
  return { strategy, reasons, signals, reranked };
}

/**
 * Rerank decision. Default mode reranks every strategy (including pure
 * keyword) for max NDCG; the cross-encoder gets to vote on every query.
 * The latency-sensitive opt-out (`alwaysRerank: false`) restores the BM25
 * fast path on keyword-routed queries.
 *
 * In every mode: a hard `latencyBudgetMs < 800` skips reranking — no
 * point breaching the budget — and `forceStrategy: "rerank"` always
 * reranks regardless.
 */
function shouldRerank(
  strategy: RetrievalStrategy,
  signals: QuerySignals,
  req: SearchRequest,
  alwaysRerank: boolean
): boolean {
  if (strategy === "rerank") return true;
  const budget = req.latencyBudgetMs;
  const budgetOK = budget === undefined || budget > 800;
  if (alwaysRerank) return budgetOK;
  if (signals.hasNegation) return true;
  if (strategy === "keyword") return false;
  return budgetOK;
}
