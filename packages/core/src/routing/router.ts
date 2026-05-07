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
 * HeuristicRouter — rule-based routing, the MVP default.
 *
 * The decision flow, in plain English:
 *
 *   1. forceStrategy → use it verbatim.
 *   2. Capability fallback — adapter only supports vector → vector.
 *   3. Non-English query → vector. Our default analyzer is English-tuned;
 *      BM25 is unhelpful on CJK / Cyrillic / Arabic etc.
 *   4. Quoted phrase → keyword. Exact phrase match is a hard constraint.
 *   5. Specific tokens / code-like / date-version AND short query (≤6) → keyword.
 *      Identifiers, error codes, semver, RFC numbers want literal match.
 *   6. Very short queries (≤2 tokens) → keyword (or vector if adapter lacks).
 *   7. Mid-query named entity AND not a question → keyword. "PgBouncer config
 *      OLTP" type queries are looking for a specific named thing.
 *   8. Question taxonomy:
 *      - procedural (how/why) → vector. Semantic match wins.
 *      - definitional (what is X) → vector. Synonym tolerance matters.
 *      - factoid (who/when/where/which) → hybrid. Entity match + semantics.
 *      - other questions ≥5 tokens → vector (preserve original behavior).
 *   9. High ambiguity → vector + rerank.
 *  10. Otherwise → hybrid. The default for "I don't know" is do both —
 *      hybrid retrieval is rarely worse than either alone, and RRF makes it
 *      cheap to combine.
 *
 * Reranking is layered on top:
 *   - If `forceStrategy === "rerank"` → always rerank.
 *   - If signals.hasNegation → always rerank. Bi-encoders famously fail on
 *     negation ("X without Y" embeds similarly to "X with Y"); the reranker
 *     actually reads the words "not"/"without".
 *   - For keyword strategies otherwise, skip rerank (scores already lexical).
 *   - For vector / hybrid, rerank when no budget OR budget > 800ms (empirical
 *     floor: a cross-encoder over 50 candidates is ~200-500ms on commodity infra).
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

    // 3. Non-English → vector (English-tuned BM25 fails on CJK and similar).
    if (signals.language === "non-en") {
      reasons.push("non-English query → semantic search");
      return finalize("vector", signals, reasons, caps, req);
    }

    let strategy: RetrievalStrategy;

    // 4. Quoted phrase → keyword.
    if (signals.hasQuotedPhrase) {
      reasons.push("query contains quoted phrase");
      strategy = "keyword";
    }
    // 5. Specific / code-like / dated short query → keyword.
    else if (
      (signals.hasSpecificTokens || signals.hasCodeLike || signals.hasDateOrVersion) &&
      signals.tokens <= 6
    ) {
      const why = signals.hasDateOrVersion
        ? "date/version/RFC token"
        : signals.hasCodeLike
          ? "code-like syntax"
          : "specific identifiers/codes";
      reasons.push(`short query with ${why} → keyword`);
      strategy = "keyword";
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

    return finalize(strategy, signals, reasons, caps, req);
  }
}

function finalize(
  strategy: RetrievalStrategy,
  signals: QuerySignals,
  reasons: string[],
  _caps: AdapterCapabilities,
  req: SearchRequest
): RoutingDecision {
  const reranked = shouldRerank(strategy, signals, req);
  if (reranked) {
    if (signals.hasNegation && strategy !== "rerank") {
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
 * Rerank decision — separate from strategy so it's easy to override later.
 *
 * Heuristic:
 *   - Forced "rerank" strategy → always rerank.
 *   - Negation present → always rerank. Bi-encoders fail on negation; the
 *     reranker reads the negation token. Even keyword retrieval benefits.
 *   - Keyword strategies otherwise skip rerank (scores already lexical).
 *   - Vector / hybrid: rerank when no budget OR budget > 800ms.
 */
function shouldRerank(
  strategy: RetrievalStrategy,
  signals: QuerySignals,
  req: SearchRequest
): boolean {
  if (strategy === "rerank") return true;
  if (signals.hasNegation) return true;
  if (strategy === "keyword") return false;
  const budget = req.latencyBudgetMs;
  if (budget === undefined) return true;
  return budget > 800;
}
