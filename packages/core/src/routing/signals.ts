import type { QuerySignals } from "../types.js";

/**
 * Compute query signals — pure function from query string to features.
 *
 * These features power the heuristic router. They're deliberately cheap
 * (O(n) over the query) so they add no measurable latency. When we add ML
 * routing later, these become the input feature vector.
 *
 * Why these specific signals:
 * - tokens / avgTokenLen: short queries with long tokens (e.g. UUIDs, error
 *   codes) are keyword-flavored; long natural-language queries are vector-flavored.
 * - hasQuotedPhrase: "exact phrase match" is a hard constraint — that's keyword.
 * - hasSpecificTokens: identifiers, numbers, hex codes — keyword wins.
 * - isQuestion: "how do I X?" is semantic — vector wins.
 * - ambiguity: low lexical signal + many short stopwords = vector or rerank.
 *
 * All thresholds are intentionally explicit constants here, not config —
 * they're easier to reason about, and we tune them with eval datasets, not
 * by twiddling a YAML file.
 */
export function computeSignals(query: string): QuerySignals {
  const trimmed = query.trim();
  const tokens = trimmed.split(/\s+/).filter(Boolean);
  const tokensLower = tokens.map((t) => t.toLowerCase());

  const tokenCount = tokens.length;
  const avgTokenLen = tokenCount === 0 ? 0 : tokens.reduce((s, t) => s + t.length, 0) / tokenCount;
  const hasQuotedPhrase = /["'][^"']{2,}["']/.test(trimmed);

  // "Specific" = numeric, identifier-like, code-like, or all-caps acronyms ≥ 3 chars.
  const hasSpecificTokens = tokens.some((t) => {
    if (/^\d+$/.test(t)) return true;
    if (/^[A-Z0-9]{3,}$/.test(t)) return true;
    if (/[_\-./]/.test(t) && /[a-zA-Z]/.test(t) && /\d/.test(t)) return true;
    if (/^0x[0-9a-fA-F]+$/.test(t)) return true;
    return false;
  });

  const isQuestion =
    /[?]\s*$/.test(trimmed) ||
    /^(how|why|what|when|where|who|which|can|does|do|is|are|should|would|could)\b/i.test(trimmed);

  // Ambiguity proxy: high stopword ratio + low lexical specificity → ambiguous.
  const stopwords = new Set([
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on",
    "at", "for", "with", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "i", "you", "we", "they",
  ]);
  const stopRatio =
    tokenCount === 0 ? 0 : tokensLower.filter((t) => stopwords.has(t)).length / tokenCount;
  // Penalize very short queries — they're inherently ambiguous.
  const lengthPenalty = tokenCount < 3 ? 0.4 : 0;
  // Reward specificity heavily.
  const specificityReward = hasSpecificTokens ? -0.3 : 0;
  let ambiguity = stopRatio + lengthPenalty + specificityReward;
  ambiguity = Math.max(0, Math.min(1, ambiguity));

  return {
    tokens: tokenCount,
    avgTokenLen,
    hasQuotedPhrase,
    hasSpecificTokens,
    isQuestion,
    ambiguity,
  };
}
