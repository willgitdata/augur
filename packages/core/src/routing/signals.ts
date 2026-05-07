import type { QuerySignals, QuestionType } from "../types.js";

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
 * - isQuestion / questionType: "how do I X?" is semantic; "who/when/where" is
 *   factoid and entity-leaning.
 * - hasNamedEntity: capitalized proper nouns outside the first token reliably
 *   indicate "user is searching for a specific named thing" — keyword wins.
 * - hasCodeLike: camelCase/dotted/snake_case is strong keyword signal.
 * - hasDateOrVersion: years, semver, RFC, CVE — specific terms users want literal.
 * - hasNegation: "not", "without", "vs" — bi-encoders fail on negation, so we
 *   force the reranker on regardless of base strategy.
 * - language: non-Latin script → BM25 won't help (English-tuned), prefer vector.
 * - ambiguity: low lexical signal + many short stopwords = vector or rerank.
 *
 * All thresholds are intentionally explicit constants here, not config —
 * they're easier to reason about, and we tune them with eval datasets, not
 * by twiddling a YAML file.
 */

const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on",
  "at", "for", "with", "is", "are", "was", "were", "be", "been", "being",
  "this", "that", "these", "those", "it", "its", "i", "you", "we", "they",
]);

const NEGATION_TOKENS = new Set([
  "not", "no", "without", "except", "vs", "never", "neither", "nor",
]);

/** Strip leading punctuation/quotes for case-detection on the original token. */
function stripPunct(token: string): string {
  return token.replace(/^["'(\[`]+|["')\].,?!:`]+$/g, "");
}

export function computeSignals(query: string): QuerySignals {
  const trimmed = query.trim();
  const tokens = trimmed.split(/\s+/).filter(Boolean);
  const tokensLower = tokens.map((t) => t.toLowerCase());

  const tokenCount = tokens.length;
  const avgTokenLen = tokenCount === 0 ? 0 : tokens.reduce((s, t) => s + t.length, 0) / tokenCount;
  const hasQuotedPhrase = /["'][^"']{2,}["']/.test(trimmed);

  const hasSpecificTokens = tokens.some((t) => {
    if (/^\d+$/.test(t)) return true;
    if (/^[A-Z0-9]{3,}$/.test(t)) return true;
    if (/[_\-./]/.test(t) && /[a-zA-Z]/.test(t) && /\d/.test(t)) return true;
    if (/^0x[0-9a-fA-F]+$/.test(t)) return true;
    return false;
  });

  // Code-like patterns. Each matched token suggests a programming-context query.
  const hasCodeLike = tokens.some((t) => {
    const bare = stripPunct(t);
    if (/[a-z][A-Z]/.test(bare)) return true;                // camelCase: useState, kubectl
    if (/^[a-z]+(_[a-z0-9]+)+$/i.test(bare)) return true;    // snake_case: pool_mode, pg_repack
    if (/[a-z]\.[a-z]/i.test(bare) && bare.length >= 5) return true; // dotted: rbac.authorization.k8s.io
    if (bare.includes("::")) return true;                    // C++/Rust scope
    if (/[a-zA-Z_]\w*\(/.test(bare)) return true;            // function call: foo(
    if (bare.includes("=") && /[a-zA-Z]/.test(bare) && bare.length >= 3) return true; // assignment: pool_mode=transaction
    return false;
  });

  // Date / semver / RFC / CVE detection — specific identifiers users want literal.
  const hasDateOrVersion =
    /\b(19|20|21)\d{2}\b/.test(trimmed) ||                   // year
    /\bv?\d+\.\d+(\.\d+)?\b/.test(trimmed) ||                // semver: v3.1.2 or 1.3
    /\bRFC\s?\d{3,5}\b/i.test(trimmed) ||                    // RFC 7519
    /\bCVE-\d{4}-\d{4,7}\b/i.test(trimmed);                  // CVE-2024-3094

  // Question detection (preserved) + refined taxonomy.
  const isQuestion =
    /[?]\s*$/.test(trimmed) ||
    /^(how|why|what|when|where|who|which|can|does|do|is|are|should|would|could)\b/i.test(trimmed);

  let questionType: QuestionType | null = null;
  if (isQuestion) {
    const head = trimmed.toLowerCase();
    if (/^(who|when|where|which)\b/.test(head)) questionType = "factoid";
    else if (/^(how|why)\b/.test(head)) questionType = "procedural";
    else if (/^what\s+(is|are|does|do|was|were)\b/.test(head)) questionType = "definitional";
    // "what version", "can", "should", etc. → leave as untyped question.
  }

  // Named entity heuristic: capitalized non-stopword token NOT at position 0.
  // Sentence-initial capitalization is too noisy (every "How do I..." starts
  // capital). Mid-query caps are a much stronger entity signal.
  const hasNamedEntity = tokens.slice(1).some((t) => {
    const bare = stripPunct(t);
    if (bare.length < 2) return false;
    if (!/^[A-Z]/.test(bare)) return false;
    if (STOPWORDS.has(bare.toLowerCase())) return false;
    // Require at least one lowercase letter — pure ALL-CAPS is already
    // covered by hasSpecificTokens. We want PgBouncer, Redis, TLS-style
    // capitalized words.
    if (!/[a-z]/.test(bare) && bare.length < 4) return false;
    return true;
  });

  const hasNegation = tokensLower.some((t) => NEGATION_TOKENS.has(t));

  // Language detection (heuristic): non-Latin chars (CJK, Arabic, Cyrillic, etc).
  // Spanish/French still pass as "en" since they're Latin script and BM25 +
  // a Latin-tokenizer handle them passably; CJK is the hard case for BM25.
  const nonLatinCount = (trimmed.match(/[^\u0020-\u024F\s\p{P}]/gu) ?? []).length;
  const totalNonSpace = trimmed.replace(/\s+/g, "").length;
  const language: "en" | "non-en" =
    totalNonSpace > 0 && nonLatinCount / totalNonSpace > 0.3 ? "non-en" : "en";

  // Ambiguity proxy: high stopword ratio + low lexical specificity.
  const stopRatio =
    tokenCount === 0 ? 0 : tokensLower.filter((t) => STOPWORDS.has(t)).length / tokenCount;
  const lengthPenalty = tokenCount < 3 ? 0.4 : 0;
  const specificityReward = hasSpecificTokens || hasCodeLike || hasDateOrVersion ? -0.3 : 0;
  let ambiguity = stopRatio + lengthPenalty + specificityReward;
  ambiguity = Math.max(0, Math.min(1, ambiguity));

  return {
    tokens: tokenCount,
    avgTokenLen,
    hasQuotedPhrase,
    hasSpecificTokens,
    isQuestion,
    ambiguity,
    hasNamedEntity,
    hasCodeLike,
    hasDateOrVersion,
    questionType,
    hasNegation,
    language,
  };
}
