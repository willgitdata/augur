import type { QuerySignals, QuestionType } from "../types.js";

/**
 * Compute query signals — pure function from query string to features.
 *
 * These features power the heuristic router. They're deliberately cheap
 * (O(n) over the query) so they add no measurable latency. When we add ML
 * routing later, these become the input feature vector.
 *
 * Why these specific signals:
 * - wordCount / avgWordLen: short queries with long words (e.g. UUIDs, error
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
 * - language: BCP-47-style code from Unicode-script analysis ("en" by default,
 *   "ja"/"zh"/"ko"/"ru"/"ar"/"hi"/"th"/"he"/"el" for non-Latin scripts). Drives
 *   both the router (non-en → vector) and the language-aware filter Augur
 *   applies at search time.
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

/**
 * Strip leading + trailing punctuation/quotes for case-detection on the
 * original token. Two single-anchor passes with bounded `{1,64}`
 * quantifiers — CodeQL flags any unbounded `+` over a character class
 * as `js/polynomial-redos` regardless of anchoring, so the bounded
 * form closes the alert. 64 is well above any realistic quote /
 * punctuation run; longer prefixes get a no-op trim and the body
 * still matches downstream regexes.
 */
function stripPunct(token: string): string {
  return token
    .replace(/^["'(\[`]{1,64}/, "")
    .replace(/["')\].,?!:`]{1,64}$/, "");
}

/**
 * Detect a query/document language code from Unicode script ranges.
 *
 * Returns one of: "en" (Latin script — default), "ja", "zh", "ko", "ru",
 * "ar", "hi", "th", "he", "el". Latin-script European languages (Spanish,
 * French, German, Italian, Portuguese, Vietnamese) collapse to "en"
 * because BM25 with a Latin tokenizer handles them passably and we'd
 * rather not aggressively partition the corpus on weak signal.
 *
 * Heuristic by design — no n-gram model, no dependencies. Trades the
 * ability to distinguish Spanish from English for predictable, sub-µs
 * decisions and zero install footprint.
 */
export function detectLanguage(text: string): string {
  if (!text) return "en";
  // Hiragana / Katakana → Japanese (must come before generic CJK
  // because Japanese also contains Han ideographs).
  if (/[぀-ゟ゠-ヿ]/.test(text)) return "ja";
  // Hangul Syllables / Jamo → Korean
  if (/[가-힯ᄀ-ᇿ]/.test(text)) return "ko";
  // Han ideographs without hiragana/hangul → Chinese
  if (/[一-鿿]/.test(text)) return "zh";
  // Cyrillic → ru (simplification; Ukrainian / Bulgarian / Serbian
  // also use Cyrillic but for retrieval-routing purposes they cluster).
  if (/[Ѐ-ӿ]/.test(text)) return "ru";
  // Arabic
  if (/[؀-ۿݐ-ݿ]/.test(text)) return "ar";
  // Devanagari → Hindi
  if (/[ऀ-ॿ]/.test(text)) return "hi";
  // Thai
  if (/[฀-๿]/.test(text)) return "th";
  // Hebrew
  if (/[֐-׿]/.test(text)) return "he";
  // Greek
  if (/[Ͱ-Ͽ]/.test(text)) return "el";
  return "en";
}

export function computeSignals(query: string): QuerySignals {
  const trimmed = query.trim();
  // Whitespace-split words. These drive routing rules; the embedder has its
  // own subword tokenization (WordPiece for MiniLM-L6 etc) and is unrelated.
  const words = trimmed.split(/\s+/).filter(Boolean);
  const wordsLower = words.map((t) => t.toLowerCase());

  const wordCount = words.length;
  const avgWordLen = wordCount === 0 ? 0 : words.reduce((s, t) => s + t.length, 0) / wordCount;
  const hasQuotedPhrase = /["'][^"']{2,}["']/.test(trimmed);

  const hasSpecificTokens = words.some((t) => {
    if (/^\d+$/.test(t)) return true;
    if (/^[A-Z0-9]{3,}$/.test(t)) return true;
    if (/[_\-./]/.test(t) && /[a-zA-Z]/.test(t) && /\d/.test(t)) return true;
    if (/^0x[0-9a-fA-F]+$/.test(t)) return true;
    return false;
  });

  // Code-like patterns. Each matched token suggests a programming-context query.
  const hasCodeLike = words.some((t) => {
    const bare = stripPunct(t);
    if (/[a-z][A-Z]/.test(bare)) return true;                // camelCase: useState, kubectl
    // snake_case: pool_mode, pg_repack. Two checks rather than a single
    // ambiguous regex (`^[a-z]+(_[a-z0-9]+)+$`) — that pattern's two
    // overlapping greedy quantifiers trip CodeQL's polynomial-redos.
    if (bare.includes("_") && /^[a-z][a-z0-9_]{1,64}$/i.test(bare)) return true;
    if (/[a-z]\.[a-z]/i.test(bare) && bare.length >= 5) return true; // dotted: rbac.authorization.k8s.io
    if (bare.includes("::")) return true;                    // C++/Rust scope
    // function call: foo(. Anchor at start + bound the wildcard so
    // CodeQL's polynomial-redos analysis sees a finite worst case.
    if (bare.includes("(") && /^[a-zA-Z_]\w{0,128}\(/.test(bare)) return true;
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
  const hasNamedEntity = words.slice(1).some((t) => {
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

  const hasNegation = wordsLower.some((t) => NEGATION_TOKENS.has(t));

  const language = detectLanguage(trimmed);

  // Ambiguity proxy: high stopword ratio + low lexical specificity.
  const stopRatio =
    wordCount === 0 ? 0 : wordsLower.filter((t) => STOPWORDS.has(t)).length / wordCount;
  const lengthPenalty = wordCount < 3 ? 0.4 : 0;
  const specificityReward = hasSpecificTokens || hasCodeLike || hasDateOrVersion ? -0.3 : 0;
  let ambiguity = stopRatio + lengthPenalty + specificityReward;
  ambiguity = Math.max(0, Math.min(1, ambiguity));

  return {
    wordCount,
    avgWordLen,
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
