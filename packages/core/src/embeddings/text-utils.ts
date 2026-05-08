/**
 * Text preprocessing utilities for keyword and TF-IDF retrieval.
 *
 * - `tokenize` (re-exported from embedder.ts) — basic whitespace + punctuation
 *   tokenizer, lowercased.
 * - `tokenizeAdvanced` — adds optional stopword filtering and Porter stemming.
 * - `stem` — Martin Porter's classic English stemmer (1980), implemented inline
 *   so the package stays dependency-free. The algorithm reduces inflectional
 *   forms to a common stem (running → run, connections → connect, agreed → agre).
 *   Not perfect — "going" stems to "go" but "wenting" stems to "went". Good
 *   enough as a retrieval-time normalizer.
 * - `STOPWORDS` — a standard English stopword set, ~120 words.
 *
 * Stemming + stopword removal typically yields +5-10% recall on BM25-style
 * keyword search at zero runtime cost (one-time precomputation per token).
 */

/**
 * English stopwords. Common closed-class words that carry no retrieval signal.
 * This is the standard "snowball english" list, widely used in IR systems.
 */
export const STOPWORDS: ReadonlySet<string> = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
  "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
  "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
  "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
  "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
  "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
  "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
  "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
  "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
  "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
  "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
  "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
  "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
  "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
  "they've", "this", "those", "through", "to", "too", "under", "until", "up",
  "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
  "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
  "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
  "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
  "yourself", "yourselves",
]);

/**
 * Tokenize + optional stopword removal + optional Porter stemming.
 *
 * Use this for BM25 keyword search. The basic `tokenize` from embedder.ts
 * stays the cheaper option when stopwords / stemming aren't needed.
 */
export function tokenizeAdvanced(
  text: string,
  opts: { stem?: boolean; dropStopwords?: boolean } = {}
): string[] {
  const { stem: doStem = false, dropStopwords = false } = opts;
  const raw = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/u)
    .filter((t) => t.length > 0);

  let out: string[] = raw;
  if (dropStopwords) out = out.filter((t) => !STOPWORDS.has(t));
  if (doStem) out = out.map(stem);
  return out;
}

// ---------- Porter stemmer (1980) ----------
//
// Compact implementation. The algorithm has 5 steps that successively strip
// suffixes based on the "measure" of the word's stem (count of vowel-consonant
// alternations). See Porter, M.F. "An Algorithm for Suffix Stripping" (1980).

const VOWELS = new Set(["a", "e", "i", "o", "u"]);

function isConsonant(s: string, i: number): boolean {
  const c = s[i];
  if (!c) return false;
  if (VOWELS.has(c)) return false;
  if (c === "y") return i === 0 ? true : !isConsonant(s, i - 1);
  return true;
}

/** Measure (m): number of (vowel-runs followed by consonant-runs) in a stem. */
function measure(stem: string): number {
  let m = 0;
  let i = 0;
  // Skip leading consonants.
  while (i < stem.length && isConsonant(stem, i)) i++;
  while (i < stem.length) {
    // Skip vowels.
    while (i < stem.length && !isConsonant(stem, i)) i++;
    if (i >= stem.length) break;
    m++;
    // Skip consonants.
    while (i < stem.length && isConsonant(stem, i)) i++;
  }
  return m;
}

function hasVowel(stem: string): boolean {
  for (let i = 0; i < stem.length; i++) {
    if (!isConsonant(stem, i)) return true;
  }
  return false;
}

function endsCVC(stem: string): boolean {
  if (stem.length < 3) return false;
  const n = stem.length;
  if (
    !isConsonant(stem, n - 3) ||
    isConsonant(stem, n - 2) ||
    !isConsonant(stem, n - 1)
  ) {
    return false;
  }
  // Final consonant cannot be w, x, or y.
  const last = stem[n - 1]!;
  return last !== "w" && last !== "x" && last !== "y";
}

function endsDoubleConsonant(stem: string): boolean {
  const n = stem.length;
  if (n < 2) return false;
  return stem[n - 1] === stem[n - 2] && isConsonant(stem, n - 1);
}

function replaceSuffix(word: string, suffix: string, replacement: string): string {
  return word.slice(0, word.length - suffix.length) + replacement;
}

/** Porter stemmer for English. */
export function stem(word: string): string {
  if (word.length <= 2) return word;
  let w = word;

  // Step 1a
  if (w.endsWith("sses")) w = replaceSuffix(w, "sses", "ss");
  else if (w.endsWith("ies")) w = replaceSuffix(w, "ies", "i");
  else if (w.endsWith("ss")) {
    /* keep */
  } else if (w.endsWith("s")) w = w.slice(0, -1);

  // Step 1b
  let step1bRan = false;
  if (w.endsWith("eed")) {
    if (measure(w.slice(0, -3)) > 0) w = w.slice(0, -1);
  } else if (w.endsWith("ed")) {
    const stemPart = w.slice(0, -2);
    if (hasVowel(stemPart)) {
      w = stemPart;
      step1bRan = true;
    }
  } else if (w.endsWith("ing")) {
    const stemPart = w.slice(0, -3);
    if (hasVowel(stemPart)) {
      w = stemPart;
      step1bRan = true;
    }
  }
  if (step1bRan) {
    if (w.endsWith("at") || w.endsWith("bl") || w.endsWith("iz")) w = w + "e";
    else if (endsDoubleConsonant(w)) {
      const last = w[w.length - 1]!;
      if (last !== "l" && last !== "s" && last !== "z") {
        w = w.slice(0, -1);
      }
    } else if (measure(w) === 1 && endsCVC(w)) {
      w = w + "e";
    }
  }

  // Step 1c
  if (w.endsWith("y")) {
    const stemPart = w.slice(0, -1);
    if (hasVowel(stemPart)) w = stemPart + "i";
  }

  // Step 2 — m>0 suffix substitutions
  const step2: Array<[string, string]> = [
    ["ational", "ate"], ["tional", "tion"], ["enci", "ence"], ["anci", "ance"],
    ["izer", "ize"], ["abli", "able"], ["alli", "al"], ["entli", "ent"],
    ["eli", "e"], ["ousli", "ous"], ["ization", "ize"], ["ation", "ate"],
    ["ator", "ate"], ["alism", "al"], ["iveness", "ive"], ["fulness", "ful"],
    ["ousness", "ous"], ["aliti", "al"], ["iviti", "ive"], ["biliti", "ble"],
  ];
  for (const [suf, rep] of step2) {
    if (w.endsWith(suf)) {
      const candidate = replaceSuffix(w, suf, rep);
      if (measure(candidate.slice(0, candidate.length - rep.length)) > 0) {
        w = candidate;
      }
      break;
    }
  }

  // Step 3 — m>0
  const step3: Array<[string, string]> = [
    ["icate", "ic"], ["ative", ""], ["alize", "al"], ["iciti", "ic"],
    ["ical", "ic"], ["ful", ""], ["ness", ""],
  ];
  for (const [suf, rep] of step3) {
    if (w.endsWith(suf)) {
      const candidate = replaceSuffix(w, suf, rep);
      if (measure(candidate.slice(0, candidate.length - rep.length)) > 0) {
        w = candidate;
      }
      break;
    }
  }

  // Step 4 — m>1, drop suffix entirely. Note: Porter's "ION" rule fires only
  // when preceded by S or T (so "nation" → "nat" → kept; "connection" →
  // "connect" → dropped). The other suffixes drop unconditionally at m>1.
  const step4 = [
    "ement", "ation", "ance", "ence", "able", "ible", "ment",
    "ant", "ent", "ism", "ate", "iti", "ous", "ive", "ize",
    "al", "er", "ic", "ou",
  ];
  let step4Matched = false;
  for (const suf of step4) {
    if (w.endsWith(suf)) {
      const candidate = w.slice(0, -suf.length);
      if (measure(candidate) > 1) {
        w = candidate;
      }
      step4Matched = true;
      break;
    }
  }
  // "ion" — only when preceded by 's' or 't'.
  if (!step4Matched && w.endsWith("ion")) {
    const candidate = w.slice(0, -3);
    const last = candidate[candidate.length - 1];
    if ((last === "s" || last === "t") && measure(candidate) > 1) {
      w = candidate;
    }
  }

  // Step 5a — final e
  if (w.endsWith("e")) {
    const candidate = w.slice(0, -1);
    const m = measure(candidate);
    if (m > 1 || (m === 1 && !endsCVC(candidate))) w = candidate;
  }
  // Step 5b — final l after double l, m>1
  if (
    measure(w) > 1 &&
    endsDoubleConsonant(w) &&
    w.endsWith("l")
  ) {
    w = w.slice(0, -1);
  }

  return w;
}
