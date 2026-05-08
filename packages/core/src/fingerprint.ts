/**
 * Document-set fingerprinting for the ad-hoc scratch-adapter cache.
 *
 * Used as the cache key for the InMemoryAdapter that's built from
 * `SearchRequest.documents`. Properties we care about:
 *
 *   - Same docs in same order → same key (cache hit).
 *   - Reordering changes the key (different docs are still different
 *     even if you flipped them — cheaper than canonicalizing order, and
 *     callers who really want order-independence can sort before passing).
 *   - One byte change → different key (no false cache hits on edits).
 *   - Doesn't allocate the full corpus content as a string.
 *   - Suffix with doc count to differentiate prefix-equal corpora.
 *
 * Implementation: FNV-1a 32-bit rolling hash over (id ‖ content) bytes,
 * using `Math.imul` for a fast 32-bit multiply. NOT cryptographic;
 * collision risk is negligible at typical cache sizes (Map<string, …>
 * with size ≤ 8). If you need cryptographic guarantees, layer SHA-256
 * on top — but for a 16-bucket LRU keyed by content the user just sent
 * us, FNV-1a is the right level.
 *
 * Encoding note: hashes per-UTF16-code-unit via `charCodeAt`. ASCII and
 * BMP content collide as expected; supplementary-plane code points each
 * contribute two surrogate halves. That's fine for fingerprint use —
 * different content produces different surrogate sequences, so the
 * collision properties hold.
 */
import type { Document } from "./types.js";

/**
 * Deterministic content fingerprint of a `Document[]`. Same input →
 * same output. Different input → different output (FNV-1a collision
 * floor; effectively zero for our cache sizes).
 */
export function fingerprintDocs(docs: ReadonlyArray<Document>): string {
  let h = 0x811c9dc5; // FNV-1a 32-bit offset basis
  for (const d of docs) {
    for (let i = 0; i < d.id.length; i++) {
      h ^= d.id.charCodeAt(i);
      h = Math.imul(h, 0x01000193); // FNV-1a 32-bit prime
    }
    // Field separator between id and content so concatenation
    // doesn't false-collide ({id:"ab",content:"c"} vs {id:"a",content:"bc"}).
    h ^= 0;
    h = Math.imul(h, 0x01000193);
    for (let i = 0; i < d.content.length; i++) {
      h ^= d.content.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    // Record separator between adjacent docs.
    h ^= 0xff;
    h = Math.imul(h, 0x01000193);
  }
  // Suffix with doc count: protects against a corpus that's a strict
  // prefix of another from colliding (the record separator handles the
  // adjacent-doc case but a count-mismatched prefix is still distinct).
  return `${docs.length}:${(h >>> 0).toString(36)}`;
}
