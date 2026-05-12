/**
 * Bounded LRU cache backed by a JS Map.
 *
 * Why this exists:
 * - Several module-level caches in this package hold heavyweight objects
 *   (ONNX pipelines at 22-280 MB each, context-cache strings keyed by
 *   user document content). Unbounded `Map` growth has bitten us in
 *   long-running indexers — see the code-review notes for the specific
 *   incidents.
 * - JS `Map` preserves insertion order, so a tiny LRU is just a delete-
 *   then-set on `get` to move the entry to the most-recent position,
 *   plus a `keys().next().value` to evict the oldest on overflow.
 * - No third-party dep. The whole class is ~40 lines.
 *
 * Eviction semantics: oldest entry by *access* time (not insertion). A
 * cache hit moves the entry to the most-recently-used slot. Eviction
 * runs after every `set` that takes the size past capacity.
 *
 * Memory note: evicting an entry only drops the cache's reference. If
 * the caller has the value live elsewhere (e.g. an in-flight Promise),
 * GC won't reclaim the underlying object. That's correct behavior —
 * the LRU bounds the *cache*, not the application's working set.
 */
export class BoundedCache<K, V> {
  private store = new Map<K, V>();
  private capacity: number;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error("BoundedCache: capacity must be a positive integer");
    }
    this.capacity = capacity;
  }

  get(key: K): V | undefined {
    const v = this.store.get(key);
    if (v === undefined) return undefined;
    // Touch: move to MRU slot.
    this.store.delete(key);
    this.store.set(key, v);
    return v;
  }

  set(key: K, value: V): void {
    if (this.store.has(key)) this.store.delete(key);
    this.store.set(key, value);
    while (this.store.size > this.capacity) {
      const oldest = this.store.keys().next().value;
      if (oldest === undefined) break;
      this.store.delete(oldest);
    }
  }

  has(key: K): boolean {
    return this.store.has(key);
  }

  delete(key: K): boolean {
    return this.store.delete(key);
  }

  size(): number {
    return this.store.size;
  }

  clear(): void {
    this.store.clear();
  }
}
