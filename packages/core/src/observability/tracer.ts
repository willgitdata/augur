import { randomUUID } from "node:crypto";
import type { SearchTrace, TraceSpan } from "../types.js";

/**
 * Tracer — the observability primitive.
 *
 * Why we built our own instead of using OpenTelemetry:
 * - The trace is a first-class API output, not a side effect — every search
 *   returns its trace in the response. OTel is fire-and-forget by design.
 * - It's tiny, has no network/IO, and doesn't depend on a collector.
 * - Users who want OTel export can wrap this tracer in ~30 lines.
 */
export class Tracer {
  private query: string;
  private startedAt: string;
  private startMs: number;
  private spans: TraceSpan[] = [];

  constructor(query: string) {
    this.query = query;
    this.startedAt = new Date().toISOString();
    this.startMs = performance.now();
  }

  /** Time a synchronous or async operation and record it as a span. */
  async span<T>(
    name: string,
    fn: () => Promise<T> | T,
    attributes?: Record<string, unknown>
  ): Promise<T> {
    const start = performance.now();
    try {
      const result = await fn();
      this.recordSpan(name, start, attributes);
      return result;
    } catch (e) {
      this.recordSpan(name, start, { ...attributes, error: String(e) });
      throw e;
    }
  }

  /** Manually record a span without wrapping a fn. Useful when timing is external. */
  recordSpan(name: string, startMs: number, attributes?: Record<string, unknown>): void {
    const endMs = performance.now();
    this.spans.push({
      name,
      startMs: startMs - this.startMs,
      endMs: endMs - this.startMs,
      durationMs: endMs - startMs,
      attributes,
    });
  }

  /** Build the final trace. The orchestrator fills in the routing decision. */
  finish(
    opts: Omit<SearchTrace, "id" | "query" | "startedAt" | "totalMs" | "spans">
  ): SearchTrace {
    return {
      id: randomUUID(),
      query: this.query,
      startedAt: this.startedAt,
      totalMs: performance.now() - this.startMs,
      spans: this.spans,
      ...opts,
    };
  }
}

/**
 * In-memory trace store — bounded ring buffer. The HTTP server exposes
 * its contents via the `/traces` endpoint for trace explorers and
 * observability backends.
 *
 * Why a ring buffer instead of `Array.shift()`:
 * - `push()` is the hot path — every search call writes one trace.
 * - `Array.shift()` on overflow is O(n), so a busy server hits O(n²)
 *   amortized once the buffer is full (n = capacity, default 1000).
 * - A fixed-size circular buffer makes both `push()` and capacity
 *   eviction O(1), at the cost of a slightly more involved `list()`.
 */
export class TraceStore {
  private capacity: number;
  /** Fixed-size circular buffer. Sized exactly once, never re-allocated. */
  private buffer: (SearchTrace | undefined)[];
  /** Next write position. Wraps modulo capacity. */
  private writeIndex = 0;
  /** Number of valid entries in the buffer (≤ capacity). */
  private count = 0;

  constructor(capacity = 1000) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error("TraceStore: capacity must be a positive integer");
    }
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }

  push(trace: SearchTrace): void {
    this.buffer[this.writeIndex] = trace;
    this.writeIndex = (this.writeIndex + 1) % this.capacity;
    if (this.count < this.capacity) this.count += 1;
    // When at capacity, the previous occupant at writeIndex is overwritten
    // — that's the eviction. No shift, no slice, no realloc.
  }

  /**
   * Most-recent-first. The semantics match the previous implementation:
   * `list(100)` returns up to 100 traces ordered newest → oldest.
   */
  list(limit = 100): SearchTrace[] {
    const n = Math.min(limit, this.count);
    const out: SearchTrace[] = new Array(n);
    // Walk backwards from the most recently written slot. writeIndex
    // points at the *next* write slot, so the last valid entry is at
    // `(writeIndex - 1 + capacity) % capacity`.
    let idx = (this.writeIndex - 1 + this.capacity) % this.capacity;
    for (let i = 0; i < n; i++) {
      out[i] = this.buffer[idx]!;
      idx = (idx - 1 + this.capacity) % this.capacity;
    }
    return out;
  }

  get(id: string): SearchTrace | undefined {
    // Walk the buffer once. O(capacity) which is fine — `/traces/:id` is
    // a low-frequency UI lookup, not a hot path.
    for (let i = 0; i < this.count; i++) {
      const t = this.buffer[i];
      if (t && t.id === id) return t;
    }
    return undefined;
  }

  clear(): void {
    this.buffer = new Array(this.capacity);
    this.writeIndex = 0;
    this.count = 0;
  }

  size(): number {
    return this.count;
  }
}
