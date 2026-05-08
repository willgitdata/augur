import { randomUUID } from "node:crypto";
import type { SearchTrace, TraceSpan } from "../types.js";

/**
 * Tracer — the observability primitive.
 *
 * Why we built our own instead of using OpenTelemetry:
 * - The trace is a first-class API output, not a side effect — every search
 *   returns its trace in the response. OTel is fire-and-forget by design.
 * - It's tiny, has no network/IO, and doesn't depend on a collector.
 * - Users who want OTel export can wrap our tracer easily — see the
 *   "OpenTelemetry export" section in ARCHITECTURE.md.
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
  finish(opts: {
    decision: SearchTrace["decision"];
    candidates: number;
    adapter: string;
    embeddingModel?: string;
  }): SearchTrace {
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
 * In-memory trace store. The HTTP server exposes its contents via the
 * `/traces` endpoint for trace explorers and observability backends.
 * Bounded — drops oldest when capacity is exceeded.
 */
export class TraceStore {
  private capacity: number;
  private traces: SearchTrace[] = [];

  constructor(capacity = 1000) {
    this.capacity = capacity;
  }

  push(trace: SearchTrace): void {
    this.traces.push(trace);
    if (this.traces.length > this.capacity) this.traces.shift();
  }

  list(limit = 100): SearchTrace[] {
    return this.traces.slice(-limit).reverse();
  }

  get(id: string): SearchTrace | undefined {
    return this.traces.find((t) => t.id === id);
  }

  clear(): void {
    this.traces = [];
  }

  size(): number {
    return this.traces.length;
  }
}
