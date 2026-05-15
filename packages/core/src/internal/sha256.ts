/**
 * Cross-runtime crypto helpers (`sha256Hex`, `randomUuid`, `utf8ByteLength`).
 *
 * Uses Web Crypto / WHATWG TextEncoder exclusively — both are available
 * in every runtime Augur targets:
 *   - Node ≥ 19 (we require ≥ 20 via `engines`)
 *   - Browsers (modern evergreen)
 *   - Cloudflare Workers, Vercel Edge, Deno, Bun
 *
 * Previously these used `node:crypto` (`createHash`, `randomUUID`) and
 * `Buffer.byteLength`, which made `@augur-rag/core` un-bundleable for
 * browsers and edge runtimes without polyfills. The Web Crypto path
 * gives identical output in every runtime — no behaviour change for
 * Node consumers, and the browser / edge bundle is now real.
 */

/**
 * Lowercase hex SHA-256 digest of a UTF-8 string. Async (Web Crypto's
 * `subtle.digest` is async); callers that previously did sync hashing
 * must await this now.
 */
export async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const hash = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return arrayBufferToHex(hash);
}

function arrayBufferToHex(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let out = "";
  for (let i = 0; i < bytes.length; i++) {
    out += bytes[i]!.toString(16).padStart(2, "0");
  }
  return out;
}

/**
 * RFC 4122 v4 UUID. `globalThis.crypto.randomUUID()` is available in
 * every runtime Augur targets (Node ≥ 19, modern browsers, edge
 * runtimes). If a host strips it, that's a host-config problem, not
 * an Augur problem.
 */
export function randomUuid(): string {
  return globalThis.crypto.randomUUID();
}

/**
 * UTF-8 byte length of a string. Drop-in replacement for
 * `Buffer.byteLength(s, "utf8")` that works outside Node.
 */
export function utf8ByteLength(s: string): number {
  return new TextEncoder().encode(s).length;
}
