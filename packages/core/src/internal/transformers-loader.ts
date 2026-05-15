/**
 * Shared loader for the optional `@huggingface/transformers` peer dep.
 *
 * `LocalEmbedder` and `LocalReranker` both pull the same module in via
 * dynamic import. Both used to fail with a raw `ERR_MODULE_NOT_FOUND` if
 * the user forgot to install the peer — confusing because the SDK still
 * imports fine; the error only fires on the first `embed()` or `rerank()`
 * call. This helper turns that opaque failure into an actionable message
 * naming the missing package and the install command.
 *
 * The progress-callback type and the loose import-error interpretation
 * live here too so the two consumers don't drift.
 */

const HINT =
  "LocalEmbedder / LocalReranker require the optional peer dependency '@huggingface/transformers'.\n" +
  "Install it with: npm install @huggingface/transformers\n" +
  "(Augur keeps it as an optional peer to avoid pulling ~100 MB into installs that don't need it.)";

export class MissingTransformersError extends Error {
  override readonly cause?: unknown;
  constructor(cause?: unknown) {
    super(HINT);
    this.name = "MissingTransformersError";
    if (cause !== undefined) this.cause = cause;
  }
}

/**
 * Event passed to `onProgress` callbacks. Mirrors the shape emitted by
 * `@huggingface/transformers`' `progress_callback` so we don't reshape
 * upstream events: `status` is always present, the rest depend on the
 * phase (initiate → download → progress → done → ready).
 *
 * Typed permissively (everything but `status` optional) so future
 * upstream additions don't break the contract.
 */
export interface DownloadProgressEvent {
  status: string;
  name?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

export type ProgressCallback = (event: DownloadProgressEvent) => void;

/**
 * Pipeline factory accepted by `transformers.pipeline()`. Re-typed here
 * rather than imported because the peer dep is optional; we'd otherwise
 * be forced to install it just for the type.
 */
export interface TransformersModule {
  pipeline: (
    task: string,
    model: string,
    opts?: {
      dtype?: string;
      device?: string;
      progress_callback?: ProgressCallback;
    }
  ) => Promise<unknown>;
}

/**
 * Best-effort classifier for "is this error caused by the peer dep
 * being missing?" Exported pure so the helper itself stays testable
 * without monkey-patching the dynamic-import path.
 */
export function interpretImportError(e: unknown): MissingTransformersError | null {
  const code = (e as { code?: string } | null | undefined)?.code;
  const message = String((e as { message?: string } | null | undefined)?.message ?? "");
  const isModuleNotFound =
    code === "ERR_MODULE_NOT_FOUND" ||
    code === "MODULE_NOT_FOUND" ||
    /Cannot find (module|package) ['"]@huggingface\/transformers['"]/.test(message);
  return isModuleNotFound ? new MissingTransformersError(e) : null;
}

/**
 * Dynamically import `@huggingface/transformers` and re-throw with a
 * `MissingTransformersError` if the package isn't installed. Any other
 * import failure (corrupt install, native-binding mismatch, etc.) is
 * propagated unchanged so the user still sees the original cause.
 */
export async function loadTransformers(): Promise<TransformersModule> {
  try {
    return (await import("@huggingface/transformers")) as unknown as TransformersModule;
  } catch (e) {
    const friendly = interpretImportError(e);
    if (friendly) throw friendly;
    throw e;
  }
}
