/**
 * Makes a fast promise take at least `minMs` to resolve.
 *
 * WHY: opening a sample is a single DB read (tens of milliseconds) — with no floor on the
 * duration, the ProcessingModal would flash and vanish, which reads as broken rather than fast.
 * Samples are pre-computed (no inference call is made), so this delay is purely the UI giving the
 * user a moment to register that something happened before showing the real, already-computed
 * result — it never fabricates the result itself (D-06).
 */
export async function withMinDuration<T>(promise: Promise<T>, minMs: number): Promise<T> {
  const [result] = await Promise.all([promise, new Promise<void>((resolve) => setTimeout(resolve, minMs))]);
  return result;
}
