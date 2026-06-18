import { useEffect, useRef } from "react";

/**
 * Repeatedly invoke an async function until `stopWhen` returns true.
 * Mirrors the original poll() helper but as a React hook.
 *
 * @param fn        async producer; its result is passed to stopWhen
 * @param intervalMs polling cadence
 * @param stopWhen  predicate that halts polling when it returns true
 * @param enabled   gate polling on/off (e.g. when no id is present)
 * @param deps      extra dependencies that restart polling when changed
 */
export function usePolling<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  stopWhen: (result: T) => boolean,
  enabled = true,
  deps: unknown[] = [],
): void {
  const fnRef = useRef(fn);
  const stopRef = useRef(stopWhen);
  fnRef.current = fn;
  stopRef.current = stopWhen;

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    // Returns true when polling should stop.
    const tick = async (): Promise<boolean> => {
      try {
        const result = await fnRef.current();
        if (cancelled) return true;
        return stopRef.current(result);
      } catch (e) {
        // keep polling on transient errors
        console.warn("poll error", e);
        return false;
      }
    };

    void tick();
    const timer = window.setInterval(async () => {
      if (await tick()) window.clearInterval(timer);
    }, intervalMs);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, ...deps]);
}
