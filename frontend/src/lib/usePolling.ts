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
    let timer: number | undefined;

    const run = async () => {
      try {
        const result = await fnRef.current();
        if (cancelled) return;
        if (stopRef.current(result)) {
          if (timer) window.clearInterval(timer);
          return;
        }
      } catch (e) {
        // keep polling on transient errors
        console.warn("poll error", e);
      }
    };

    void run();
    timer = window.setInterval(run, intervalMs);

    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, ...deps]);
}
