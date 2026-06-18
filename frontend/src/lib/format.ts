import type { JobStatus, ReportStatus } from "@/types";

export function fmtConf(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function capitalize(s: string | null | undefined): string {
  return s ? s[0].toUpperCase() + s.slice(1) : "";
}

type BadgeVisual = { cls: string; label: string; pulse: boolean };

export function statusVisual(status: JobStatus | ReportStatus | string): BadgeVisual {
  const map: Record<string, [string, string]> = {
    queued: ["badge-gray", "Queued"],
    running: ["badge-amber", "Running"],
    generating: ["badge-amber", "Generating"],
    pending: ["badge-gray", "Pending"],
    completed: ["badge-green", "Completed"],
    failed: ["badge-red", "Failed"],
  };
  const [cls, label] = map[status] || ["badge-gray", status];
  return { cls, label, pulse: status === "running" || status === "generating" };
}
