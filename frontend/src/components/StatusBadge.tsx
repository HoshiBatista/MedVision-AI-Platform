import { statusVisual } from "@/lib/format";

export function StatusBadge({ status }: { status: string }) {
  const { cls, label, pulse } = statusVisual(status);
  return (
    <span className={`badge ${cls}`}>
      {label}
      {pulse && <span className="dot pulse" />}
    </span>
  );
}
