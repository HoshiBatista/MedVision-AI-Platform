import { fmtConf } from "@/lib/format";
import type { JobResult } from "@/types";

export function ResultsPanel({ job }: { job: JobResult }) {
  if (job.status === "failed") {
    return (
      <div className="alert alert-error">
        <span className="alert-icon"></span>
        <div>
          <strong>Analysis failed</strong>
          <br />
          {job.error || "Unknown error"}
        </div>
      </div>
    );
  }

  if (job.status !== "completed") {
    return (
      <div className="empty-state">
        <div className="spinner spinner-lg" />
        <div className="empty-title">Analysis in progress…</div>
        <div className="empty-sub">Results will appear here when the job completes.</div>
      </div>
    );
  }

  const r = job.results || {};
  const findings = r.findings || [];
  const conf = r.top_confidence;

  return (
    <>
      <div className="conf-meter mb-16">
        <div className="conf-label">Top confidence</div>
        <div className="conf-value text-cyan">{fmtConf(conf)}</div>
        <div className="conf-bar">
          <div
            className="conf-fill"
            style={{ width: `${conf != null ? (conf * 100).toFixed(1) : 0}%` }}
          />
        </div>
      </div>

      <div className="flex gap-12 mb-16" style={{ flexWrap: "wrap" }}>
        <div className="badge badge-cyan">{job.task}</div>
        <div className="badge badge-gray">
          {r.orig_size ? r.orig_size.join("×") : "—"} px
        </div>
        <div className="badge badge-purple">
          {findings.length} finding{findings.length !== 1 ? "s" : ""}
        </div>
        {r.processing_time_s != null && (
          <div className="badge badge-gray">{r.processing_time_s}s</div>
        )}
      </div>

      <div className="findings-list">
        {findings.length === 0 ? (
          <div className="empty-state" style={{ padding: "32px 20px" }}>
            <div className="empty-icon"></div>
            <div className="empty-title">No findings detected</div>
            <div className="empty-sub">
              The model found no regions above confidence threshold.
            </div>
          </div>
        ) : (
          findings.map((f, i) => (
            <div className="finding-item" key={i}>
              <div className="finding-rank">{i + 1}</div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div className="finding-label">{f.label || `Class ${f.class_id}`}</div>
                <div className="finding-meta">
                  BBox: [{f.bbox.map((v) => Math.round(v)).join(", ")}] &nbsp;•&nbsp; Area:{" "}
                  {Math.round(f.area_px2).toLocaleString()} px²
                </div>
              </div>
              <div className="finding-conf">{fmtConf(f.confidence)}</div>
            </div>
          ))
        )}
      </div>
    </>
  );
}
