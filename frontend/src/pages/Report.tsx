import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AppLayout } from "@/components/Layout/AppLayout";
import { ReportViewer } from "@/components/ReportViewer/ReportViewer";
import { copyReportText, printReport } from "@/components/ReportViewer/parse";
import { StatusBadge } from "@/components/StatusBadge";
import { ReportAPI } from "@/api/reports";
import { usePolling } from "@/lib/usePolling";
import { useStudyStore } from "@/store/study";
import { toast } from "@/store/toast";
import { fmtDate } from "@/lib/format";
import type { Report as ReportType } from "@/types";

export function Report() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const storedId = useStudyStore((s) => s.currentReportId);
  const reportId = params.get("report") || storedId;

  const [report, setReport] = useState<ReportType | null>(null);

  usePolling<ReportType>(
    async () => {
      const r = await ReportAPI.getReport(reportId!);
      setReport(r);
      return r;
    },
    4000,
    (r) => r.status === "completed" || r.status === "failed",
    !!reportId,
    [reportId],
  );

  const isDone = report?.status === "completed";

  const headerActions = isDone ? (
    <div className="flex gap-8">
      <button
        className="btn btn-ghost btn-sm"
        onClick={() => {
          if (report) {
            copyReportText(report);
            toast("Copied to clipboard", "success", 2000);
          }
        }}
      >
        Copy
      </button>
      <button
        className="btn btn-ghost btn-sm"
        onClick={() => report && printReport(report)}
      >
        Print
      </button>
      <button className="btn btn-secondary btn-sm" onClick={() => navigate("/upload")}>
        + New Study
      </button>
    </div>
  ) : undefined;

  return (
    <AppLayout
      title="Clinical Report"
      headerExtra={report && <StatusBadge status={report.status} />}
      headerActions={headerActions}
    >
      <div
        style={{
          maxWidth: 820,
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        {report && (
          <div className="card fade-in">
            <div className="card-body">
              <div
                className="flex justify-between items-center"
                style={{ flexWrap: "wrap", gap: 12 }}
              >
                <div className="flex gap-12 items-center">
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-2)" }}>
                      Report ID
                    </div>
                    <div className="mono" style={{ fontSize: 13 }}>
                      {report.id}
                    </div>
                  </div>
                </div>
                <div className="flex gap-16 items-center" style={{ flexWrap: "wrap" }}>
                  <div>
                    <div style={{ fontSize: 12, color: "var(--text-3)" }}>Modality</div>
                    <div style={{ fontWeight: 600 }}>{report.modality || "—"}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, color: "var(--text-3)" }}>Generated</div>
                    <div style={{ fontSize: 13 }}>{fmtDate(report.created_at)}</div>
                  </div>
                  <StatusBadge status={report.status} />
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="card fade-in fade-in-1">
          <div className="card-header">
            <span className="card-icon"></span>
            <h2>AI-Generated Clinical Report</h2>
            <div style={{ flex: 1 }} />
            <span className="badge badge-purple">BioGPT</span>
          </div>
          <div className="card-body">
            {report ? (
              <ReportViewer report={report} />
            ) : (
              <div className="empty-state">
                <div className="empty-icon"></div>
                <div className="empty-title">No report selected</div>
                <div className="empty-sub">
                  Run an analysis and click "Generate AI Report" to create a report.
                </div>
                <button className="btn btn-primary mt-16" onClick={() => navigate("/upload")}>
                  Upload Study
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
