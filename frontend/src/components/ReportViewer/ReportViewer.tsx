import { Fragment } from "react";
import type { Report } from "@/types";
import { parseReportSections } from "./parse";

/** Render newline-separated text with <br/> breaks. */
function MultilineText({ text }: { text: string }) {
  const lines = text.split(/\n+/);
  return (
    <>
      {lines.map((line, i) => (
        <Fragment key={i}>
          {line}
          {i < lines.length - 1 && <br />}
        </Fragment>
      ))}
    </>
  );
}

export function ReportViewer({ report }: { report: Report }) {
  if (report.status === "pending" || report.status === "generating") {
    return (
      <div className="empty-state">
        <div className="spinner spinner-lg" />
        <div className="empty-title">Generating report…</div>
        <div className="empty-sub">
          BioGPT is analysing the findings. This may take 20–60 seconds.
        </div>
      </div>
    );
  }

  if (report.status === "failed") {
    return (
      <div className="alert alert-error">
        <span className="alert-icon"></span>
        <div>
          <strong>Report generation failed</strong>
          <br />
          {report.error || "Unknown error"}
        </div>
      </div>
    );
  }

  const sections = parseReportSections(report.content || "");

  return (
    <div className="report-content fade-in">
      {sections.length === 0 ? (
        <p className="text-muted">Report content unavailable.</p>
      ) : (
        sections.map((s, i) => (
          <div className="report-section" key={i}>
            <div className="report-section-title">{s.title}</div>
            <p>
              <MultilineText text={s.text} />
            </p>
          </div>
        ))
      )}
      <div className="report-disclaimer">
        <strong>AI-Assisted Report Disclaimer</strong>
        This report was generated with AI assistance (BioGPT) based on automated image analysis. It{" "}
        <strong>must be reviewed and validated</strong> by a qualified radiologist or clinician
        before any clinical use. This output is not a substitute for professional medical judgment.
      </div>
    </div>
  );
}
