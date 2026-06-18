import type { Report } from "@/types";

export interface ReportSection {
  title: string;
  text: string;
}

const SECTION_MARKERS: [string, string][] = [
  ["TECHNIQUE", "Technique"],
  ["CLINICAL INDICATION", "Clinical Indication"],
  ["FINDINGS", "Findings"],
  ["IMPRESSION", "Impression"],
];

/**
 * Split raw BioGPT output into structured radiology sections.
 * Ported from report_viewer.js#parseReportSections.
 */
export function parseReportSections(raw: string): ReportSection[] {
  const cleanedFull = raw.replace(/DISCLAIMER:.*/s, "").trim();
  const sections: ReportSection[] = [];
  let remaining = cleanedFull;

  for (const [marker, title] of SECTION_MARKERS) {
    const re = new RegExp(`(?:^|\\n)${marker}[:\\s]*`, "i");
    const match = remaining.search(re);
    if (match >= 0) {
      const before = remaining.slice(0, match).trim();
      if (before && sections.length === 0) {
        sections.push({ title: "Report", text: before });
      }
      const afterStart = remaining.slice(match).search(/\n[A-Z]{4,}[:\s]/);
      let sectionText: string;
      if (afterStart > 0) {
        sectionText = remaining.slice(match).slice(0, afterStart);
        remaining = remaining.slice(match + sectionText.length);
      } else {
        sectionText = remaining.slice(match);
        remaining = "";
      }
      sectionText = sectionText.replace(re, "").trim();
      sections.push({ title, text: sectionText });
    }
  }

  if (remaining.trim()) {
    if (sections.length === 0) {
      sections.push({ title: "Generated Report", text: remaining.trim() });
    } else {
      sections.push({ title: "Continuation", text: remaining.trim() });
    }
  }

  return sections;
}

/** Copy raw report text to clipboard. */
export function copyReportText(report: Report): void {
  void navigator.clipboard.writeText(report.content || "").catch(() => undefined);
}

/** Open a print-friendly window for the report. */
export function printReport(report: Report): void {
  const win = window.open("", "_blank");
  if (!win) return;
  const esc = (s: string) =>
    s.replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string,
    );
  win.document.write(`
    <!DOCTYPE html><html><head>
    <title>MedVision AI Report</title>
    <style>
      body{font-family:Georgia,serif;max-width:720px;margin:48px auto;line-height:1.8;color:#111}
      h1{font-size:22px;margin-bottom:4px}
      .disc{background:#fffbe6;border:1px solid #f59e0b;padding:12px 16px;margin-top:24px;font-size:13px;color:#92400e}
    </style></head><body>
    <h1>MedVision AI — Clinical Report</h1>
    <p style="color:#666;font-size:14px">Generated: ${new Date().toLocaleString()} · Modality: ${esc(report.modality || "")}</p>
    <hr>
    <pre style="white-space:pre-wrap;font-family:inherit">${esc(report.content || "")}</pre>
    <div class="disc">AI-Assisted. Requires radiologist review before clinical use.</div>
    </body></html>`);
  win.document.close();
  win.print();
}
