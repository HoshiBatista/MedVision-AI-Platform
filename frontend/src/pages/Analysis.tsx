import { useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AppLayout } from "@/components/Layout/AppLayout";
import { DicomViewer, type DicomViewerHandle } from "@/components/DicomViewer/DicomViewer";
import { ResultsPanel } from "@/components/ResultsPanel/ResultsPanel";
import { StatusBadge } from "@/components/StatusBadge";
import { AnalysisAPI } from "@/api/analysis";
import { GradCAMAPI } from "@/api/gradcam";
import { ReportAPI } from "@/api/reports";
import { errMessage } from "@/api/client";
import { usePolling } from "@/lib/usePolling";
import { useStudyStore } from "@/store/study";
import { toast } from "@/store/toast";
import { fmtDate } from "@/lib/format";
import type { JobResult, Task } from "@/types";

const MODEL_MAP: Record<Task, string> = {
  segmentation: "mri_segmentation",
  detection: "pneumonia_detection",
  classification: "skin_classification",
};

const basename = (p: string) => p.split("/").pop() || "";

function NoJob() {
  const navigate = useNavigate();
  return (
    <div className="empty-state">
      <div className="empty-icon"></div>
      <div className="empty-title">No analysis selected</div>
      <div className="empty-sub">
        Upload a study first, or pass a job ID in the URL: ?job=&lt;id&gt;
      </div>
      <button className="btn btn-primary mt-16" onClick={() => navigate("/upload")}>
        Upload Study
      </button>
    </div>
  );
}

export function Analysis() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const jobId = params.get("job");
  const currentJob = useStudyStore((s) => s.currentJob);
  const setCurrentReportId = useStudyStore((s) => s.setCurrentReportId);

  const viewerRef = useRef<DicomViewerHandle>(null);
  const [job, setJob] = useState<JobResult | null>(null);
  const [showBoxes, setShowBoxes] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapReady, setHeatmapReady] = useState(false);
  const [explaining, setExplaining] = useState(false);
  const [generating, setGenerating] = useState(false);

  const modality = currentJob?.modality || "MRI";

  usePolling<JobResult>(
    async () => {
      const result = await AnalysisAPI.getResult(jobId!);
      setJob(result);
      return result;
    },
    3000,
    (j) => j.status === "completed" || j.status === "failed",
    !!jobId,
    [jobId],
  );

  const imageSrc = useMemo(() => {
    const path = job?.results?.image_path;
    return path ? `/static/studies/${basename(path)}` : null;
  }, [job]);

  const findings = job?.results?.findings ?? [];
  const isDone = job?.status === "completed";

  const toggleBoxes = () => {
    const next = !showBoxes;
    setShowBoxes(next);
    if (next && findings.length) {
      const [ow, oh] = job?.results?.orig_size || [640, 640];
      viewerRef.current?.drawBoxes(findings, ow, oh);
    } else {
      viewerRef.current?.clearOverlay();
      if (showHeatmap) viewerRef.current?.toggleOverlay(true);
    }
  };

  const toggleHeatmap = () => {
    const next = !showHeatmap;
    setShowHeatmap(next);
    viewerRef.current?.toggleOverlay(next);
  };

  const requestHeatmap = async () => {
    if (!job?.results) return;
    setExplaining(true);
    try {
      const res = await GradCAMAPI.explain(
        MODEL_MAP[job.task],
        job.results.image_path || "",
        job.results.findings?.[0]?.class_id ?? null,
      );
      await viewerRef.current?.loadHeatmap(`/static/heatmaps/${basename(res.heatmap_path)}`);
      setHeatmapReady(true);
      toast("Heatmap ready — click Heatmap to toggle", "success");
    } catch (err) {
      toast("Heatmap: " + errMessage(err), "warning");
    } finally {
      setExplaining(false);
    }
  };

  const generateReport = async () => {
    if (!job?.results) return;
    setGenerating(true);
    try {
      const f = job.results;
      const report = await ReportAPI.generate({
        study_id: currentJob?.study_id || job.study_id,
        modality,
        job_ids: [job.id],
        findings: {
          confidence: f.top_confidence,
          regions: f.findings || [],
          bbox_count: f.num_detections,
          top_class: String(f.findings?.[0]?.class_id ?? ""),
        },
        patient_context: null,
      });
      const reportId = report.report_id || report.id;
      if (reportId) {
        setCurrentReportId(reportId);
        navigate(`/report?report=${reportId}`);
      }
    } catch (err) {
      toast(errMessage(err), "error");
      setGenerating(false);
    }
  };

  return (
    <AppLayout
      title="Analysis Results"
      headerExtra={job && <StatusBadge status={job.status} />}
      headerActions={
        <button className="btn btn-secondary btn-sm" onClick={() => navigate("/upload")}>
          + New Study
        </button>
      }
    >
      {!jobId ? (
        <NoJob />
      ) : (
        <div className="analysis-layout">
          {/* Left: viewer */}
          <div>
            <div className="card fade-in">
              <div className="card-header">
                <span className="card-icon"></span>
                <h2>Study Image</h2>
                <div style={{ flex: 1 }} />
                <span className="badge badge-purple">{modality}</span>
              </div>
              <div className="card-body">
                <DicomViewer ref={viewerRef} src={imageSrc} />
                <div className="viewer-toolbar">
                  <button
                    className="btn btn-ghost btn-sm"
                    data-tip="Zoom in"
                    onClick={() => viewerRef.current?.zoomIn()}
                  >
                    +
                  </button>
                  <button
                    className="btn btn-ghost btn-sm"
                    data-tip="Zoom out"
                    onClick={() => viewerRef.current?.zoomOut()}
                  >
                    −
                  </button>
                  <button
                    className="btn btn-ghost btn-sm"
                    data-tip="Reset view"
                    onClick={() => viewerRef.current?.resetView()}
                  >
                    Reset
                  </button>
                  <div style={{ flex: 1 }} />
                  <button
                    className={`btn btn-sm ${showHeatmap ? "btn-primary" : "btn-secondary"}`}
                    data-tip="Toggle CAM overlay"
                    disabled={!heatmapReady}
                    onClick={toggleHeatmap}
                  >
                    Heatmap
                  </button>
                  <button
                    className={`btn btn-sm ${showBoxes ? "btn-primary" : "btn-secondary"}`}
                    data-tip="Toggle bounding boxes"
                    disabled={!isDone || findings.length === 0}
                    onClick={toggleBoxes}
                  >
                    Boxes
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right: results */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="card fade-in fade-in-1">
              <div className="card-header">
                <span className="card-icon"></span>
                <h2>Job Details</h2>
              </div>
              <div className="card-body flex-col gap-8">
                <div className="flex justify-between items-center">
                  <span className="text-muted" style={{ fontSize: 13 }}>
                    Job ID
                  </span>
                  <span className="mono" style={{ fontSize: 12 }}>
                    {job?.id || "—"}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted" style={{ fontSize: 13 }}>
                    Task
                  </span>
                  <span>
                    {job ? job.task : "—"}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted" style={{ fontSize: 13 }}>
                    Started
                  </span>
                  <span style={{ fontSize: 13 }}>{fmtDate(job?.created_at)}</span>
                </div>
                <div className="divider" />
                <div>
                  <div className="progress-bar">
                    <div
                      className={`progress-fill${isDone || job?.status === "failed" ? "" : " indeterminate"}`}
                      style={{
                        width: isDone ? "100%" : job?.status === "failed" ? "0" : undefined,
                        background: job?.status === "failed" ? "var(--red)" : undefined,
                      }}
                    />
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "var(--text-3)",
                      marginTop: 6,
                      textAlign: "center",
                    }}
                  >
                    {isDone
                      ? "Analysis complete"
                      : job?.status === "failed"
                        ? "Analysis failed"
                        : `${job?.status ?? "waiting"}…`}
                  </div>
                </div>
              </div>
            </div>

            <div className="card fade-in fade-in-2">
              <div className="card-header">
                <span className="card-icon"></span>
                <h2>Findings</h2>
              </div>
              <div className="card-body">
                {job ? (
                  <ResultsPanel job={job} />
                ) : (
                  <div className="empty-state" style={{ padding: "32px 20px" }}>
                    <div className="spinner spinner-lg" />
                    <div className="empty-title">Awaiting results…</div>
                  </div>
                )}
              </div>
            </div>

            {isDone && (
              <div className="card fade-in fade-in-3">
                <div className="card-body flex-col gap-10">
                  <button
                    className="btn btn-primary btn-full"
                    disabled={generating}
                    onClick={generateReport}
                  >
                    {generating ? (
                      <>
                        <span className="spinner" /> Generating report…
                      </>
                    ) : (
                      "Generate AI Report"
                    )}
                  </button>
                  <button
                    className="btn btn-secondary btn-full"
                    disabled={explaining}
                    onClick={requestHeatmap}
                  >
                    {explaining ? (
                      <>
                        <span className="spinner" /> Requesting heatmap…
                      </>
                    ) : (
                      "Request GradCAM Heatmap"
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </AppLayout>
  );
}
