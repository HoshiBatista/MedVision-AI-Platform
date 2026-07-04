import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AppLayout } from "@/components/Layout/AppLayout";
import { StatusBadge } from "@/components/StatusBadge";
import { AnalysisAPI } from "@/api/analysis";
import { UploadAPI } from "@/api/upload";
import { errMessage } from "@/api/client";
import { fmtDate } from "@/lib/format";
import { useStudyStore } from "@/store/study";
import { toast } from "@/store/toast";
import type { JobSummary, Study } from "@/types";

type Tab = "jobs" | "studies";

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function History() {
  const navigate = useNavigate();
  const setCurrentJob = useStudyStore((s) => s.setCurrentJob);

  const [tab, setTab] = useState<Tab>("jobs");
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [studies, setStudies] = useState<Study[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [jobRes, studyRes] = await Promise.all([
          AnalysisAPI.listJobs(50),
          UploadAPI.listStudies(50),
        ]);
        setJobs(jobRes.items);
        setStudies(studyRes.items);
      } catch (err) {
        toast(errMessage(err), "error");
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  const stats = useMemo(() => {
    const completed = jobs.filter((j) => j.status === "completed").length;
    const running = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
    const failed = jobs.filter((j) => j.status === "failed").length;
    return { studies: studies.length, completed, running, failed };
  }, [jobs, studies]);

  const openJob = (job: JobSummary) => {
    const study = studies.find((s) => s.id === job.study_id);
    setCurrentJob({
      id: job.job_id,
      study_id: job.study_id,
      modality: study?.modality || "MRI",
    });
    navigate(`/analysis?job=${job.job_id}`);
  };

  return (
    <AppLayout
      title="Study History"
      headerActions={
        <button className="btn btn-primary btn-sm" onClick={() => navigate("/upload")}>
          + New Study
        </button>
      }
    >
      <div className="stats-grid fade-in" style={{ marginBottom: 20 }}>
        <div className="stat-card">
          <div className="stat-icon cyan"></div>
          <div className="stat-value">{stats.studies}</div>
          <div className="stat-label">Studies uploaded</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon green"></div>
          <div className="stat-value">{stats.completed}</div>
          <div className="stat-label">Completed analyses</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon amber"></div>
          <div className="stat-value">{stats.running}</div>
          <div className="stat-label">In progress</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon purple"></div>
          <div className="stat-value">{stats.failed}</div>
          <div className="stat-label">Failed</div>
        </div>
      </div>

      <div className="card fade-in fade-in-1">
        <div className="tabs">
          <button
            className={`tab-btn${tab === "jobs" ? " active" : ""}`}
            onClick={() => setTab("jobs")}
          >
            Analysis jobs
          </button>
          <button
            className={`tab-btn${tab === "studies" ? " active" : ""}`}
            onClick={() => setTab("studies")}
          >
            Uploaded studies
          </button>
        </div>

        <div className="card-body">
          {loading ? (
            <div className="empty-state" style={{ padding: 40 }}>
              <div className="spinner spinner-lg" />
            </div>
          ) : tab === "jobs" ? (
            jobs.length === 0 ? (
              <div className="empty-state">
                <div className="empty-title">No analyses yet</div>
                <div className="empty-sub">Upload a study to queue your first job.</div>
              </div>
            ) : (
              <div className="job-list">
                {jobs.map((job) => (
                  <button
                    key={job.job_id}
                    type="button"
                    className="job-item"
                    onClick={() => openJob(job)}
                  >
                    <div>
                      <div className="job-title">{job.task}</div>
                      <div className="job-meta mono">{job.job_id.slice(0, 8)}…</div>
                    </div>
                    <div className="job-meta">{fmtDate(job.created_at)}</div>
                    <StatusBadge status={job.status} />
                  </button>
                ))}
              </div>
            )
          ) : studies.length === 0 ? (
            <div className="empty-state">
              <div className="empty-title">No studies yet</div>
              <div className="empty-sub">Upload DICOM or raster images from the Upload page.</div>
            </div>
          ) : (
            <div className="job-list">
              {studies.map((study) => (
                <div key={study.id} className="job-item" style={{ cursor: "default" }}>
                  <div>
                    <div className="job-title">{study.original_filename}</div>
                    <div className="job-meta mono">{study.id.slice(0, 8)}…</div>
                  </div>
                  <span className="badge badge-purple">{study.modality}</span>
                  <div className="job-meta">{formatBytes(study.file_size_bytes)}</div>
                  <div className="job-meta">{fmtDate(study.created_at)}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
