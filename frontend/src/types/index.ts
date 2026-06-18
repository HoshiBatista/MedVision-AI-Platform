/**
 * Shared API types for the MedVision AI platform.
 * Mirrors the service contracts defined in CLAUDE.md.
 */

export type Modality = "MRI" | "CXR" | "DERM";
export type Task = "segmentation" | "detection" | "classification";
export type JobStatus = "queued" | "running" | "completed" | "failed";
export type ReportStatus = "pending" | "generating" | "completed" | "failed";

/* ── Auth ───────────────────────────────────────────────────────────────── */
export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in?: number;
}

export interface User {
  id: number;
  email: string;
  full_name?: string | null;
  role: string;
  is_active?: boolean;
}

/* ── Upload ─────────────────────────────────────────────────────────────── */
export interface UploadResponse {
  study_id: string;
  status: string;
  storage_path: string;
}

/* ── Analysis ───────────────────────────────────────────────────────────── */
export interface Finding {
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number];
  area_px2: number;
  label?: string;
}

export interface AnalysisResults {
  findings?: Finding[];
  top_confidence?: number | null;
  num_detections?: number;
  orig_size?: [number, number];
  processing_time_s?: number;
  image_path?: string;
  mask_url?: string;
  gradcam_url?: string;
}

export interface JobResult {
  id: string;
  job_id?: string;
  study_id: string;
  task: Task;
  status: JobStatus;
  results?: AnalysisResults | null;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobQueuedResponse {
  job_id: string;
  status: string;
}

/* ── GradCAM ────────────────────────────────────────────────────────────── */
export interface ExplainResponse {
  heatmap_path: string;
  heatmap_url?: string;
  top_regions?: unknown[];
}

/* ── Reports ────────────────────────────────────────────────────────────── */
export interface ReportGenerateRequest {
  study_id: string;
  modality: Modality;
  job_ids: string[];
  findings: {
    confidence?: number | null;
    regions?: Finding[];
    bbox_count?: number;
    top_class?: string;
  };
  patient_context?: Record<string, unknown> | null;
}

export interface ReportGenerateResponse {
  report_id?: string;
  id?: string;
  status: ReportStatus;
}

export interface Report {
  id: string;
  study_id: string;
  modality: Modality;
  status: ReportStatus;
  content?: string;
  error?: string | null;
  created_at: string;
}
