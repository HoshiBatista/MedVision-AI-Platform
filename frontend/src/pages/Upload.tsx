import { useEffect, useRef, useState, type DragEvent } from "react";
import { useNavigate } from "react-router-dom";
import { AppLayout } from "@/components/Layout/AppLayout";
import { UploadAPI } from "@/api/upload";
import { AnalysisAPI } from "@/api/analysis";
import { errMessage } from "@/api/client";
import { useStudyStore } from "@/store/study";
import { toast } from "@/store/toast";
import type { Modality, Task } from "@/types";

const MODALITY_TASK: Record<Modality, Task> = {
  MRI: "segmentation",
  CXR: "detection",
  DERM: "classification",
};

const MODALITY_INFO: Record<Modality, string> = {
  MRI: "MRI mode: Segmentation — YOLOv11 tumour detection & mask.",
  CXR: "CXR mode: Detection — YOLOv11 pneumonia region detection.",
  DERM: "Derm mode: Classification — YOLOv11 lesion classification (7 classes).",
};

const MODALITIES: { value: Modality; name: string; desc: string }[] = [
  { value: "MRI", name: "Brain MRI", desc: "DICOM · FLAIR/T1/T2" },
  { value: "CXR", name: "Chest X-Ray", desc: "DICOM · PA/LAT" },
  { value: "DERM", name: "Dermoscopy", desc: "PNG/JPEG · HAM10000" },
];

export function Upload() {
  const navigate = useNavigate();
  const setCurrentJob = useStudyStore((s) => s.setCurrentJob);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [modality, setModality] = useState<Modality>("MRI");
  const [file, setFile] = useState<File | null>(null);
  const [thumb, setThumb] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!file || !file.type.startsWith("image/")) {
      setThumb(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => setThumb(e.target?.result as string);
    reader.readAsDataURL(file);
  }, [file]);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  };

  const removeFile = () => {
    setFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSubmit = async () => {
    if (!file) return;
    setSubmitting(true);
    try {
      const uploaded = await UploadAPI.uploadFile(file, modality);
      toast("File uploaded ", "success", 2000);

      const job = await AnalysisAPI.analyze(uploaded.study_id, MODALITY_TASK[modality]);
      setCurrentJob({ id: job.job_id, study_id: uploaded.study_id, modality });
      toast("Analysis job queued ", "success", 2000);

      setTimeout(() => navigate(`/analysis?job=${job.job_id}`), 600);
    } catch (err) {
      toast(errMessage(err), "error");
      setSubmitting(false);
    }
  };

  return (
    <AppLayout
      title="New Study Upload"
      headerExtra={<span className="badge badge-cyan">Upload & Analyse</span>}
    >
      <div style={{ maxWidth: 760, margin: "0 auto" }}>
        {/* Step 1: modality */}
        <div className="card fade-in fade-in-1 mb-16">
          <div className="card-header">
            <span className="card-icon">1</span>
            <h2>Select Modality</h2>
          </div>
          <div className="card-body">
            <div className="modality-grid">
              {MODALITIES.map((m) => (
                <label
                  key={m.value}
                  className={`modality-card${modality === m.value ? " selected" : ""}`}
                >
                  <input
                    type="radio"
                    name="modality"
                    value={m.value}
                    checked={modality === m.value}
                    onChange={() => setModality(m.value)}
                  />
                  <div className="modality-name">{m.name}</div>
                  <div className="modality-desc">{m.desc}</div>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Step 2: file */}
        <div className="card fade-in fade-in-2 mb-16">
          <div className="card-header">
            <span className="card-icon">2</span>
            <h2>Upload Image</h2>
          </div>
          <div className="card-body">
            <div
              className={`upload-zone${dragOver ? " drag-over" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                className="upload-file-input"
                type="file"
                accept=".dcm,.dicom,.png,.jpg,.jpeg,.bmp,.tiff"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <div className="upload-icon"></div>
              <div className="upload-title">Drop file here or click to browse</div>
              <div className="upload-sub">DICOM, PNG, JPEG — up to 512 MB</div>
              <div className="upload-hint">Supports .dcm, .dicom, .png, .jpg</div>
            </div>

            {file && (
              <div className="file-preview">
                <div className="file-thumb">
                  {thumb ? (
                    <img
                      src={thumb}
                      style={{ width: 52, height: 52, objectFit: "cover", borderRadius: 6 }}
                      alt="preview"
                    />
                  ) : (
                    ""
                  )}
                </div>
                <div>
                  <div className="file-name">{file.name}</div>
                  <div className="file-meta">
                    {(file.size / 1024 / 1024).toFixed(2)} MB · {file.type || "DICOM"}
                  </div>
                </div>
                <button
                  className="btn-icon"
                  style={{ marginLeft: "auto" }}
                  title="Remove"
                  onClick={removeFile}
                >

                </button>
              </div>
            )}
          </div>
        </div>

        {/* Step 3: submit */}
        <div className="card fade-in fade-in-3">
          <div className="card-header">
            <span className="card-icon">3</span>
            <h2>Run Analysis</h2>
          </div>
          <div className="card-body flex-col gap-16">
            <div className="alert alert-info">
              <span className="alert-icon"></span>
              <span>{MODALITY_INFO[modality]}</span>
            </div>
            {submitting && (
              <div className="progress-bar">
                <div className="progress-fill indeterminate" />
              </div>
            )}
            <button
              className="btn btn-primary btn-lg"
              disabled={!file || submitting}
              onClick={handleSubmit}
            >
              {submitting ? (
                <>
                  <span className="spinner" /> Uploading…
                </>
              ) : (
                "Upload & Analyse"
              )}
            </button>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
