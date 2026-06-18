import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { drawBoxes as drawBoxesUtil, clearCanvas } from "@/components/HeatmapOverlay/drawing";
import type { Finding } from "@/types";

export interface DicomViewerHandle {
  zoomIn(): void;
  zoomOut(): void;
  resetView(): void;
  loadHeatmap(url: string): Promise<void>;
  toggleOverlay(show: boolean): void;
  drawBoxes(findings: Finding[], origW: number, origH: number): void;
  clearOverlay(): void;
}

interface DicomViewerProps {
  src?: string | null;
}

/**
 * Study image viewer with zoom + CAM heatmap overlay.
 * Ported from the original ImageViewer class into an imperative React component.
 */
export const DicomViewer = forwardRef<DicomViewerHandle, DicomViewerProps>(
  ({ src }, ref) => {
    const wrapRef = useRef<HTMLDivElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const overlayImgRef = useRef<HTMLImageElement | null>(null);
    const showOverlayRef = useRef(false);

    const [scale, setScale] = useState(1);
    const [bright, setBright] = useState(1);
    const [contrast, setContrast] = useState(1);
    const [hasImage, setHasImage] = useState(false);

    const syncCanvas = () => {
      const wrap = wrapRef.current;
      const canvas = canvasRef.current;
      if (!wrap || !canvas) return;
      const r = wrap.getBoundingClientRect();
      canvas.width = r.width;
      canvas.height = r.height;
    };

    const drawOverlay = () => {
      const canvas = canvasRef.current;
      const img = imgRef.current;
      const overlay = overlayImgRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !img || !overlay || !ctx) return;

      const cw = canvas.width;
      const ch = canvas.height;
      ctx.clearRect(0, 0, cw, ch);

      const iw = img.naturalWidth || cw;
      const ih = img.naturalHeight || ch;
      const ratio = Math.min(cw / iw, ch / ih);
      const dw = iw * ratio;
      const dh = ih * ratio;
      const dx = (cw - dw) / 2;
      const dy = (ch - dh) / 2;

      ctx.globalAlpha = 0.55;
      // Heatmap render may be a multi-panel strip; use the first panel.
      ctx.drawImage(overlay, 0, 0, overlay.width / 3, overlay.height, dx, dy, dw, dh);
      ctx.globalAlpha = 1;
    };

    useImperativeHandle(ref, () => ({
      zoomIn: () => setScale((s) => Math.min(3, s + 0.2)),
      zoomOut: () => setScale((s) => Math.max(0.4, s - 0.2)),
      resetView: () => {
        setScale(1);
        setBright(1);
        setContrast(1);
      },
      loadHeatmap: (url: string) =>
        new Promise<void>((resolve) => {
          const tmp = new Image();
          tmp.crossOrigin = "anonymous";
          tmp.onload = () => {
            overlayImgRef.current = tmp;
            if (showOverlayRef.current) drawOverlay();
            resolve();
          };
          tmp.onerror = () => resolve();
          tmp.src = url;
        }),
      toggleOverlay: (show: boolean) => {
        showOverlayRef.current = show;
        if (show) drawOverlay();
        else clearCanvas(canvasRef.current);
      },
      drawBoxes: (findings, origW, origH) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        syncCanvas();
        clearCanvas(canvas);
        drawBoxesUtil(canvas, findings, origW, origH);
      },
      clearOverlay: () => clearCanvas(canvasRef.current),
    }));

    // Keep the overlay canvas sized to the wrapper.
    useEffect(() => {
      syncCanvas();
      const ro = new ResizeObserver(() => {
        syncCanvas();
        if (showOverlayRef.current) drawOverlay();
      });
      if (wrapRef.current) ro.observe(wrapRef.current);
      return () => ro.disconnect();
    }, []);

    return (
      <div className="viewer-wrap" ref={wrapRef}>
        {src && (
          <img
            ref={imgRef}
            src={src}
            alt="Study image"
            style={{
              filter: `brightness(${bright}) contrast(${contrast})`,
              transform: `scale(${scale})`,
            }}
            onLoad={() => {
              setHasImage(true);
              syncCanvas();
            }}
            onError={() => setHasImage(false)}
          />
        )}
        <canvas className="viewer-overlay-canvas" ref={canvasRef} />
        {!hasImage && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 12,
              padding: "60px 20px",
            }}
          >
            <div style={{ fontSize: 48, opacity: 0.3 }}></div>
            <div style={{ color: "var(--text-3)", fontSize: 14 }}>
              Image will appear after upload
            </div>
          </div>
        )}
      </div>
    );
  },
);

DicomViewer.displayName = "DicomViewer";
