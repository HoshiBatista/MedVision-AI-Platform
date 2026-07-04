import {
  forwardRef,
  useCallback,
  useEffect,
  useId,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { Enums, RenderingEngine, type Types } from "@cornerstonejs/core";
import { drawBoxes as drawBoxesUtil, clearCanvas } from "@/components/HeatmapOverlay/drawing";
import { initCornerstone } from "@/lib/cornerstoneInit";
import { isDicomPath } from "@/lib/studyUrl";
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
 * Medical image viewer: Cornerstone3 for DICOM, raster fallback for PNG/JPEG.
 * Canvas overlay supports GradCAM heatmaps and detection boxes.
 */
export const DicomViewer = forwardRef<DicomViewerHandle, DicomViewerProps>(
  ({ src }, ref) => {
    const uid = useId().replace(/:/g, "");
    const engineId = `medvision-engine-${uid}`;
    const viewportId = `medvision-viewport-${uid}`;

    const wrapRef = useRef<HTMLDivElement>(null);
    const viewportRef = useRef<HTMLDivElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const overlayImgRef = useRef<HTMLImageElement | null>(null);
    const showOverlayRef = useRef(false);
    const engineRef = useRef<RenderingEngine | null>(null);
    const modeRef = useRef<"dicom" | "raster">("raster");

    const [scale, setScale] = useState(1);
    const [bright, setBright] = useState(1);
    const [contrast, setContrast] = useState(1);
    const [hasImage, setHasImage] = useState(false);

    const useDicom = !!src && isDicomPath(src);

    const syncCanvas = useCallback(() => {
      const wrap = wrapRef.current;
      const canvas = canvasRef.current;
      if (!wrap || !canvas) return;
      const r = wrap.getBoundingClientRect();
      canvas.width = r.width;
      canvas.height = r.height;
    }, []);

    const drawOverlay = useCallback(() => {
      const canvas = canvasRef.current;
      const overlay = overlayImgRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !overlay || !ctx) return;

      const cw = canvas.width;
      const ch = canvas.height;
      ctx.clearRect(0, 0, cw, ch);

      const iw = overlay.width / 3;
      const ih = overlay.height;
      const ratio = Math.min(cw / iw, ch / ih);
      const dw = iw * ratio;
      const dh = ih * ratio;
      const dx = (cw - dw) / 2;
      const dy = (ch - dh) / 2;

      ctx.globalAlpha = 0.55;
      ctx.drawImage(overlay, 0, 0, iw, ih, dx, dy, dw, dh);
      ctx.globalAlpha = 1;
    }, []);

    const applyDicomZoom = useCallback(
      (nextScale: number) => {
        const engine = engineRef.current;
        if (!engine) return;
        try {
          const viewport = engine.getViewport(viewportId) as Types.IStackViewport;
          viewport.setZoom(nextScale);
          viewport.render();
        } catch {
          /* viewport not ready yet */
        }
      },
      [viewportId],
    );

    useEffect(() => {
      if (!src || useDicom) {
        if (!useDicom) modeRef.current = "raster";
        return;
      }

      modeRef.current = "raster";
      setHasImage(false);
    }, [src, useDicom]);

    useEffect(() => {
      if (!src || !useDicom) return;

      let cancelled = false;
      modeRef.current = "dicom";

      const loadDicom = async () => {
        const element = viewportRef.current;
        if (!element) return;

        await initCornerstone();
        if (cancelled) return;

        engineRef.current?.destroy();
        const engine = new RenderingEngine(engineId);
        engineRef.current = engine;

        engine.enableElement({
          viewportId,
          type: Enums.ViewportType.STACK,
          element,
        });

        const viewport = engine.getViewport(viewportId) as Types.IStackViewport;
        const imageId = `wadouri:${window.location.origin}${src}`;
        await viewport.setStack([imageId]);
        viewport.setZoom(1);
        viewport.render();
        if (!cancelled) {
          setHasImage(true);
          syncCanvas();
        }
      };

      void loadDicom().catch(() => {
        if (!cancelled) setHasImage(false);
      });

      return () => {
        cancelled = true;
        engineRef.current?.destroy();
        engineRef.current = null;
      };
    }, [src, useDicom, engineId, viewportId, syncCanvas]);

    useImperativeHandle(
      ref,
      () => ({
        zoomIn: () => {
          setScale((s) => {
            const next = Math.min(3, s + 0.2);
            if (modeRef.current === "dicom") applyDicomZoom(next);
            return next;
          });
        },
        zoomOut: () => {
          setScale((s) => {
            const next = Math.max(0.4, s - 0.2);
            if (modeRef.current === "dicom") applyDicomZoom(next);
            return next;
          });
        },
        resetView: () => {
          setScale(1);
          setBright(1);
          setContrast(1);
          if (modeRef.current === "dicom") applyDicomZoom(1);
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
      }),
      [applyDicomZoom, drawOverlay, syncCanvas],
    );

    useEffect(() => {
      syncCanvas();
      const ro = new ResizeObserver(() => {
        syncCanvas();
        if (showOverlayRef.current) drawOverlay();
      });
      if (wrapRef.current) ro.observe(wrapRef.current);
      return () => ro.disconnect();
    }, [drawOverlay, syncCanvas]);

    return (
      <div className="viewer-wrap" ref={wrapRef}>
        {useDicom ? (
          <div
            ref={viewportRef}
            className="viewer-cornerstone"
            style={{ width: "100%", height: 480, background: "#000" }}
          />
        ) : (
          src && (
            <img
              ref={imgRef}
              src={src}
              alt="Study image"
              style={{
                filter: `brightness(${bright}) contrast(${contrast})`,
                transform: `scale(${scale})`,
              }}
              onLoad={() => {
                modeRef.current = "raster";
                setHasImage(true);
                syncCanvas();
              }}
              onError={() => setHasImage(false)}
            />
          )
        )}
        <canvas className="viewer-overlay-canvas" ref={canvasRef} />
        {!hasImage && (
          <div className="viewer-empty">
            <div style={{ color: "var(--text-3)", fontSize: 14 }}>
              {src ? "Loading image…" : "Image will appear after upload"}
            </div>
          </div>
        )}
      </div>
    );
  },
);

DicomViewer.displayName = "DicomViewer";
