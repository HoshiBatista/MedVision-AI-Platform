/**
 * Canvas drawing helpers for CAM heatmaps and detection boxes.
 * Ported from the original heatmap_overlay.js.
 */
import type { Finding } from "@/types";

const JET: Uint8Array = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    lut[i * 3 + 0] = Math.round(Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3))) * 255);
    lut[i * 3 + 1] = Math.round(Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2))) * 255);
    lut[i * 3 + 2] = Math.round(Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1))) * 255);
  }
  return lut;
})();

/** Draw a CAM float array ([0,1]) onto a canvas, scaled to its size. */
export function drawCAM(
  canvas: HTMLCanvasElement,
  camData: number[] | Float32Array,
  camH: number,
  camW: number,
  alpha = 0.5,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const tmp = document.createElement("canvas");
  tmp.width = camW;
  tmp.height = camH;
  const tctx = tmp.getContext("2d");
  if (!tctx) return;

  const imgd = tctx.createImageData(camW, camH);
  for (let i = 0; i < camH * camW; i++) {
    const v = Math.max(0, Math.min(1, camData[i]));
    const idx = Math.round(v * 255);
    imgd.data[i * 4 + 0] = JET[idx * 3 + 0];
    imgd.data[i * 4 + 1] = JET[idx * 3 + 1];
    imgd.data[i * 4 + 2] = JET[idx * 3 + 2];
    imgd.data[i * 4 + 3] = Math.round(v * 220);
  }
  tctx.putImageData(imgd, 0, 0);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = 1;
}

const BOX_COLORS = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ef4444"];

/** Draw detection boxes (in original-image coords) scaled to the canvas. */
export function drawBoxes(
  canvas: HTMLCanvasElement,
  findings: Finding[],
  origW: number,
  origH: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const scaleX = canvas.width / origW;
  const scaleY = canvas.height / origH;
  ctx.lineWidth = 2;

  findings.forEach((f, i) => {
    const [x1, y1, x2, y2] = f.bbox;
    const sx1 = x1 * scaleX;
    const sy1 = y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;
    const color = BOX_COLORS[i % BOX_COLORS.length];

    ctx.strokeStyle = color;
    ctx.fillStyle = color + "22";
    ctx.strokeRect(sx1, sy1, sw, sh);
    ctx.fillRect(sx1, sy1, sw, sh);

    const label = `#${i + 1} ${(f.confidence * 100).toFixed(0)}%`;
    ctx.font = "bold 12px Inter, sans-serif";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(sx1, sy1 - 18, tw + 10, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, sx1 + 5, sy1 - 4);
  });
}

export function clearCanvas(canvas: HTMLCanvasElement | null): void {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx?.clearRect(0, 0, canvas.width, canvas.height);
}
