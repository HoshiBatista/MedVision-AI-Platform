import { init as csRenderInit } from "@cornerstonejs/core";
import cornerstoneDICOMImageLoader from "@cornerstonejs/dicom-image-loader";

let initialized = false;

export async function initCornerstone(): Promise<void> {
  if (initialized) return;
  await csRenderInit();
  cornerstoneDICOMImageLoader.init({ maxWebWorkers: 1 });
  initialized = true;
}
