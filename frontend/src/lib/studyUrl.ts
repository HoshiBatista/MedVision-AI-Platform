/** Map a server-side study path to a gateway static URL. */
export function studyStaticUrl(filePath: string): string {
  const marker = "/studies/";
  const idx = filePath.indexOf(marker);
  if (idx >= 0) {
    return `/static/studies/${filePath.slice(idx + marker.length)}`;
  }
  return filePath;
}

export function isDicomPath(path: string): boolean {
  const lower = path.toLowerCase();
  return lower.endsWith(".dcm") || lower.endsWith(".dicom");
}

export function heatmapStaticUrl(filePath: string): string {
  const marker = "/heatmaps/";
  const idx = filePath.indexOf(marker);
  if (idx >= 0) {
    return `/static/heatmaps/${filePath.slice(idx + marker.length)}`;
  }
  const base = filePath.split("/").pop();
  return base ? `/static/heatmaps/${base}` : filePath;
}
