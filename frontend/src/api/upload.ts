import { api } from "./client";
import type { Modality, UploadResponse } from "@/types";

export const UploadAPI = {
  async uploadFile(file: File, modality: Modality): Promise<UploadResponse> {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("modality", modality);
    const { data } = await api.post<UploadResponse>("/upload", fd, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },
};
