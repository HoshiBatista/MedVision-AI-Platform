import { api } from "./client";
import type { Modality, Study, StudyListResponse, UploadResponse } from "@/types";

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

  async listStudies(limit = 50, offset = 0): Promise<StudyListResponse> {
    const { data } = await api.get<StudyListResponse>("/upload", { params: { limit, offset } });
    return data;
  },

  async getStudy(studyId: string): Promise<Study> {
    const { data } = await api.get<Study>(`/upload/${studyId}`);
    return data;
  },
};
