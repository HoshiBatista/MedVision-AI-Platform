import { api } from "./client";
import type { Report, ReportGenerateRequest, ReportGenerateResponse } from "@/types";

export const ReportAPI = {
  async generate(body: ReportGenerateRequest): Promise<ReportGenerateResponse> {
    const { data } = await api.post<ReportGenerateResponse>("/reports/generate", body);
    return data;
  },

  async getReport(reportId: string): Promise<Report> {
    const { data } = await api.get<Report>(`/reports/${reportId}`);
    return data;
  },

  async listForStudy(studyId: string): Promise<Report[]> {
    const { data } = await api.get<Report[]>(`/reports/study/${studyId}`);
    return data;
  },
};
