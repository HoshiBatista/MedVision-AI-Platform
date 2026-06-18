import { api } from "./client";
import type { JobQueuedResponse, JobResult, Task } from "@/types";

export const AnalysisAPI = {
  async analyze(
    studyId: string,
    task: Task,
    config: Record<string, unknown> = {},
  ): Promise<JobQueuedResponse> {
    const { data } = await api.post<JobQueuedResponse>("/analyze", {
      study_id: studyId,
      task,
      config,
    });
    return data;
  },

  async getResult(jobId: string): Promise<JobResult> {
    const { data } = await api.get<JobResult>(`/results/${jobId}`);
    return data;
  },
};
