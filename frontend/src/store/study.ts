import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { Modality } from "@/types";

interface CurrentJob {
  id: string;
  study_id: string;
  modality: Modality;
}

interface StudyState {
  currentJob: CurrentJob | null;
  currentReportId: string | null;
  setCurrentJob: (job: CurrentJob) => void;
  setCurrentReportId: (id: string) => void;
}

/**
 * Cross-page state that mirrors the original sessionStorage usage
 * (`mv_current_job` / `mv_current_report`).
 */
export const useStudyStore = create<StudyState>()(
  persist(
    (set) => ({
      currentJob: null,
      currentReportId: null,
      setCurrentJob: (job) => set({ currentJob: job }),
      setCurrentReportId: (id) => set({ currentReportId: id }),
    }),
    {
      name: "mv_study",
      storage: createJSONStorage(() => sessionStorage),
    },
  ),
);
