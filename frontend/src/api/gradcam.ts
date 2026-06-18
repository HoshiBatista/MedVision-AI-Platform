import { api } from "./client";
import type { ExplainResponse } from "@/types";

export const GradCAMAPI = {
  async explain(
    modelName: string,
    imagePath: string,
    targetClass: number | null = null,
  ): Promise<ExplainResponse> {
    const { data } = await api.post<ExplainResponse>("/explain", {
      model_name: modelName,
      image_path: imagePath,
      target_class: targetClass,
    });
    return data;
  },
};
