/**
 * Axios instance shared by all API clients.
 * All requests go through the Nginx gateway at /api/v1.
 */
import axios, { AxiosError } from "axios";

export const TOKEN_KEY = "mv_token";

export const tokenStore = {
  get: () => localStorage.getItem(TOKEN_KEY),
  save: (token: string) => localStorage.setItem(TOKEN_KEY, token),
  clear: () => localStorage.removeItem(TOKEN_KEY),
  isSet: () => !!localStorage.getItem(TOKEN_KEY),
};

export const api = axios.create({
  baseURL: "/api/v1",
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  const token = tokenStore.get();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (error: AxiosError<{ detail?: string }>) => {
    if (error.response?.status === 401) {
      tokenStore.clear();
      // Hard redirect so all in-memory state is cleared.
      if (window.location.pathname !== "/") window.location.href = "/";
    }
    return Promise.reject(error);
  },
);

/** Normalise an axios error into a human-readable message. */
export function errMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const e = error as AxiosError<{ detail?: string }>;
    return e.response?.data?.detail || e.message || "Request failed";
  }
  return error instanceof Error ? error.message : "Request failed";
}
