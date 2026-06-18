import { create } from "zustand";

export type ToastType = "success" | "error" | "info" | "warning";

export interface ToastItem {
  id: number;
  message: string;
  type: ToastType;
  duration: number;
}

interface ToastState {
  toasts: ToastItem[];
  push: (message: string, type?: ToastType, duration?: number) => void;
  dismiss: (id: number) => void;
}

let nextId = 1;

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (message, type = "info", duration = 4000) => {
    const id = nextId++;
    set((s) => ({ toasts: [{ id, message, type, duration }, ...s.toasts] }));
    window.setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, duration);
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

/** Convenience helper usable outside React components. */
export const toast = (message: string, type?: ToastType, duration?: number) =>
  useToastStore.getState().push(message, type, duration);
