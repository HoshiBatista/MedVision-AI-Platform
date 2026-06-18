import { create } from "zustand";
import { tokenStore } from "@/api/client";
import { AuthAPI } from "@/api/auth";
import type { User } from "@/types";

interface AuthState {
  token: string | null;
  user: User | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  loadUser: () => Promise<void>;
  isAuthenticated: () => boolean;
}

export const useAuth = create<AuthState>((set, get) => ({
  token: tokenStore.get(),
  user: null,

  async login(username, password) {
    const res = await AuthAPI.login(username, password);
    tokenStore.save(res.access_token);
    set({ token: res.access_token });
  },

  async logout() {
    try {
      await AuthAPI.logout();
    } catch {
      // ignore — clear locally regardless
    }
    tokenStore.clear();
    set({ token: null, user: null });
  },

  async loadUser() {
    try {
      const user = await AuthAPI.me();
      set({ user });
    } catch {
      // non-fatal; sidebar just shows placeholder
    }
  },

  isAuthenticated: () => !!get().token,
}));
