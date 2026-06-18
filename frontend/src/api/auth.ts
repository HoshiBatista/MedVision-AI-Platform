import { api } from "./client";
import type { LoginResponse, User } from "@/types";

export const AuthAPI = {
  async login(username: string, password: string): Promise<LoginResponse> {
    const form = new URLSearchParams({ username, password });
    const { data } = await api.post<LoginResponse>("/auth/login", form, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    return data;
  },

  async logout(): Promise<void> {
    await api.post("/auth/logout");
  },

  async me(): Promise<User> {
    const { data } = await api.get<User>("/users/me");
    return data;
  },
};
