import { api } from "./client";
import type { AdminUpdateUserRequest, User } from "@/types";

export const AdminAPI = {
  async listUsers(skip = 0, limit = 100): Promise<User[]> {
    const { data } = await api.get<User[]>("/admin/users", { params: { skip, limit } });
    return data;
  },

  async getUser(userId: number): Promise<User> {
    const { data } = await api.get<User>(`/admin/users/${userId}`);
    return data;
  },

  async updateUser(userId: number, body: AdminUpdateUserRequest): Promise<User> {
    const { data } = await api.patch<User>(`/admin/users/${userId}`, body);
    return data;
  },

  async deactivateUser(userId: number): Promise<void> {
    await api.delete(`/admin/users/${userId}`);
  },
};
