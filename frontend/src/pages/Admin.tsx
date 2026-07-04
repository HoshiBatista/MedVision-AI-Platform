import { useEffect, useState } from "react";
import { AppLayout } from "@/components/Layout/AppLayout";
import { AdminAPI } from "@/api/admin";
import { errMessage } from "@/api/client";
import { fmtDate } from "@/lib/format";
import { toast } from "@/store/toast";
import type { AdminUpdateUserRequest, User, UserRole } from "@/types";

const ROLES: UserRole[] = ["admin", "user", "radiologist"];

interface EditState {
  full_name: string;
  role: UserRole;
  is_active: boolean;
}

function rowEdit(user: User): EditState {
  return {
    full_name: user.full_name || "",
    role: (user.role as UserRole) || "user",
    is_active: user.is_active !== false,
  };
}

export function Admin() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [edits, setEdits] = useState<Record<number, EditState>>({});
  const [savingId, setSavingId] = useState<number | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const list = await AdminAPI.listUsers();
      setUsers(list);
      setEdits(Object.fromEntries(list.map((u) => [u.id, rowEdit(u)])));
    } catch (err) {
      toast(errMessage(err), "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const patchEdit = (id: number, patch: Partial<EditState>) => {
    setEdits((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
  };

  const saveUser = async (user: User) => {
    const edit = edits[user.id];
    if (!edit) return;
    setSavingId(user.id);
    try {
      const body: AdminUpdateUserRequest = {
        full_name: edit.full_name.trim() || null,
        role: edit.role,
        is_active: edit.is_active,
      };
      const updated = await AdminAPI.updateUser(user.id, body);
      setUsers((prev) => prev.map((u) => (u.id === user.id ? updated : u)));
      setEdits((prev) => ({ ...prev, [user.id]: rowEdit(updated) }));
      toast("User updated", "success");
    } catch (err) {
      toast(errMessage(err), "error");
    } finally {
      setSavingId(null);
    }
  };

  const deactivate = async (user: User) => {
    if (!window.confirm(`Deactivate ${user.email}?`)) return;
    setSavingId(user.id);
    try {
      await AdminAPI.deactivateUser(user.id);
      await load();
      toast("User deactivated", "success");
    } catch (err) {
      toast(errMessage(err), "error");
    } finally {
      setSavingId(null);
    }
  };

  return (
    <AppLayout title="Admin — Users">
      <div className="card fade-in">
        <div className="card-header">
          <span className="card-icon"></span>
          <h2>User management</h2>
          <div style={{ flex: 1 }} />
          <span className="badge badge-purple">{users.length} users</span>
        </div>
        <div className="card-body">
          {loading ? (
            <div className="empty-state" style={{ padding: 40 }}>
              <div className="spinner spinner-lg" />
            </div>
          ) : (
            <div className="admin-table-wrap">
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Email</th>
                    <th>Full name</th>
                    <th>Role</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => {
                    const edit = edits[user.id] || rowEdit(user);
                    const busy = savingId === user.id;
                    return (
                      <tr key={user.id}>
                        <td className="mono" style={{ fontSize: 12 }}>
                          {user.email}
                        </td>
                        <td>
                          <input
                            className="form-input form-input-sm"
                            value={edit.full_name}
                            onChange={(e) => patchEdit(user.id, { full_name: e.target.value })}
                          />
                        </td>
                        <td>
                          <select
                            className="form-select form-input-sm"
                            value={edit.role}
                            onChange={(e) =>
                              patchEdit(user.id, { role: e.target.value as UserRole })
                            }
                          >
                            {ROLES.map((r) => (
                              <option key={r} value={r}>
                                {r}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td>
                          <label className="flex items-center gap-8" style={{ fontSize: 13 }}>
                            <input
                              type="checkbox"
                              checked={edit.is_active}
                              onChange={(e) =>
                                patchEdit(user.id, { is_active: e.target.checked })
                              }
                            />
                            {edit.is_active ? (
                              <span className="badge badge-green">Active</span>
                            ) : (
                              <span className="badge badge-red">Inactive</span>
                            )}
                          </label>
                        </td>
                        <td style={{ fontSize: 12 }}>{fmtDate(user.created_at)}</td>
                        <td>
                          <div className="flex gap-8">
                            <button
                              className="btn btn-primary btn-sm"
                              disabled={busy}
                              onClick={() => void saveUser(user)}
                            >
                              {busy ? "…" : "Save"}
                            </button>
                            <button
                              className="btn btn-secondary btn-sm"
                              disabled={busy || !edit.is_active}
                              onClick={() => void deactivate(user)}
                            >
                              Deactivate
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
