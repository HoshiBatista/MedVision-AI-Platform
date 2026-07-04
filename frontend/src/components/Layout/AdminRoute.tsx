import { useEffect } from "react";
import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "@/store/auth";

export function AdminRoute() {
  const user = useAuth((s) => s.user);
  const loadUser = useAuth((s) => s.loadUser);
  const isAuthed = useAuth((s) => s.isAuthenticated());

  useEffect(() => {
    if (!user) void loadUser();
  }, [user, loadUser]);

  if (!isAuthed) return <Navigate to="/" replace />;
  if (!user) {
    return (
      <div className="empty-state" style={{ minHeight: "100vh" }}>
        <div className="spinner spinner-lg" />
      </div>
    );
  }
  if (user.role !== "admin") return <Navigate to="/upload" replace />;
  return <Outlet />;
}
