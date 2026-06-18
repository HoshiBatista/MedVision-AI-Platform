import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "@/store/auth";

export function ProtectedRoute() {
  const isAuthed = useAuth((s) => s.isAuthenticated());
  if (!isAuthed) return <Navigate to="/" replace />;
  return <Outlet />;
}
