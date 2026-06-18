import { useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "@/store/auth";
import { capitalize } from "@/lib/format";

const NAV = [
  { to: "/upload", label: "Upload" },
  { to: "/analysis", label: "Analysis" },
  { to: "/report", label: "Reports" },
];

export function Sidebar() {
  const navigate = useNavigate();
  const user = useAuth((s) => s.user);
  const loadUser = useAuth((s) => s.loadUser);
  const logout = useAuth((s) => s.logout);

  useEffect(() => {
    if (!user) void loadUser();
  }, [user, loadUser]);

  const displayName = user?.full_name || user?.email || "Loading…";
  const avatar = (user?.full_name || user?.email || "?")[0].toUpperCase();

  const handleLogout = async () => {
    await logout();
    navigate("/", { replace: true });
  };

  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon"></div>
        <div>
          <div className="brand">MedVision AI</div>
          <div className="tagline">Imaging Platform</div>
        </div>
      </div>

      <div className="sidebar-nav">
        <div className="nav-section-label">Workspace</div>
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
          >
            {item.label}
          </NavLink>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="user-card" onClick={handleLogout} title="Click to sign out">
          <div className="avatar">{avatar}</div>
          <div>
            <div className="user-name">{displayName}</div>
            <div className="user-role">{capitalize(user?.role)}</div>
          </div>
        </div>
      </div>
    </nav>
  );
}
