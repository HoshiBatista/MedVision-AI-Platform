import { useState, type FormEvent } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { useAuth } from "@/store/auth";
import { errMessage } from "@/api/client";

export function Login() {
  const navigate = useNavigate();
  const isAuthed = useAuth((s) => s.isAuthenticated());
  const login = useAuth((s) => s.login);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (isAuthed) return <Navigate to="/upload" replace />;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password) return;
    setError(null);
    setLoading(true);
    try {
      await login(username.trim(), password);
      navigate("/upload", { replace: true });
    } catch (err) {
      setError(errMessage(err));
      setPassword("");
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo">
          <div className="login-icon"></div>
          <h1 className="gradient-text">MedVision AI</h1>
          <p className="login-tagline">Medical Imaging Intelligence Platform</p>
        </div>

        <form className="login-form" onSubmit={handleSubmit} noValidate>
          <div className="form-group">
            <label className="form-label" htmlFor="username">
              Username
            </label>
            <div className="input-wrap">
              <span className="input-icon"></span>
              <input
                className="form-input"
                id="username"
                type="text"
                placeholder="admin"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password">
              Password
            </label>
            <div className="input-wrap">
              <span className="input-icon"></span>
              <input
                className="form-input"
                id="password"
                type="password"
                placeholder="••••••••"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          {error && (
            <div className="alert alert-error" style={{ marginTop: 4 }}>
              <span className="alert-icon"></span>
              <span>{error}</span>
            </div>
          )}

          <button
            className="btn btn-primary btn-lg btn-full"
            type="submit"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" /> Signing in…
              </>
            ) : (
              "Sign In"
            )}
          </button>
        </form>

        <div className="login-footer">
          <strong>Demo credentials:</strong> admin / admin
          <br />
          MedVision AI v1.0 — Research Use Only
        </div>
      </div>
    </div>
  );
}
