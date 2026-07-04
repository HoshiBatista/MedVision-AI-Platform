import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ToastContainer } from "@/components/Toast/ToastContainer";
import { AdminRoute } from "@/components/Layout/AdminRoute";
import { ProtectedRoute } from "@/components/Layout/ProtectedRoute";
import { Admin } from "@/pages/Admin";
import { Analysis } from "@/pages/Analysis";
import { History } from "@/pages/History";
import { Login } from "@/pages/Login";
import { Report } from "@/pages/Report";
import { Upload } from "@/pages/Upload";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/upload" element={<Upload />} />
          <Route path="/history" element={<History />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/report" element={<Report />} />
        </Route>
        <Route element={<AdminRoute />}>
          <Route path="/admin" element={<Admin />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ToastContainer />
    </BrowserRouter>
  );
}
