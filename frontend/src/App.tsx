import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ProtectedRoute } from "@/components/Layout/ProtectedRoute";
import { ToastContainer } from "@/components/Toast/ToastContainer";
import { Login } from "@/pages/Login";
import { Upload } from "@/pages/Upload";
import { Analysis } from "@/pages/Analysis";
import { Report } from "@/pages/Report";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route element={<ProtectedRoute />}>
          <Route path="/upload" element={<Upload />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/report" element={<Report />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ToastContainer />
    </BrowserRouter>
  );
}
