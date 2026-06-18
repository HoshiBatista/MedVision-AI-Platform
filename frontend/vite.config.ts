import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// In dev, /api/v1 and /static/* are proxied to the gateway (Nginx on :80).
// Override with VITE_API_PROXY when the gateway runs elsewhere.
const proxyTarget = process.env.VITE_API_PROXY ?? "http://localhost";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 3000,
    host: true,
    proxy: {
      "/api/v1": { target: proxyTarget, changeOrigin: true },
      "/static": { target: proxyTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
