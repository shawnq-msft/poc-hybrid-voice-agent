import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  publicDir: false,
  build: {
    outDir: "web",
    emptyOutDir: false,
    sourcemap: false,
    lib: {
      entry: "frontend/src/main.jsx",
      formats: ["es"],
      fileName: () => "react-ui.js",
    },
    rollupOptions: {
      output: {
        assetFileNames: "react-[name][extname]",
      },
    },
  },
});