import { readFileSync } from "node:fs";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [{
    name: "third-party-notices",
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "licenses/THREE-MIT.txt",
        source: readFileSync(new URL("../third_party/licenses/THREE-MIT.txt", import.meta.url)),
      });
    },
  }],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8090",
    },
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
  build: {
    outDir: "../server/webui",
    emptyOutDir: true,
  },
});
