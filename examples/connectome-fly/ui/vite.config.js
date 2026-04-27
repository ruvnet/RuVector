import { defineConfig } from 'vite';

// BASE is set by CI / manual build so the same repo can target:
//   - `npm run dev`           → /          (local dev)
//   - `VITE_BASE=/Connectome-OS/ npm run build` → /Connectome-OS/
//     (GitHub Pages at ruvnet.github.io/Connectome-OS/)
const base = process.env.VITE_BASE || '/';

export default defineConfig({
  root: '.',
  base,
  publicDir: 'public',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:5174',
        changeOrigin: false,
        ws: false,
      },
    },
  },
});
