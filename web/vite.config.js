import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `npm run dev` serves the UI on :3000 and proxies API calls to the FastAPI server on :8000.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 3000, proxy: { '/api': 'http://localhost:8000', '/stream-proxy': 'http://localhost:8000' } },
})
