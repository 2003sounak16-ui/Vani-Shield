import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env from project root directory (..)
  const rootDir = path.resolve(__dirname, '..')
  const env = loadEnv(mode, rootDir, '')

  const mapboxToken = env.MAPBOX_ACCESS_TOKEN || env.VITE_MAPBOX_ACCESS_TOKEN || process.env.MAPBOX_ACCESS_TOKEN || process.env.VITE_MAPBOX_ACCESS_TOKEN || ''
  const backendUrl = env.BACKEND_URL || process.env.BACKEND_URL || 'http://127.0.0.1:8000'
  const backendWsUrl = env.BACKEND_WS_URL || process.env.BACKEND_WS_URL || backendUrl.replace(/^http/, 'ws')

  return {
    plugins: [react()],
    envDir: '../',
    envPrefix: ['VITE_', 'MAPBOX_'],
    define: {
      'import.meta.env.VITE_MAPBOX_ACCESS_TOKEN': JSON.stringify(mapboxToken),
      'import.meta.env.MAPBOX_ACCESS_TOKEN': JSON.stringify(mapboxToken),
    },
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/ws': {
          target: backendWsUrl,
          ws: true,
        }
      }
    }
  }
})
