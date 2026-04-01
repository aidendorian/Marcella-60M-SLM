import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  publicDir: 'public',   // ui/public/ → served at /
  server: {
    port: 5173,
    proxy: {
      '/generate': 'http://localhost:8000',
      '/health':   'http://localhost:8000',
    }
  }
})