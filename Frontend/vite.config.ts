import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import {defineConfig} from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    // Listen on all interfaces so a phone on the same LAN can open the dev server.
    host: '0.0.0.0',
  },
});
