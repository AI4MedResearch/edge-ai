/// <reference types="vitest" />
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Custom plugin to strip invalid sourceMappingURL comments from MediaPipe
const stripSourceMapPlugin = (): Plugin => {
  return {
    name: 'strip-sourcemap',
    enforce: 'pre',
    transform(code, id) {
      if (id.includes('@mediapipe/tasks-genai')) {
        return {
          code: code.replace(/\/\/# sourceMappingURL=.*/g, ''),
          map: { mappings: '' }
        }
      }
    }
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(), 
    tailwindcss(), 
    stripSourceMapPlugin(),
    {
      name: 'serve-models',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.endsWith('.litertlm') || req.url?.endsWith('.task') || req.url?.endsWith('.tflite')) {
            res.setHeader('Content-Type', 'application/octet-stream');
          }
          next();
        });
      }
    }
  ],
  assetsInclude: ['**/*.litertlm', '**/*.bin', '**/*.tflite', '**/*.task', '**/*.json'],
  server: {
    fs: {
      allow: ['..'],
    },
    headers: {
      'Accept-Ranges': 'bytes',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    watch: {
      usePolling: true,
    },
  },
  // Ensure large model files are treated as raw assets and not compressed by Vite during dev
  assetsInclude: ['**/*.litertlm', '**/*.bin', '**/*.tflite', '**/*.task', '**/*.json'],
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-genai'],
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  },
})
