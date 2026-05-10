import { defineConfig, type Plugin } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { cpSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

function copyPublicWithoutLargeModels() {
  const publicDir = join(import.meta.dirname, 'public')
  const distDir = join(import.meta.dirname, 'dist')
  const excluded = new Set([
    'models/litertlm_medgemma-1.5-4b-it-fp8.litertlm',
    'models/gemma-3n-E2B-it-int4-Web.litertlm',
  ])

  const copy = (sourceDir: string) => {
    for (const entry of readdirSync(sourceDir)) {
      const source = join(sourceDir, entry)
      const relativePath = relative(publicDir, source)
      if (excluded.has(relativePath)) continue

      const target = join(distDir, relativePath)
      const stat = statSync(source)
      if (stat.isDirectory()) {
        mkdirSync(target, { recursive: true })
        copy(source)
      } else {
        mkdirSync(join(target, '..'), { recursive: true })
        cpSync(source, target)
      }
    }
  }

  mkdirSync(distDir, { recursive: true })
  copy(publicDir)
}

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
    ,
    {
      name: 'copy-public-without-large-models',
      apply: 'build',
      closeBundle() {
        copyPublicWithoutLargeModels()
      },
    },
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
  build: {
    copyPublicDir: false,
  },
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-genai'],
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
  },
})
