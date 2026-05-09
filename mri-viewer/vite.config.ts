import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { cpSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

function copyPublicWithoutLargeModels() {
  const publicDir = join(import.meta.dirname, 'public')
  const distDir = join(import.meta.dirname, 'dist')
  const excluded = new Set([
    'models/litertlm_medgemma-1.5-4b-it-int4.litertlm',
    'models/gemma-4-E2B-it.litertlm',
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

const stripSourceMapPlugin = (): Plugin => ({
  name: 'strip-mediapipe-sourcemap',
  enforce: 'pre',
  transform(code, id) {
    if (id.includes('@mediapipe/tasks-genai')) {
      return {
        code: code.replace(/\/\/# sourceMappingURL=.*/g, ''),
        map: { mappings: '' },
      }
    }
  },
})

const serveModelAssetsPlugin = (): Plugin => ({
  name: 'serve-model-assets',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url?.match(/\.(litertlm|task|tflite)(\?.*)?$/)) {
        res.setHeader('Content-Type', 'application/octet-stream')
      }
      next()
    })
  },
})

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    stripSourceMapPlugin(),
    serveModelAssetsPlugin(),
    {
      name: 'copy-public-without-large-models',
      apply: 'build',
      closeBundle() {
        copyPublicWithoutLargeModels()
      },
    },
  ],
  server: {
    headers: {
      'Accept-Ranges': 'bytes',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  assetsInclude: ['**/*.litertlm', '**/*.bin', '**/*.tflite', '**/*.task', '**/*.json'],
  optimizeDeps: {
    include: [
      '@cornerstonejs/dicom-image-loader',
      '@cornerstonejs/codec-charls/decodewasmjs',
      '@cornerstonejs/codec-libjpeg-turbo-8bit/decodewasmjs',
      '@cornerstonejs/codec-openjpeg/decodewasmjs',
      '@cornerstonejs/codec-openjph/wasmjs',
    ],
    exclude: ['@mediapipe/tasks-genai'],
  },
  // Ensure wasm files are handled
  build: {
    copyPublicDir: false,
    commonjsOptions: {
      include: [/node_modules/],
    },
  },
})
