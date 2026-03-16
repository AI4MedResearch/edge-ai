#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

exec env PYTHONUNBUFFERED=1 /tmp/medgemma_clean_venv/bin/python serve_litert_text_chat.py \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8010}" \
  --model "${MODEL_PATH:-/tmp/medgemma_local/model.tflite}" \
  --embedder "${EMBEDDER_PATH:-/tmp/medgemma_local/embedder.tflite}" \
  --tokenizer "${TOKENIZER_PATH:-/tmp/medgemma_local/tokenizer.json}"
