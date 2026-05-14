#!/usr/bin/env bash
set -euo pipefail

# This script converts MedGemma-1.5-4b-it to a 4-bit (INT4) "-Web" profile 
# optimized for browser deployment via MediaPipe / WebGPU.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Ensure environment is ready
if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Running setup_env.sh..."
  ./setup_env.sh
fi

source .venv/bin/activate

echo "🚀 Starting MedGemma-1.5-4b-it INT4-Web conversion..."

# Optimized parameters for WebGPU/MediaPipe GenAI Web:
# - dynamic_wi4_afp32: 4-bit weights, float32 activations (ideal for WebGPU VRAM limits).
# - cache_length 2048: Balances context window with KV-cache VRAM usage.
# - prefill_lengths: Optimized buckets for typical medical chat queries.

python convert_medgemma15_4b_to_litert.py \
  --model ai4med-id/medgemma-1.5-4b-it-litertlm \
  --output_dir out/medgemma-1.5-4b-it-int4-Web \
  --quantization_recipe dynamic_wi4_afp32 \
  --vision_quantization_recipe weight_only_wi4_afp32 \
  --cache_length 2048 \
  --prefill_lengths 128,256,512,1024 \
  --enable_dynamic_shape \
  --trust_remote_code

echo "✅ Conversion script finished."
echo "Artifacts available in: ${ROOT_DIR}/out/medgemma-1.5-4b-it-int4-Web"
echo "Bundle file: litertlm_medgemma-1.5-4b-it-int4-Web.litertlm"
