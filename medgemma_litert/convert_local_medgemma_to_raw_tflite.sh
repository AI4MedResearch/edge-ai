#!/usr/bin/env bash
set -euo pipefail

# This script converts the local MedGemma model in models/medgemma-hf/
# to RAW TFLite files (not a bundle) optimized for browser deployment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

LOCAL_MODEL_DIR="models/medgemma-hf"
OUTPUT_DIR="out/medgemma-local-raw-tflite"

# Ensure environment is ready
if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Running setup_env.sh..."
  ./setup_env.sh
fi

source .venv/bin/activate

echo "🚀 Starting raw TFLite conversion from local model (INT4)..."

# Convert local model to raw TFLite files
python convert_medgemma15_4b_to_litert.py \
  --model "${LOCAL_MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --quantization_recipe dynamic_wi4_afp32 \
  --vision_quantization_recipe weight_only_wi4_afp32 \
  --cache_length 2048 \
  --prefill_lengths 128,256,512,1024 \
  --trust_remote_code \
  --experimental_lightweight_conversion \
  --skip_model_info_check \
  --no_bundle

echo "✅ Raw conversion finished."
echo "Raw TFLite components available in: ${ROOT_DIR}/${OUTPUT_DIR}"

echo "🚀 Bundling components into a single TFLite file for MediaPipe GenAI Web..."
python create_single_tflite_bundle.py

echo "✨ All-in-one raw TFLite model is ready."

