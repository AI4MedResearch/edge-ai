#!/usr/bin/env bash
set -euo pipefail

# This script converts the local MedGemma model in models/medgemma-hf/
# to a LiteRT-LM TFLite/task bundle optimized for browser deployment (WebGPU)
# with quantization targeting < 3 GB size.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

LOCAL_MODEL_DIR="models/medgemma-hf"
OUTPUT_DIR="out/medgemma-local-tflite"

# Ensure environment is ready
if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Running setup_env.sh..."
  ./setup_env.sh
fi

source .venv/bin/activate

echo "🚀 Starting LiteRT-LM conversion from local model (INT4, WebGPU optimized)..."

# Convert local model
python convert_medgemma15_4b_to_litert.py \
  --model "${LOCAL_MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --quantization_recipe dynamic_wi4_afp32 \
  --vision_quantization_recipe weight_only_wi4_afp32 \
  --cache_length 2048 \
  --prefill_lengths 128,256,512,1024 \
  --trust_remote_code \
  --experimental_lightweight_conversion \
  --skip_model_info_check

echo "✅ Conversion finished."
echo "Artifact available in: ${ROOT_DIR}/${OUTPUT_DIR}"

# Rename output to .tflite for clarity as requested
if [[ -f "${OUTPUT_DIR}/model.litertlm" ]]; then
    cp "${OUTPUT_DIR}/model.litertlm" "${OUTPUT_DIR}/medgemma-1.5-4b-it-int4.tflite"
    echo "Copied model.litertlm to ${OUTPUT_DIR}/medgemma-1.5-4b-it-int4.tflite"
fi
