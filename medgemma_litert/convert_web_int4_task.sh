#!/usr/bin/env bash
set -euo pipefail

# This script converts MedGemma-1.5-4b-it to a LiteRT-LM bundle
# optimized for browser deployment (WebGPU) using the .task extension.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODEL_ID="google/medgemma-1.5-4b-it"
OUTPUT_DIR="out/medgemma-web"

# Ensure environment is ready
if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Running setup_env.sh..."
  ./setup_env.sh
fi

source .venv/bin/activate

# Ensure necessary dependencies for multimodal export are present
echo "Installing/Updating dependencies..."
python -m pip install -r requirements.txt
python -m pip install torchvision

echo "🚀 Starting LiteRT-LM conversion (INT4, WebGPU optimized)..."

# Using convert_medgemma15_4b_to_litert.py which leverages litert_torch.
# This supports the multimodal vision tower which standard mediapipe converter may not.
python convert_medgemma15_4b_to_litert.py \
  --model "${MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --quantization_recipe dynamic_wi4_afp32 \
  --vision_quantization_recipe weight_only_wi4_afp32 \
  --cache_length 2048 \
  --prefill_lengths 128,256,512,1024 \
  --enable_dynamic_shape \
  --trust_remote_code \
  --experimental_lightweight_conversion

echo "✅ Conversion finished."
echo "Moving model to web application public directory..."
mkdir -p ../mri-viewer/public/models
cp "${OUTPUT_DIR}/model.litertlm" ../mri-viewer/public/models/medgemma.task

echo "Artifact available in: ${ROOT_DIR}/${OUTPUT_DIR}/model.litertlm"
echo "Web asset deployed to: ../mri-viewer/public/models/medgemma.task"
