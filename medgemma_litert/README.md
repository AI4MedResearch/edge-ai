# MedGemma 1.5 4B Multimodal -> LiteRT

This folder contains a custom conversion wrapper for:

- Source model: `google/medgemma-1.5-4b-it`
- Target artifact: LiteRT-LM bundle (`model.litertlm`)

The script uses LiteRT Torch's HF export pipeline with multimodal task
`image_text_to_text` and bundles:

- text prefill/decode TFLite
- vision encoder TFLite
- vision adapter TFLite
- tokenizer + metadata

## 1) Prerequisites

1. Accept MedGemma terms on Hugging Face (`google/medgemma-1.5-4b-it`).
2. Create a HF access token with read permissions.
3. Ensure enough disk and RAM/VRAM for a 4B multimodal export.

## 2) Prepare environment

```bash
cd medgemma_litert
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

## 3) Configure auth

```bash
export HF_TOKEN='hf_xxx'
```

## 4) Dry run (recommended)

```bash
python convert_medgemma15_4b_to_litert.py --dry_run
```

## 5) Run conversion

```bash
python convert_medgemma15_4b_to_litert.py \
  --output_dir out/medgemma-1.5-4b-it-litertlm \
  --prefill_lengths 128,256,512 \
  --cache_length 4096 \
  --quantization_recipe dynamic_wi8_afp32 \
  --vision_quantization_recipe weight_only_wi8_afp32
```

Expected output:

- `out/medgemma-1.5-4b-it-litertlm/model.litertlm`

## Useful flags

- `--enable_dynamic_shape`
- `--experimental_lightweight_conversion`
- `--trust_remote_code`
- `--skip_model_info_check`
- `--hf_token` or `--hf_token_env`

## Notes

- The model is gated. Missing/unauthorized HF token will fail at model download.
- Output is LiteRT-LM (`.litertlm`), which is the current LiteRT GenAI bundle
  produced by the upstream exporter.

## 6) Run local text/chat endpoint

`serve_litert_text_chat.py` exposes:

- `GET /health`
- `POST /chat`
- `POST /v1/chat/completions` (OpenAI-style response shape)

Example:

```bash
cd medgemma_litert
/tmp/medgemma_clean_venv/bin/python serve_litert_text_chat.py \
  --host 127.0.0.1 \
  --port 8010 \
  --model /tmp/medgemma_local/model.tflite \
  --embedder /tmp/medgemma_local/embedder.tflite \
  --tokenizer /tmp/medgemma_local/tokenizer.json
```

Test:

```bash
curl -sS http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Answer one word."}],"max_tokens":24}'
```
