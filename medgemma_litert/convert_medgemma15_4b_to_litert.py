#!/usr/bin/env python3
"""Custom MedGemma 1.5 4B multimodal -> LiteRT-LM conversion wrapper."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path


DEFAULT_MODEL = "google/medgemma-1.5-4b-it"


def _ensure_typing_self_compat() -> None:
  """Backfill Python 3.11+ symbols for Python 3.10 runtimes."""
  import enum
  import typing

  if hasattr(typing, "Self"):
    pass
  else:
    try:
      from typing_extensions import Self as _Self
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise RuntimeError(
          "typing.Self missing and typing_extensions is unavailable."
      ) from exc
    typing.Self = _Self  # type: ignore[attr-defined]

  if hasattr(enum, "StrEnum"):
    return
  try:
    from backports.strenum import StrEnum as _StrEnum
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise RuntimeError(
        "enum.StrEnum missing and backports.strenum is unavailable."
    ) from exc
  enum.StrEnum = _StrEnum  # type: ignore[attr-defined]


def _parse_prefill_lengths(value: str) -> list[int]:
  parts = [x.strip() for x in value.split(",") if x.strip()]
  if not parts:
    raise argparse.ArgumentTypeError("prefill lengths cannot be empty")
  try:
    values = [int(x) for x in parts]
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        "prefill lengths must be comma-separated integers"
    ) from exc
  if any(v <= 0 for v in values):
    raise argparse.ArgumentTypeError("prefill lengths must be > 0")
  return sorted(set(values))


def _get_hf_token(token: str | None, token_env: str) -> str | None:
  if token:
    return token
  return os.environ.get(token_env)


def _maybe_hf_login(token: str | None) -> None:
  if not token:
    return
  from huggingface_hub import login

  login(token=token, add_to_git_credential=False)


def _print_model_metadata(model_id: str, token: str | None) -> None:
  from huggingface_hub import HfApi

  info = HfApi().model_info(model_id, token=token)
  tags = sorted(set(info.tags or []))
  config = info.config or {}
  model_type = config.get("model_type")
  architectures = config.get("architectures", [])
  print("Model metadata:")
  print(f"  id: {info.id}")
  print(f"  gated: {info.gated}")
  print(f"  pipeline_tag: {info.pipeline_tag}")
  print(f"  model_type: {model_type}")
  print(f"  architectures: {architectures}")
  print(f"  has gemma3 tag: {'gemma3' in tags}")


def _assert_litert_torch_importable() -> None:
  try:
    import litert_torch  # noqa: F401
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise RuntimeError(
        "Failed to import litert_torch. Run ./setup_env.sh first."
    ) from exc


def _patch_gemma3_vision_aliases() -> None:
  """Bridge attribute layout differences for Gemma3 vision export.

  This is a critical step because some Hugging Face Transformers versions 
  nest vision modules under `model.model.*`, whereas the `litert_torch` 
  export logic expects them at the top level of the `Gemma3ForConditionalGeneration` 
  class. This monkey-patch ensures the vision tower and projector are discoverable
  during the graph-tracing phase.
  """
  try:
    from transformers.models.gemma3 import (  # pylint: disable=import-error
        modeling_gemma3,
    )
  except Exception:
    return

  model_cls = getattr(
      modeling_gemma3, "Gemma3ForConditionalGeneration", None
  )
  if model_cls is None:
    return

  # Map nested vision attributes to the top level via properties
  if not hasattr(model_cls, "vision_tower"):
    model_cls.vision_tower = property(  # type: ignore[attr-defined]
        lambda self: self.model.vision_tower
    )
  if not hasattr(model_cls, "multi_modal_projector"):
    model_cls.multi_modal_projector = property(  # type: ignore[attr-defined]
        lambda self: self.model.multi_modal_projector
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Convert MedGemma 1.5 4B multimodal model into LiteRT-LM."
  )
  parser.add_argument("--model", default=DEFAULT_MODEL)
  parser.add_argument(
      "--output_dir",
      default="out/medgemma-1.5-4b-it-litertlm",
      help="Output directory for converted artifacts.",
  )
  parser.add_argument(
      "--hf_token",
      default=None,
      help="Hugging Face token. If omitted, --hf_token_env is used.",
  )
  parser.add_argument(
      "--hf_token_env",
      default="HF_TOKEN",
      help="Env var used when --hf_token is not provided.",
  )
  parser.add_argument(
      "--prefill_lengths",
      type=_parse_prefill_lengths,
      default=[128, 256, 512],
      help="Comma-separated prefill lengths. These define the bucket sizes for initial prompt processing.",
  )
  parser.add_argument(
      "--cache_length", 
      type=int, 
      default=4096,
      help="The maximum sequence length supported by the KV-cache.",
  )
  parser.add_argument(
      "--quantization_recipe",
      default="dynamic_wi8_afp32",
      help="Quantization strategy for the text LLM (e.g., 8-bit integer weights).",
  )
  parser.add_argument(
      "--vision_quantization_recipe",
      default="weight_only_wi8_afp32",
      help="Quantization strategy for the vision encoder component.",
  )
  parser.add_argument(
      "--enable_dynamic_shape",
      action="store_true",
      help="If true, generates TFLite signatures that support variable input lengths.",
  )
  parser.add_argument(
      "--trust_remote_code",
      action="store_true",
      help="Required for certain HF models that execute custom modeling code.",
  )
  parser.add_argument(
      "--experimental_lightweight_conversion",
      action="store_true",
      help="Enables a faster conversion path that may bypass intensive graph optimizations.",
  )
  parser.add_argument(
      "--skip_model_info_check",
      action="store_true",
      help="Skip fetching model metadata from HF Hub before conversion.",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Validate parameters and print config without executing the export.",
  )
  parser.add_argument(
      "--no_bundle",
      action="store_true",
      help="If true, generates raw .tflite files instead of a .litertlm bundle.",
  )
  parser.add_argument(
      "--externalize_embedder",
      action="store_true",
      default=True,
      help="Whether to externalize the embedder model.",
  )
  parser.add_argument(
      "--no_externalize_embedder",
      action="store_false",
      dest="externalize_embedder",
      help="Disable externalizing the embedder model.",
  )
  parser.add_argument(
      "--externalize_rope",
      action="store_true",
      default=False,
      help="Whether to externalize the RoPE embeddings.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  # 1. Handle runtime environment compatibility
  _ensure_typing_self_compat()
  
  # 2. Patch vision module paths for Gemma 3 architecture
  _patch_gemma3_vision_aliases()
  
  # 3. Ensure the core export library is available
  _assert_litert_torch_importable()

  # 4. Authenticate with Hugging Face (MedGemma is typically gated)
  token = _get_hf_token(args.hf_token, args.hf_token_env)
  if token:
    _maybe_hf_login(token)
  else:
    print(
        "WARNING: No HF token detected. This model is gated; conversion will"
        " fail unless you provide one."
    )

  # 5. Fetch and print model info to verify visibility/accessibility
  if not args.skip_model_info_check:
    _print_model_metadata(args.model, token)

  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  # 6. Define the full export configuration
  # The 'image_text_to_text' task is essential for MedGemma's multimodal nature.
  bundle_litert_lm = not args.no_bundle
  config_preview = {
      "model": args.model,
      "output_dir": str(output_dir),
      "task": "image_text_to_text",
      "prefill_lengths": args.prefill_lengths,
      "cache_length": args.cache_length,
      "quantization_recipe": args.quantization_recipe,
      "vision_encoder_quantization_recipe": args.vision_quantization_recipe,
      "enable_dynamic_shape": args.enable_dynamic_shape,
      "trust_remote_code": args.trust_remote_code,
      "bundle_litert_lm": bundle_litert_lm, # Creates the final .litertlm container
      "export_vision_encoder": True,
      "externalize_embedder": args.externalize_embedder,
      "externalize_rope": args.externalize_rope,
      "experimental_lightweight_conversion": (
          args.experimental_lightweight_conversion
      ),
  }
  print("Resolved export config:")
  for key, value in config_preview.items():
    print(f"  {key}: {value}")

  if args.dry_run:
    return 0

  # 7. Dynamically invoke the HF export library
  # We use signature inspection to maintain compatibility with different 
  # nightly builds of litert_torch.
  from litert_torch.generative.export_hf import export as export_lib

  sig = inspect.signature(export_lib.export)
  params = sig.parameters

  export_kwargs = {
      "model": args.model,
      "output_dir": str(output_dir),
      "trust_remote_code": args.trust_remote_code,
      "prefill_lengths": args.prefill_lengths,
      "cache_length": args.cache_length,
      "quantization_recipe": args.quantization_recipe,
      "enable_dynamic_shape": args.enable_dynamic_shape,
      "bundle_litert_lm": bundle_litert_lm,
      "externalize_embedder": args.externalize_embedder,
      "externalize_rope": args.externalize_rope,
  }

  # Map internal config to library-specific argument names
  if "task" in params:
    export_kwargs["task"] = "image_text_to_text"
  if "export_vision_encoder" in params:
    export_kwargs["export_vision_encoder"] = True
  if "vision_encoder_quantization_recipe" in params:
    export_kwargs["vision_encoder_quantization_recipe"] = args.vision_quantization_recipe
  if "experimental_lightweight_conversion" in params:
    export_kwargs["experimental_lightweight_conversion"] = args.experimental_lightweight_conversion

  # Prune arguments that are not supported by the current library version
  final_kwargs = {k: v for k, v in export_kwargs.items() if k in params}
  
  missing_params = [k for k in export_kwargs if k not in params]
  if missing_params:
    print(f"WARNING: The following arguments are NOT supported by your version of litert_torch and will be IGNORED: {missing_params}")
    if "task" in missing_params or "export_vision_encoder" in missing_params:
        print("CRITICAL WARNING: Multimodal support (vision) seems to be missing in this version of litert_torch. Conversion will likely only handle text.")

  # 8. Execute the conversion (This triggers graph tracing, optimization, and quantization)
  export_lib.export(**final_kwargs)

  # 9. Verify the resulting artifacts
  litert_lm_path = output_dir / "model.litertlm"
  if litert_lm_path.exists():
    print(f"SUCCESS: {litert_lm_path}")
  else:
    # Some older versions might output raw .tflite files instead of a bundle
    tflite_files = list(output_dir.glob("*.tflite"))
    if tflite_files:
        print(f"SUCCESS: Exported TFLite files: {[f.name for f in tflite_files]}")
    else:
        print("WARNING: Export finished but no model artifacts were found in output directory.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
