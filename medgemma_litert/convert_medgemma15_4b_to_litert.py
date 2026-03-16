#!/usr/bin/env python3
"""Custom MedGemma 1.5 4B multimodal -> LiteRT-LM conversion wrapper."""

from __future__ import annotations

import argparse
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

  Some transformers versions expose vision modules at `model.model.*` while
  litert_torch's Gemma3 exportables currently expect top-level attributes on
  `Gemma3ForConditionalGeneration`.
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
      help="Comma-separated prefill lengths, e.g. 128,256,512",
  )
  parser.add_argument("--cache_length", type=int, default=4096)
  parser.add_argument(
      "--quantization_recipe",
      default="dynamic_wi8_afp32",
      help="LiteRT text quantization recipe.",
  )
  parser.add_argument(
      "--vision_quantization_recipe",
      default="weight_only_wi8_afp32",
      help="LiteRT vision quantization recipe.",
  )
  parser.add_argument(
      "--enable_dynamic_shape",
      action="store_true",
      help="Enable dynamic prefill/cache shapes in exported model.",
  )
  parser.add_argument(
      "--trust_remote_code",
      action="store_true",
      help="Pass trust_remote_code=True while loading the HF model.",
  )
  parser.add_argument(
      "--experimental_lightweight_conversion",
      action="store_true",
      help="Enable lightweight conversion path if supported.",
  )
  parser.add_argument(
      "--skip_model_info_check",
      action="store_true",
      help="Skip HF model metadata check before export.",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Print resolved config and exit without conversion.",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  _ensure_typing_self_compat()
  _patch_gemma3_vision_aliases()
  _assert_litert_torch_importable()

  token = _get_hf_token(args.hf_token, args.hf_token_env)
  if token:
    _maybe_hf_login(token)
  else:
    print(
        "WARNING: No HF token detected. This model is gated; conversion will"
        " fail unless you provide one."
    )

  if not args.skip_model_info_check:
    _print_model_metadata(args.model, token)

  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

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
      "bundle_litert_lm": True,
      "export_vision_encoder": True,
      "experimental_lightweight_conversion": (
          args.experimental_lightweight_conversion
      ),
  }
  print("Resolved export config:")
  for key, value in config_preview.items():
    print(f"  {key}: {value}")

  if args.dry_run:
    return 0

  from litert_torch.generative.export_hf import export as export_lib

  export_lib.export(
      model=args.model,
      output_dir=str(output_dir),
      task="image_text_to_text",
      trust_remote_code=args.trust_remote_code,
      prefill_lengths=args.prefill_lengths,
      cache_length=args.cache_length,
      quantization_recipe=args.quantization_recipe,
      enable_dynamic_shape=args.enable_dynamic_shape,
      bundle_litert_lm=True,
      export_vision_encoder=True,
      vision_encoder_quantization_recipe=args.vision_quantization_recipe,
      experimental_lightweight_conversion=(
          args.experimental_lightweight_conversion
      ),
  )

  litert_lm_path = output_dir / "model.litertlm"
  if litert_lm_path.exists():
    print(f"SUCCESS: {litert_lm_path}")
  else:
    print(
        "WARNING: Export finished but model.litertlm was not found in output."
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
