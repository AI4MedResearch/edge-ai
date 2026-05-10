from litert_torch.generative.export_hf import export as export_lib

export_kwargs = {
    "model": "google/medgemma-1.5-4b-it",
    "output_dir": str(output_dir),
    "trust_remote_code": True,
    "prefill_lengths":[128, 256, 512],
    "cache_length": 4096,
    "quantization_recipe": "dynamic_wi8_afp32",
    "enable_dynamic_shape": True,
    "task": "image_text_to_text",
    "bundle_litert_lm": True,
    "export_vision_encoder": True,
    "vision_encoder_quantization_recipe": "weight_only_wi8_afp32",
    "experimental_lightweight_conversion": True
}

# There may be necessary steps to re-adjust the export_kwargs for a specific model or version of the export library. For example, if the export library has updated and changed some of the argument names or added new arguments, may need to modify the export_kwargs accordingly. Always refer to the latest documentation of the export library for any changes or updates that may affect your export process.

export_lib.export(**export_kwargs)
