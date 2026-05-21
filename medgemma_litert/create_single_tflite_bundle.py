#!/usr/bin/env python3
"""Create a single raw TFLite model from component TFLite files for MediaPipe GenAI Web usage."""

import os
import sys
from mediapipe.tasks.python.genai import bundler
from mediapipe.tasks.python.genai.bundler import BundleConfig

def main():
    # Paths to individual components (assumes they exist in the output dir)
    base_dir = "out/medgemma-local-raw-tflite"
    # The components are typically in a temporary directory inside out/
    # We look for the most recent tmp dir if not found in base_dir
    if not os.path.exists(os.path.join(base_dir, "model_quantized.tflite")):
        # Try to find the components in the raw conversion output
        tmp_dirs = [d for d in os.listdir(base_dir) if d.startswith("tmp") and os.path.isdir(os.path.join(base_dir, d))]
        if tmp_dirs:
            base_dir = os.path.join(base_dir, tmp_dirs[-1]) # Use latest tmp dir
    
    print(f"🚀 Using component base directory: {base_dir}")

    config = BundleConfig(
        tflite_model=os.path.join(base_dir, "model_quantized.tflite"),
        tokenizer_model=os.path.join(base_dir, "tokenizer.model"),
        start_token="<bos>",
        stop_tokens=["<eos>", "<end_of_turn>"],
        output_filename="out/medgemma-local-raw-tflite/medgemma-1.5-4b-it-int4-single.tflite",
        tflite_embedder=os.path.join(base_dir, "embedder_quantized.tflite"),
        tflite_vision_encoder=os.path.join(base_dir, "vision_encoder_quantized.tflite"),
        tflite_vision_adapter=os.path.join(base_dir, "vision_adapter_quantized.tflite")
    )

    print("🚀 Creating single raw TFLite file using MediaPipe Bundler...")
    # The bundler creates a .task file, we will rename it to .tflite afterward
    bundler.create_bundle(config)
    
    task_file = config.output_filename + ".task"
    final_file = config.output_filename
    if os.path.exists(task_file):
        if os.path.exists(final_file):
            os.remove(final_file)
        os.rename(task_file, final_file)
        print(f"✅ Successfully created single raw TFLite: {final_file}")
    else:
        print(f"⚠️  Warning: Expected {task_file} not found. Check bundler output.")

if __name__ == "__main__":
    main()
