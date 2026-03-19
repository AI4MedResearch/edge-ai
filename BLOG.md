# On-Device Medical Intelligence: Converting MedGemma 1.5 4B to LiteRT

Medical AI is rapidly shifting from massive cloud-based models to specialized, on-device intelligence. **MedGemma 1.5 4B**, a domain-tuned version of Google's Gemma model, represents a significant leap for medical-specific text and image understanding.

In this guide, we’ll walk through how to convert the **MedGemma 1.5 4B Multimodal** model into the **LiteRT** (`.litertlm`) format, enabling high-performance, private, and efficient medical AI on edge devices.

---

## Deep Dive: MedGemma 1.5 4B

Released as a major update to the MedGemma family, the **1.5 4B** model is built on the cutting-edge **Gemma 3** architecture. While previous iterations established a strong baseline for medical text processing, the 1.5 4B version is a specialized multimodal engine designed for the complexities of modern clinical data.

### What makes 1.5 4B different?

The leap from MedGemma 1 to 1.5 is primarily defined by its shift from 2D-centric vision to **high-dimensional and longitudinal reasoning**.

*   **3D Medical Imaging**: Unlike its predecessors which were largely limited to 2D representations (X-rays, dermoscopy), MedGemma 1.5 can natively interpret 3D volumes from **CT and MRI scans**.
*   **Longitudinal Analysis**: One of the most critical clinical tasks is comparing "prior vs. current" scans. MedGemma 1.5 is optimized for time-series reviews, allowing it to track disease progression or treatment response over multiple historical records.
*   **Whole-Slide Histopathology (WSI)**: It can process high-resolution pathology slides by analyzing multiple patches simultaneously, a feat that previously required separate, disconnected pipelines.
*   **Precision Localization**: The model has seen a massive jump in anatomical localization (identifying exactly *where* a nodule is). Benchmarks show an improvement from ~3% IoU in version 1 to **~38% IoU** in version 1.5.

### Performance Comparison

| Capability | MedGemma 1 (4B) | MedGemma 1.5 (4B) |
| :--- | :--- | :--- |
| **Base Architecture** | Gemma 2 | **Gemma 3** |
| **Imaging Support** | 2D Focus (X-ray) | **3D (CT/MRI) & WSI** |
| **Temporal Reasoning** | Limited / Single-scan | **Longitudinal Tracking** |
| **EHR Question Answering**| ~68% Accuracy | **~90% Accuracy** |
| **Anatomy Localization** | ~3% IoU | **~38% IoU** |

By bringing this level of intelligence to the edge via LiteRT, we are enabling tools that can assist radiologists and clinicians in real-time, even in bandwidth-constrained or privacy-sensitive environments.

---

## Why MedGemma on LiteRT?

**MedGemma 1.5 4B** is designed for clinical reasoning, answering medical questions, and processing multimodal inputs (like medical imaging). By converting it to **LiteRT** (formerly TensorFlow Lite), we unlock:

1.  **Low Latency**: Faster response times for critical clinical workflows.
2.  **Privacy & Security**: Sensitive medical data stays on the device.
3.  **Reduced Cost**: No need for expensive cloud GPU infrastructure for inference.
4.  **Accessibility**: Runs on mobile devices, medical tablets, and edge servers.

---

## Understanding the .litertlm Format

As we move towards standardized on-device GenAI, the **.litertlm** format emerges as LiteRT’s specialized bundle for Large Language Models. But how does it compare to other common edge formats like GGUF or ONNX?

### What is a .litertlm bundle?
Unlike a standard `.tflite` file which typically represents a single computational graph, a `.litertlm` file is a **container**. It bundles together:
*   **Prefill & Decode Graphs**: Optimized paths for processing initial prompts vs. generating tokens one-by-one.
*   **Embedders**: The logic for turning text into numerical vectors.
*   **Multimodal Components**: Vision encoders (like those needed for MedGemma) and adapters.
*   **Metadata**: Tokenizer configurations and model hyperparameters.

### Comparison at a Glance

| Feature | .litertlm (LiteRT) | GGUF (llama.cpp) | .task (MediaPipe) | .bin / .tflite (Raw) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Target** | Mobile & Edge GenAI | Desktop & Server LLM | High-level Task API | Generic Inference |
| **Packaging** | Multi-graph Bundle | Single-file Quantized | Task-specific Bundle | Raw Graph/Weights |
| **Metadata** | Built-in (Tokenizer/KV) | Built-in | Built-in | Manual / External |
| **Multimodal** | Native Support | Complex / External | Limited to Task | Manual Wiring |

### Why .litertlm is Favorable for GenAI

If we look at the history of Google's edge AI, we might encounter **.task** files (used by MediaPipe) or generic **.bin** files for raw weights. Here is why **.litertlm** is the superior choice for models like MedGemma:

1.  **GenAI Optimization**: Unlike `.task` files, which were designed for static tasks like object detection, `.litertlm` is architected for the stateful, iterative nature of LLMs. It natively understands KV-caching and differentiates between "prefill" (processing the prompt) and "decode" (generating tokens) phases.
2.  **Unified Multimodal Logic**: For MedGemma, we aren't just running a text model; we're running a vision encoder and an LLM together. `.litertlm` bundles these separate graphs into one artifact, ensuring the vision adapter and the language head are always in sync.
3.  **Self-Describing**: Generic `.bin` or raw `.tflite` files require us to manually manage tokenizers and input/output shapes. `.litertlm` includes the tokenizer configuration and model hyperparameters, making it a "plug-and-play" artifact for the LiteRT SDK.
4.  **Hardware Acceleration**: It is optimized to leverage the **LiteRT GenAI API**, which provides specialized kernels for mobile GPUs and NPUs, significantly outperforming generic graph execution.

---

## 1. Prerequisites

Before starting, ensure we have:

*   **Model Access**: Accept the terms of use for `google/medgemma-1.5-4b-it` on Hugging Face.
*   **HF Token**: Create a Hugging Face read access token.
*   **Hardware**: A machine with sufficient RAM/VRAM (16GB+ recommended) for the export process.

---

## 2. Setting Up the Environment

We use a custom wrapper that leverages the latest `litert_torch` export pipeline.

```bash
cd medgemma_litert
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

This script sets up a virtual environment and installs the necessary dependencies, including the `litert-torch-nightly` build, which contains the essential `image_text_to_text` export task.

---

## 3. The Conversion Workflow

The conversion is handled by `convert_medgemma15_4b_to_litert.py`. This script performs a sophisticated multimodal export, bundling:

*   Text prefill and decode TFLite graphs.
*   The vision encoder TFLite graph.
*   Vision adapter layers.
*   Tokenizer and necessary metadata.

### Step 3a: Configure Authentication

```bash
export HF_TOKEN='your_huggingface_token_here'
```

### Step 3b: Run a Dry Run (Optional)

Verify our configuration before starting the compute-intensive export.

```bash
python convert_medgemma15_4b_to_litert.py --dry_run
```

### Step 3c: Start the Conversion

Execute the conversion with optimization flags for quantization and performance.

```bash
python convert_medgemma15_4b_to_litert.py \
  --output_dir out/medgemma-1.5-4b-it-litertlm \
  --prefill_lengths 128,256,512 \
  --cache_length 4096 \
  --quantization_recipe dynamic_wi8_afp32 \
  --vision_quantization_recipe weight_only_wi8_afp32
```

### Key Parameters Explained:

*   **`--prefill_lengths`**: Generates optimized graphs for these specific input sizes, improving initial latency.
*   **`--cache_length`**: Sets the maximum sequence length (context window) for the model.
*   **`--quantization_recipe`**: Uses `dynamic_wi8_afp32` (8-bit weights, 32-bit floating point activations) for the LLM core to balance size and accuracy.
*   **`--vision_quantization_recipe`**: Specifically optimizes the vision encoder weights.

---

## 4. Understanding the Output

After a successful conversion, our `out/medgemma-1.5-4b-it-litertlm/` directory will contain several key artifacts:

*   **`model.litertlm`**: The primary bundle for LiteRT GenAI. This is a single-file container for the graphs and metadata.
*   **`model.tflite`**: The main LLM (text) TFLite graph.
*   **`embedder.tflite`**: The token embedding TFLite graph.
*   **`vision_encoder.tflite`**: The specialized vision encoder for processing medical images.
*   **`tokenizer.json`**: The model's tokenizer configuration.

---

## 5. Model Deployment on Edge Devices

*To be confirmed*

---

## Conclusion

Converting MedGemma 1.5 4B to LiteRT is a powerful step toward bringing expert-level medical AI to the edge. By utilizing `litert_torch` and domain-specific models like MedGemma, developers can build responsive, private, and capable healthcare applications that work anywhere.

For more details on custom configurations and multimodal tasks, check out the [LiteRT documentation](https://ai.google.dev/edge/litert).
