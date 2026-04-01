# Edge AI Medical Intelligence (Edge-AI-MS)

This repository contains a full-stack implementation for deploying specialized medical AI models directly to edge devices (web browsers). By leveraging **MedGemma 1.5 4B** and **LiteRT** (formerly TensorFlow Lite), we enable high-performance, 100% private, and offline-capable clinical reasoning using WebGPU.

## 🌟 Key Features

*   **100% Private & Offline**: Medical data never leaves the user's device. All inference is performed locally in the browser.
*   **Multimodal Intelligence**: Supports both text and image inputs (e.g., MRI/CT scans) for comprehensive medical analysis.
*   **Next-Gen Architecture**: Powered by **Google LiteRT** and **WebGPU** for hardware-accelerated performance on modern workstations.
*   **ChatGPT-like Experience**: Implements fluid response streaming with a smart accumulator to handle varied model outputs.
*   **Robust Fallback System**: Automatically scales down to **Gemma 3 2B** if local hardware (e.g., VRAM) is insufficient for the primary 4B model.

## 📁 Repository Structure

*   **`chat-ui/`**: A modern React + TypeScript + Vite web application that serves as the clinician's interface.
*   **`medgemma_litert/`**: Python conversion scripts and environment setups to transform raw Hugging Face weights into optimized `.litertlm` bundles.
*   **`BLOG.md`**: A detailed technical deep-dive into the architecture, conversion logic, and deployment principles.

## 🚀 Getting Started

### 1. Web UI Deployment (`chat-ui`)

To run the local medical chat interface:

```bash
cd chat-ui
npm install
npm run dev
```

**Note**: You will need to place your converted `model.litertlm` file in `chat-ui/public/models/model.litertlm` (or use the provided symlink setup). This repository should use the litertlm model from [https://huggingface.co/ai4med-id/medgemma-1.5-4b-it-litertlm](https://huggingface.co/ai4med-id/medgemma-1.5-4b-it-litertlm).

### 2. Model Conversion (`medgemma_litert`)

If you want to convert the model yourself from the original Hugging Face weights:

```bash
cd medgemma_litert
./setup_env.sh
source .venv/bin/activate
export HF_TOKEN='your_token'
python convert_medgemma15_4b_to_litert.py --quantization_recipe dynamic_wi8_afp32
```

## 💻 Technical Requirements

*   **Browser**: Modern Chrome or Edge with **WebGPU** support enabled.
*   **Hardware**: 
    *   **RAM**: 16GB+ recommended for MedGemma 1.5 4B.
    *   **GPU**: Dedicated GPU recommended for optimal inference speeds.
*   **Security**: The web server must serve headers for Cross-Origin Opener Policy (**COOP: same-origin**) and Cross-Origin Embedder Policy (**COEP: require-corp**) to enable advanced browser features.

---
*Disclaimer: This is a technical demonstration. Always consult with a qualified medical professional for clinical diagnosis.*
