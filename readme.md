# 🎭 CheapFakeStudio

**Open-source Multi-Model AI Image Studio** — Generate, Img2Img & Inpaint with FLUX, Qwen, Z-Image and many more. Zero-setup for Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuntasirMalek/CheapFakeStudio/blob/main/CheapFakeStudio.ipynb)

## Features

Multiple state-of-the-art models in one unified studio. No local installation required. Fits perfectly in the free 15GB VRAM Colab T4 tier.

| Tab | What it does |
|-----|--------------|
| 🖼️ **Generate** | Text → high-quality image (Supports Z-Image Turbo, FLUX.2) |
| ✏️ **Img2Img** | Upload photo + prompt → restyled image |
| 🎨 **Inpaint** | Paint mask on photo → only painted area gets regenerated |

### Inpaint Superpowers
Powered by structural-locking constraints and native model APIs, CheapFakeStudio allows for incredibly precise inpainting:
- Drop your Denoise to `0.55` to mathematically lock your original clothing silhouette even under massive, sloppy paint masks.
- Supports native Qwen-Image-Edit conversational instructions (e.g., *"change her saree to green"*)
- Supports traditional mask inpainting via FLUX.2 and Z-Image.

## Quick Start

1. Open the Colab notebook (badge above)
2. Connect to a free **T4 GPU** runtime
3. Select which models you want to download
4. Run **Step 1** (~5 min — installs ComfyUI backend + downloads models)
5. Run **Step 2** (launches Gradio UI)
6. Click the `share` link → use the app!

## Available Models

Choose to download any or all of the following models on startup:

| Model | Size | Strength |
|------|------|------|
| **Z-Image Turbo** | ~6 GB | Blistering fast text-to-image (8 steps) |
| **FLUX.2-klein 4B** | ~11 GB | Excellent photorealism and general purpose Inpaint/Img2Img |
| **FLUX.2-klein 9B Hybrid** | ~13 GB | Original 10GB FP8 UNET + GGUF text encoder for maximum detail |
| **Qwen-Image-Edit-2511** | ~15 GB | Instruction-based conversational Img2Img & Inpaint |

## Credits

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Tongyi-MAI
- [Fooocus](https://github.com/lllyasviel/Fooocus) by lllyasviel — UI layout inspiration
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — backend engine powering all workflows

## License

Apache-2.0
