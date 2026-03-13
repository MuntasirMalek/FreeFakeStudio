# ⚡ Z-Fooocus

**Generate · Img2Img · Inpaint** — A [Fooocus](https://github.com/lllyasviel/Fooocus)-style app for [Z-Image Turbo](https://github.com/Tongyi-MAI/Z-Image) on Free Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuntasirMalek/Z-Fooocus/blob/main/Z_Fooocus.ipynb)

## Features

One model. Three features. Zero quality loss. Fits in 15GB VRAM.

| Tab | What it does |
|-----|--------------|
| 🖼️ **Generate** | Text → high-quality image (Z-Image Turbo FP8) |
| ✏️ **Img2Img** | Upload photo + prompt → restyled image |
| 🎨 **Inpaint** | Paint mask on photo → only painted area gets regenerated |

### Inpaint Algorithm (Fooocus-inspired)
1. Crop to region around your painted mask
2. Fill masked area with blurred surroundings (`fooocus_fill`)
3. VAE encode → add noise only to masked region
4. Z-Image Turbo regenerates the masked area
5. Composite result back with feathered edges

## Quick Start

1. Open the Colab notebook (badge above)
2. Connect to a **T4 GPU** runtime
3. Run **Step 1** (~5 min — installs + downloads 3 model files)
4. Run **Step 2** (launches Gradio UI)
5. Click the `share` link → use the app!

## Model Files (~13 GB total)

| File | Size | What |
|------|------|------|
| `z-image-turbo-fp8-e4m3fn.safetensors` | ~6 GB | Diffusion transformer (FP8) |
| `qwen_3_4b.safetensors` | ~6.5 GB | Text encoder (3.4B) |
| `ae.safetensors` | ~0.3 GB | VAE |

## Credits

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Tongyi-MAI (Alibaba)
- [Fooocus](https://github.com/lllyasviel/Fooocus) by lllyasviel — inpaint algorithm
- [NeuralFalconYT/Z-Image-Colab](https://github.com/NeuralFalconYT/Z-Image-Colab) — original Colab
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — backend engine
- [T5B/Z-Image-Turbo-FP8](https://huggingface.co/T5B/Z-Image-Turbo-FP8) — FP8 quantized model

## License

Apache-2.0 (same as Z-Image)
