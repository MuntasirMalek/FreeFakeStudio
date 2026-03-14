# ============================================================
#  Engine: Qwen-Image-Edit-2511 (GGUF Q4_K_M)
#  Instruction-based image editing — img2img + inpaint
#  Uses diffusers QwenImageEditPlusPipeline
# ============================================================

import gc, os, torch, numpy as np
from PIL import Image

_loaded = False
_pipeline = None

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _pipeline
    if _loaded:
        return
    print("⏳ Loading Qwen-Image-Edit-2511...")
    from diffusers import QwenImageEditPlusPipeline

    # Load from the downloaded GGUF or from HF repo with quantization
    # Try loading with 4-bit quantization via bitsandbytes
    try:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        _pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
        )
    except Exception as e:
        print(f"⚠️ 4-bit loading failed ({e}), trying FP16...")
        _pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.float16,
        )

    _pipeline.to("cuda")
    _pipeline.set_progress_bar_config(disable=None)

    # Enable memory optimizations
    try:
        _pipeline.enable_model_cpu_offload()
    except Exception:
        pass  # Some versions don't support this

    _loaded = True
    print("✅ Qwen-Image-Edit loaded!")

def unload():
    global _loaded, _pipeline
    if _pipeline is not None:
        del _pipeline
    _pipeline = None
    _loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🗑️ Qwen-Image-Edit unloaded")

def is_loaded():
    return _loaded

# ── Helpers ────────────────────────────────────────────────
def _resize_for_edit(img, max_dim=1024):
    """Resize image to fit within max_dim while maintaining aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ── Img2Img (instruction-based editing) ───────────────────
@torch.inference_mode()
def img2img(input_image, prompt, negative="", seed=0, cfg=1.0,
            denoise=0.8, steps=40, true_cfg=4.0):
    """
    Instruction-based image editing.
    prompt: editing instruction, e.g. "change the dress to a red saree"
    """
    input_image = _resize_for_edit(input_image.convert("RGB"))

    generator = torch.manual_seed(seed) if seed > 0 else None

    inputs = {
        "image": [input_image],
        "prompt": prompt,
        "negative_prompt": negative if negative else " ",
        "num_inference_steps": int(steps),
        "guidance_scale": float(cfg),
        "true_cfg_scale": float(true_cfg),
        "num_images_per_prompt": 1,
    }
    if generator:
        inputs["generator"] = generator

    output = _pipeline(**inputs)
    return output.images[0]

# ── Inpaint (instruction-based, no mask needed) ───────────
@torch.inference_mode()
def inpaint(original, mask_combined, prompt, negative="", seed=0,
            cfg=1.0, denoise=0.8, steps=40, true_cfg=4.0):
    """
    Qwen-Image-Edit does instruction-based editing.
    The mask is IGNORED — the model figures out what to change from the prompt.
    We still accept mask_combined for API compatibility but don't use it.
    """
    return img2img(original, prompt, negative, seed, cfg, denoise, steps, true_cfg)
