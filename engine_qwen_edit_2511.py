# ============================================================
#  Engine: Qwen-Image-Edit-2511 (GGUF Q4_K_M)
#  Instruction-based image editing — img2img + inpaint
#  Uses ComfyUI-GGUF nodes to load the GGUF model
#  Model: https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF
#  File:  qwen-image-edit-2511-Q4_K_M.gguf
# ============================================================

import gc, torch, numpy as np
from PIL import Image, ImageFilter

_loaded = False
_unet = None
_clip = None
_vae = None
_nodes = {}

# ── Node references (ComfyUI + ComfyUI-GGUF) ──────────────
def _get_nodes():
    global _nodes
    if not _nodes:
        import sys
        sys.path.insert(0, "/content/ComfyUI")
        sys.path.insert(0, "/content/ComfyUI/custom_nodes/ComfyUI-GGUF")
        from nodes import NODE_CLASS_MAPPINGS

        # Import ComfyUI-GGUF nodes
        try:
            import importlib
            gguf_module = importlib.import_module("custom_nodes.ComfyUI-GGUF.nodes")
            gguf_mappings = gguf_module.NODE_CLASS_MAPPINGS if hasattr(gguf_module, 'NODE_CLASS_MAPPINGS') else {}
        except Exception:
            # Try alternate import path
            try:
                from custom_nodes import ComfyUI_GGUF
                gguf_mappings = ComfyUI_GGUF.NODE_CLASS_MAPPINGS
            except Exception:
                gguf_mappings = {}

        # Merge GGUF nodes into standard nodes
        all_nodes = {**NODE_CLASS_MAPPINGS, **gguf_mappings}

        _nodes = {
            "UnetLoaderGGUF":   all_nodes.get("UnetLoaderGGUF", all_nodes.get("UNETLoader"))(),
            "CLIPLoaderGGUF":   all_nodes.get("DualCLIPLoaderGGUF", all_nodes.get("CLIPLoader"))(),
            "VAELoader":        all_nodes["VAELoader"](),
            "CLIPTextEncode":   all_nodes["CLIPTextEncode"](),
            "KSampler":         all_nodes["KSampler"](),
            "VAEDecode":        all_nodes["VAEDecode"](),
            "VAEEncode":        all_nodes["VAEEncode"](),
            "EmptyLatentImage": all_nodes["EmptyLatentImage"](),
            "SetLatentNoiseMask": all_nodes["SetLatentNoiseMask"](),
        }
    return _nodes

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _unet, _clip, _vae
    if _loaded:
        return
    n = _get_nodes()
    print("⏳ Loading Qwen-Image-Edit-2511 Q4_K_M GGUF...")
    with torch.inference_mode():
        # Load GGUF diffusion model
        _unet = n["UnetLoaderGGUF"].load_unet(
            "qwen-image-edit-2511-Q4_K_M.gguf"
        )[0]
        # Load GGUF text encoder (Qwen2.5-VL)
        _clip = n["CLIPLoaderGGUF"].load_clip(
            "Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
            type="qwen2vl"
        )[0]
        # Load VAE
        _vae = n["VAELoader"].load_vae("qwen_image_vae.safetensors")[0]
    _loaded = True
    print("✅ Qwen-Image-Edit loaded!")

def unload():
    global _loaded, _unet, _clip, _vae
    _unet = None
    _clip = None
    _vae = None
    _loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🗑️ Qwen-Image-Edit unloaded")

def is_loaded():
    return _loaded

# ── Helpers ────────────────────────────────────────────────
def _pil_to_tensor(img):
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

def _resize_to_multiple(img, multiple=64, max_dim=1024):
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = max(multiple, int(w * scale) // multiple * multiple)
    new_h = max(multiple, int(h * scale) // multiple * multiple)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ── Img2Img (instruction-based editing) ───────────────────
@torch.inference_mode()
def img2img(input_image, prompt, negative, seed, cfg, denoise, steps=40):
    """
    Instruction-based image editing via Qwen-Image-Edit.
    The model takes an input image + prompt instruction and edits accordingly.
    """
    n = _get_nodes()
    input_image = _resize_to_multiple(input_image.convert("RGB"))
    img_tensor = _pil_to_tensor(input_image)

    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    latent = n["VAEEncode"].encode(_vae, img_tensor)[0]

    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Inpaint (Fooocus-style mask-based) ────────────────────
def _compute_crop_region(mask_np, padding=0.30):
    indices = np.where(mask_np > 0)
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        return None
    a, b = np.min(indices[0]), np.max(indices[0])
    c, d = np.min(indices[1]), np.max(indices[1])
    h_center, h_half = (b + a) // 2, (b - a) // 2
    w_center, w_half = (d + c) // 2, (d - c) // 2
    size = int(max(h_half, w_half) * (1.0 + padding))
    a = max(0, h_center - size)
    b = min(mask_np.shape[0], h_center + size + 1)
    c = max(0, w_center - size)
    d = min(mask_np.shape[1], w_center + size + 1)
    return (a, b, c, d)

def _fooocus_fill(image_np, mask_np):
    current = image_np.copy()
    raw = image_np.copy()
    area = np.where(mask_np < 127)
    store = raw[area]
    for k, repeats in [(512,2),(256,2),(128,4),(64,4),(33,8),(15,8),(5,16),(3,16)]:
        for _ in range(repeats):
            pil_img = Image.fromarray(current)
            pil_img = pil_img.filter(ImageFilter.BoxBlur(k))
            current = np.array(pil_img)
            current[area] = store
    return current

@torch.inference_mode()
def inpaint(original, mask_combined, prompt, negative, seed, cfg, denoise, steps=40):
    """Mask-based inpaint using Qwen-Image-Edit GGUF."""
    n = _get_nodes()

    crop = _compute_crop_region(mask_combined)
    if crop is None:
        raise ValueError("No mask detected.")
    a, b, c, d = crop

    cropped_img = np.array(original)[a:b, c:d]
    cropped_mask = mask_combined[a:b, c:d]

    crop_pil = Image.fromarray(cropped_img)
    crop_pil = _resize_to_multiple(crop_pil, multiple=64, max_dim=1024)
    cw, ch = crop_pil.size

    mask_pil = Image.fromarray(cropped_mask).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    filled = _fooocus_fill(np.array(crop_pil), mask_resized)
    filled_pil = Image.fromarray(filled)

    filled_tensor = _pil_to_tensor(filled_pil)
    latent_base = n["VAEEncode"].encode(_vae, filled_tensor)[0]
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0)

    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]

    latent = n["SetLatentNoiseMask"].set_mask(latent_base, mask_tensor)[0]
    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]

    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    result_crop = np.array(decoded * 255, dtype=np.uint8)[0]
    result_crop_pil = Image.fromarray(result_crop).resize((d - c, b - a), Image.LANCZOS)

    # Composite back
    result = np.array(original).copy()
    result_crop_np = np.array(result_crop_pil)
    mask_composite = Image.fromarray(cropped_mask).resize((d - c, b - a), Image.LANCZOS)
    mask_float = np.array(mask_composite).astype(np.float32)[:, :, None] / 255.0

    mask_blur = Image.fromarray((mask_float[:, :, 0] * 255).astype(np.uint8))
    mask_blur = mask_blur.filter(ImageFilter.GaussianBlur(3))
    mask_float = np.array(mask_blur).astype(np.float32)[:, :, None] / 255.0

    old_region = result[a:b, c:d].astype(np.float32)
    new_region = result_crop_np.astype(np.float32)
    blended = new_region * mask_float + old_region * (1 - mask_float)
    result[a:b, c:d] = blended.clip(0, 255).astype(np.uint8)

    return Image.fromarray(result)
