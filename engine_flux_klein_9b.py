# ============================================================
#  Engine: FLUX.2-klein 9B KV FP8
#  Generation (4 steps distilled), img2img, mask-based inpaint
#  Uses ComfyUI nodes — EXPERIMENTAL (brand new model)
# ============================================================

import gc, torch, numpy as np
from PIL import Image, ImageFilter

_loaded = False
_unet = None
_clip = None
_vae = None
_nodes = {}

# ── Node references ────────────────────────────────────────
def _get_nodes():
    global _nodes
    if not _nodes:
        import sys
        sys.path.insert(0, "/content/ComfyUI")
        from nodes import NODE_CLASS_MAPPINGS
        _nodes = {
            "UNETLoader":       NODE_CLASS_MAPPINGS["UNETLoader"](),
            "CLIPLoader":       NODE_CLASS_MAPPINGS["CLIPLoader"](),
            "VAELoader":        NODE_CLASS_MAPPINGS["VAELoader"](),
            "CLIPTextEncode":   NODE_CLASS_MAPPINGS["CLIPTextEncode"](),
            "KSampler":         NODE_CLASS_MAPPINGS["KSampler"](),
            "VAEDecode":        NODE_CLASS_MAPPINGS["VAEDecode"](),
            "VAEEncode":        NODE_CLASS_MAPPINGS["VAEEncode"](),
            "EmptyLatentImage": NODE_CLASS_MAPPINGS["EmptyLatentImage"](),
            "SetLatentNoiseMask": NODE_CLASS_MAPPINGS["SetLatentNoiseMask"](),
        }
    return _nodes

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _unet, _clip, _vae
    if _loaded:
        return
    # Linux Swap workaround for free Colab instances
    import psutil
    sys_ram_gb = psutil.virtual_memory().total / (1024**3)
    if sys_ram_gb < 16:
        import os
        if not os.path.exists("/swapfile"):
            print(f"⚠️ Low System RAM detected ({sys_ram_gb:.1f}GB).")
            print("⚠️ Creating 8GB Swap File to prevent OS crash (May be slow due to disk swapping)...")
            os.system("dd if=/dev/zero of=/swapfile bs=1M count=8192")
            os.system("chmod 600 /swapfile")
            os.system("mkswap /swapfile")
            os.system("swapon /swapfile")
            print("✅ Swap file active.")
    n = _get_nodes()
    print("⏳ Loading FLUX.2-klein 9B GGUF...")
    with torch.inference_mode():
        # Load GGUF UNET
        _unet = n["UnetLoaderGGUF"].load_unet_gguf(
            unet_name="flux-2-klein-9b-q4_k_m.gguf"
        )[0]
        
        # Load GGUF CLIP (Single Qwen3 8B, no T5 needed for klein)
        _clip = n["CLIPLoaderGGUF"].load_clip(
            clip_name="Qwen3-8B-Q4_K_M.gguf",
            type="flux2"
        )[0]
        
        # Load VAE
        _vae  = n["VAELoader"].load_vae("flux2-vae.safetensors")[0]
    _loaded = True
    print("✅ FLUX.2-klein loaded!")

def unload():
    global _loaded, _unet, _clip, _vae
    _unet = None
    _clip = None
    _vae = None
    _loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🗑️ FLUX.2-klein unloaded")

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

# ── Generate ───────────────────────────────────────────────
@torch.inference_mode()
def generate(prompt, negative, width, height, seed, cfg, denoise, steps=4):
    n = _get_nodes()
    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    latent = n["EmptyLatentImage"].generate(width, height, batch_size=1)[0]
    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Img2Img ────────────────────────────────────────────────
@torch.inference_mode()
def img2img(input_image, prompt, negative, seed, cfg, denoise, steps=4):
    n = _get_nodes()
    input_image = _resize_to_multiple(input_image)
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
def inpaint(original, mask_combined, prompt, negative, seed, cfg, denoise, steps=4):
    """original: PIL Image, mask_combined: numpy uint8 array (255=masked)"""
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
