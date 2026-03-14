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
        if "/content/ComfyUI" not in sys.path:
            sys.path.insert(0, "/content/ComfyUI")
        from nodes import NODE_CLASS_MAPPINGS

        # Import ComfyUI-GGUF nodes specifically
        try:
            import importlib
            gguf_module = importlib.import_module("custom_nodes.ComfyUI-GGUF.nodes")
            gguf_mappings = gguf_module.NODE_CLASS_MAPPINGS if hasattr(gguf_module, 'NODE_CLASS_MAPPINGS') else {}
        except Exception:
            try:
                from custom_nodes import ComfyUI_GGUF
                gguf_mappings = ComfyUI_GGUF.NODE_CLASS_MAPPINGS
            except Exception:
                gguf_mappings = {}

        all_nodes = {**NODE_CLASS_MAPPINGS, **gguf_mappings}

        _nodes = {
            "UNETLoader":       all_nodes["UNETLoader"](),
            "CLIPLoader":       all_nodes["CLIPLoader"](),
            "VAELoader":        all_nodes["VAELoader"](),
            "CLIPTextEncode":   all_nodes["CLIPTextEncode"](),
            "KSampler":         all_nodes["KSampler"](),
            "VAEDecode":        all_nodes["VAEDecode"](),
            "VAEEncode":        all_nodes["VAEEncode"](),
            "EmptyLatentImage": all_nodes["EmptyLatentImage"](),
            "SetLatentNoiseMask": all_nodes["SetLatentNoiseMask"](),
        }
        if "CLIPLoaderGGUF" in all_nodes:
            _nodes["CLIPLoaderGGUF"] = all_nodes["CLIPLoaderGGUF"]()
            
        try:
            import nodes
            if hasattr(nodes, "init_extra_nodes"):
                nodes.init_extra_nodes()
            from nodes import NODE_CLASS_MAPPINGS as ALL_NODES
            if "DifferentialDiffusion" in ALL_NODES:
                _nodes["DifferentialDiffusion"] = ALL_NODES["DifferentialDiffusion"]()
        except:
            pass
            
        if "DifferentialDiffusion" not in _nodes:
            try:
                from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
                _nodes["DifferentialDiffusion"] = DifferentialDiffusion()
            except ImportError:
                pass
                
    return _nodes

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _unet, _clip, _vae
    if _loaded:
        return
    n = _get_nodes()
    print("⏳ Loading FLUX.2-klein 9B (Hybrid: FP8 UNET + GGUF CLIP)...")
    with torch.inference_mode():
        # Load Safetensors UNET
        _unet = n["UNETLoader"].load_unet(
            unet_name="flux-2-klein-9b-kv-fp8.safetensors",
            weight_dtype="fp8_e4m3fn_fast"
        )[0]
        
        # Load GGUF CLIP (Single Qwen3 8B, no T5 needed for klein)
        _clip = n["CLIPLoaderGGUF"].load_clip(
            clip_name="Qwen3-8B-Q3_K_M.gguf",
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

    # Pass the full image to maintain aspect ratio and body proportions
    crop_pil = _resize_to_multiple(original, multiple=64, max_dim=1024)
    cw, ch = crop_pil.size

    mask_pil = Image.fromarray(mask_combined).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    # Encode raw unadulterated pixels so DifferentialDiffusion retains full structural context under the denoise
    crop_tensor = _pil_to_tensor(crop_pil)
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0)
    latent_base = n["VAEEncode"].encode(_vae, crop_tensor)[0]

    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]

    latent = n["SetLatentNoiseMask"].set_mask(latent_base, mask_tensor)[0]
    
    model_to_sample = _unet
    if "DifferentialDiffusion" in n:
        try:
            diff_node = n["DifferentialDiffusion"]
            if hasattr(diff_node, "apply"):
                res = diff_node.apply(model_to_sample)
            elif hasattr(diff_node.__class__, "execute"):
                res = diff_node.__class__.execute(model_to_sample)
            elif hasattr(diff_node, "execute"):
                res = diff_node.execute(model_to_sample)
            else:
                res = model_to_sample
                
            if isinstance(res, tuple):
                model_to_sample = res[0]
            elif hasattr(res, "__class__") and "NodeOutput" in res.__class__.__name__:
                model_to_sample = res.args[0] if hasattr(res, "args") else res[0]
            else:
                model_to_sample = res
        except Exception as e:
            print(f"⚠️ DifferentialDiffusion ignored due to API change: {e}")

    samples = n["KSampler"].sample(
        model_to_sample, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]

    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    result_full_np = np.array(decoded * 255, dtype=np.uint8)[0]
    result_full_pil = Image.fromarray(result_full_np).resize(original.size, Image.LANCZOS)

    # Composite the generated full image back over the original image using the mask
    result = np.array(original).copy()
    mask_composite = Image.fromarray(mask_combined).resize(original.size, Image.LANCZOS)
    mask_float = np.array(mask_composite).astype(np.float32)[:, :, None] / 255.0

    mask_blur = Image.fromarray((mask_float[:, :, 0] * 255).astype(np.uint8))
    mask_blur = mask_blur.filter(ImageFilter.GaussianBlur(32))
    mask_float = np.array(mask_blur).astype(np.float32)[:, :, None] / 255.0

    old_region = result.astype(np.float32)
    new_region = np.array(result_full_pil).astype(np.float32)
    blended = new_region * mask_float + old_region * (1 - mask_float)

    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
