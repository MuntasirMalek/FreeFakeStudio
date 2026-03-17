# ============================================================
#  Engine: Qwen-Image-Edit-2511 (GGUF Q4_K_M)
#  Instruction-based image editing — img2img + inpaint
#  Uses ComfyUI-GGUF nodes to load the GGUF model
#  Model: https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF
#  File:  qwen-image-edit-2511-Q4_K_M.gguf
# ============================================================

import os, gc, torch, math, numpy as np
from PIL import Image, ImageFilter

_loaded = False
_unet = None
_clip = None
_vae_diffusers = None
_nodes = {}

# ── Node references (ComfyUI + ComfyUI-GGUF) ──────────────
def _get_nodes():
    global _nodes
    if not _nodes:
        import sys
        if "/content/ComfyUI" not in sys.path:
            sys.path.insert(0, "/content/ComfyUI")
        from nodes import NODE_CLASS_MAPPINGS

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
            "UnetLoaderGGUF":   all_nodes.get("UnetLoaderGGUF", all_nodes.get("UNETLoader"))(),
            "CLIPLoaderGGUF":   all_nodes.get("CLIPLoaderGGUF", all_nodes.get("CLIPLoader"))(),
            "CLIPTextEncode":   all_nodes["CLIPTextEncode"](),
            "KSampler":         all_nodes["KSampler"](),
        }
    return _nodes

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _unet, _clip, _vae_diffusers
    if _loaded:
        return
    n = _get_nodes()
    print("⏳ Loading Qwen-Image-Edit-2511 Q4_K_M GGUF...")
    with torch.inference_mode():
        _unet = n["UnetLoaderGGUF"].load_unet(
            "qwen-image-edit-2511-Q4_0.gguf"
        )[0]
        # Try best available CLIP quantization (Q4_K_S > Q3_K_M > Q2_K)
        _clip_candidates = [
            "Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf",
            "Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf",
            "Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
        ]
        _clip_file = None
        for c in _clip_candidates:
            _clip_path = os.path.join("/content/ComfyUI/models/clip", c)
            if os.path.exists(_clip_path):
                _clip_file = c
                break
        if _clip_file is None:
            _clip_file = "Qwen2.5-VL-7B-Instruct-Q2_K.gguf"

        print(f"  📊 Using CLIP: {_clip_file}")
        _clip = n["CLIPLoaderGGUF"].load_clip(
            _clip_file,
            type="qwen2vl"
        )[0]

        # Diagnostic: check what latent format the model uses
        try:
            lf = _unet.model.latent_format
            print(f"  📊 Model latent_format: {type(lf).__name__}")
            if hasattr(lf, 'latents_mean'):
                print(f"  📊 Has latents_mean/std normalization: YES")
            else:
                print(f"  📊 Has latents_mean/std normalization: NO (scale_factor={lf.scale_factor})")
        except Exception as e:
            print(f"  📊 Could not inspect latent format: {e}")

        # Apply shift override — critical for quality!
        # Default shift is 1.15 which produces blurry/cartoon results.
        # Unsloth recommends shift=12-13 for good outputs.
        try:
            from nodes import NODE_CLASS_MAPPINGS as _ncm
            msf_cls = _ncm.get("ModelSamplingFlux")
            if msf_cls:
                _unet = msf_cls().patch(
                    _unet, max_shift=13.0, base_shift=0.5,
                    width=1024, height=1024
                )[0]
                print("  ✅ Applied shift=13.0 (quality fix)")
            else:
                print("  ⚠️ ModelSamplingFlux node not found, using default shift")
        except Exception as e:
            print(f"  ⚠️ Could not apply shift override: {e}")

        # Load VAE via diffusers
        from diffusers import AutoencoderKLQwenImage
        import comfy.model_management as mm

        print("  ⏳ Loading Qwen VAE via diffusers...")
        vae_local = "/content/ComfyUI/models/vae/qwen_image_vae.safetensors"

        import json
        vae_dir = "/content/qwen_vae_local"
        os.makedirs(vae_dir, exist_ok=True)

        vae_config = {
            "_class_name": "AutoencoderKLQwenImage",
            "_diffusers_version": "0.36.0.dev0",
            "attn_scales": [],
            "base_dim": 96,
            "dim_mult": [1, 2, 4, 4],
            "dropout": 0.0,
            "num_res_blocks": 2,
            "temperal_downsample": [False, True, True],
            "z_dim": 16,
            "latents_mean": [
                -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653,
                -0.1517,  1.5508,  0.4134, -0.0715,  0.5517, -0.3632,
                -0.1922, -0.9497,  0.2503, -0.2921
            ],
            "latents_std": [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708,
                2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
                1.6382, 1.1253, 2.8251, 1.9160
            ]
        }
        with open(os.path.join(vae_dir, "config.json"), "w") as f:
            json.dump(vae_config, f)

        link_path = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(link_path):
            os.symlink(vae_local, link_path)

        _vae_diffusers = AutoencoderKLQwenImage.from_pretrained(
            vae_dir, torch_dtype=torch.bfloat16
        )
        _vae_diffusers = _vae_diffusers.to(mm.vae_device())
        _vae_diffusers.eval()

        try:
            _vae_diffusers.enable_slicing()
            _vae_diffusers.enable_tiling()
            print("  ✅ VAE slicing & tiling enabled")
        except Exception as e:
            print(f"  ⚠️ Could not enable VAE tiling: {e}")

        print("  ✅ Qwen VAE loaded via diffusers")
    _loaded = True
    print("✅ Qwen-Image-Edit loaded!")

def unload():
    global _loaded, _unet, _clip, _vae_diffusers
    _unet = None
    _clip = None
    _vae_diffusers = None
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

def _resize_to_multiple(img, multiple=16, max_dim=1024):
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = max(multiple, round(w * scale / multiple) * multiple)
    new_h = max(multiple, round(h * scale / multiple) * multiple)
    return img.resize((new_w, new_h), Image.LANCZOS)

def _vae_encode(image_tensor):
    """Encode image [B,H,W,C] 0-1 → 5D latent [B,C,1,H/8,W/8]."""
    x = image_tensor.permute(0, 3, 1, 2).to(_vae_diffusers.device, dtype=_vae_diffusers.dtype)
    x = x * 2.0 - 1.0
    x = x.unsqueeze(2)  # [B,C,H,W] → [B,C,1,H,W]
    with torch.no_grad():
        latent = _vae_diffusers.encode(x).latent_dist.mode()
    return latent.float().cpu()

def _vae_decode(latent_dict):
    """Decode latent dict → image tensor [B,H,W,C] 0-1."""
    latent = latent_dict["samples"].to(_vae_diffusers.device, dtype=_vae_diffusers.dtype)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    with torch.no_grad():
        decoded = _vae_diffusers.decode(latent).sample
    if decoded.ndim == 5:
        decoded = decoded.squeeze(2)
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded.clamp(0, 1)
    return decoded.permute(0, 2, 3, 1).float().cpu()

def _prepare_clip_image(img_pil, target_area=384*384):
    """Resize image for CLIP VL conditioning (~384×384 as per official pipeline)."""
    img = img_pil.convert("RGB")
    w, h = img.size
    scale = math.sqrt(target_area / (w * h))
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    return _pil_to_tensor(img)

def _encode_prompt(prompt, source_image_pil=None, is_negative=False):
    """Encode prompt, optionally with VL image conditioning. Falls back to text-only."""
    n = _get_nodes()

    if source_image_pil is not None and not is_negative:
        try:
            clip_img = _prepare_clip_image(source_image_pil)
            tokens = _clip.tokenize(prompt, images=[clip_img[:, :, :, :3]])
            cond = _clip.encode_from_tokens_scheduled(tokens)
            print("  📊 CLIP: image+text conditioning ✓")
            return cond
        except Exception as e:
            print(f"  ⚠️ CLIP image conditioning failed ({e}), using text-only")

    # Text-only fallback
    return n["CLIPTextEncode"].encode(_clip, prompt)[0]

# ── Img2Img (instruction-based editing) ───────────────────
@torch.inference_mode()
def img2img(input_image, prompt, negative, seed, cfg, denoise, steps=40):
    """Instruction-based image editing via Qwen-Image-Edit."""
    import node_helpers

    input_image = _resize_to_multiple(input_image.convert("RGB"))
    img_tensor = _pil_to_tensor(input_image)

    # VAE encode source image → 5D latent
    ref_latent = _vae_encode(img_tensor)
    latent = {"samples": ref_latent.clone()}

    # CLIP encoding with VL image conditioning + text
    pos = _encode_prompt(prompt, source_image_pil=input_image)
    neg = _encode_prompt(negative, is_negative=True)

    # Attach source image as reference_latents (proper QwenImage conditioning)
    pos = node_helpers.conditioning_set_values(
        pos, {"reference_latents": [ref_latent]}, append=True
    )

    print(f"  📊 img2img latent shape: {ref_latent.shape}, denoise: {denoise}")

    samples = _get_nodes()["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = _vae_decode(samples).detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])

# ── Inpaint ───────────────────────────────────────────────
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
    import node_helpers, comfy.model_management as mm

    crop_pil = _resize_to_multiple(original, multiple=16, max_dim=1024)
    cw, ch = crop_pil.size

    mask_pil = Image.fromarray(mask_combined).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    img_tensor = _pil_to_tensor(crop_pil)

    # VAE encode source image → 5D reference latent
    ref_latent = _vae_encode(img_tensor)

    # Start from empty 5D latent (denoise=1.0 means full generation)
    h_lat, w_lat = ref_latent.shape[3], ref_latent.shape[4]
    empty_latent = torch.zeros(
        [1, 16, 1, h_lat, w_lat],
        device=mm.intermediate_device()
    )
    latent = {"samples": empty_latent}

    # CLIP encoding with VL image conditioning + reference_latents
    pos = _encode_prompt(prompt, source_image_pil=crop_pil)
    neg = _encode_prompt(negative, is_negative=True)

    pos = node_helpers.conditioning_set_values(
        pos, {"reference_latents": [ref_latent]}, append=True
    )

    print(f"  📊 inpaint ref_latent: {ref_latent.shape}, start: {empty_latent.shape}")

    samples = _get_nodes()["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=1.0
    )[0]

    decoded = _vae_decode(samples).detach()
    result_np = np.array(decoded * 255, dtype=np.uint8)[0]
    result_pil = Image.fromarray(result_np).resize(original.size, Image.LANCZOS)

    # Composite back using mask
    result = np.array(original).copy()
    mask_float = np.array(Image.fromarray(mask_combined)).astype(np.float32)[:, :, None] / 255.0

    mask_blur = Image.fromarray((mask_float[:, :, 0] * 255).astype(np.uint8))
    mask_blur = mask_blur.filter(ImageFilter.GaussianBlur(3))
    mask_float = np.array(mask_blur).astype(np.float32)[:, :, None] / 255.0

    old_region = result.astype(np.float32)
    new_region = np.array(result_pil).astype(np.float32)
    blended = new_region * mask_float + old_region * (1 - mask_float)
    result = blended.clip(0, 255).astype(np.uint8)

    return Image.fromarray(result)
