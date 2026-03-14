# ============================================================
#  Engine: Qwen-Image-Edit-2511 (GGUF Q4_K_M)
#  Instruction-based image editing — img2img + inpaint
#  Uses ComfyUI-GGUF nodes to load the GGUF model
#  Model: https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF
#  File:  qwen-image-edit-2511-Q4_K_M.gguf
# ============================================================

import os, gc, torch, numpy as np
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
        if "/content/ComfyUI" not in sys.path:
            sys.path.insert(0, "/content/ComfyUI")
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
            "CLIPLoaderGGUF":   all_nodes.get("CLIPLoaderGGUF", all_nodes.get("CLIPLoader"))(),
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
            "qwen-image-edit-2511-Q4_0.gguf"
        )[0]
        # Load GGUF text encoder (Qwen2.5-VL)
        _clip = n["CLIPLoaderGGUF"].load_clip(
            "Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
            type="qwen2vl"
        )[0]
        # Load VAE — use diffusers directly since ComfyUI has no native
        # support for the Qwen 96-ch 3D AutoencoderKLQwenImage architecture
        from diffusers import AutoencoderKLQwenImage
        import comfy.model_management as mm
        
        print("  ⏳ Loading Qwen VAE via diffusers...")
        vae_local = "/content/ComfyUI/models/vae/qwen_image_vae.safetensors"
        
        # Setup local directory with config so from_pretrained works offline
        import json, shutil
        vae_dir = "/content/qwen_vae_local"
        os.makedirs(vae_dir, exist_ok=True)
        
        # Write the config.json that diffusers needs
        vae_config = {
            "_class_name": "AutoencoderKLQwenImage",
            "_diffusers_version": "0.36.0.dev0",
            "attn_scales": [],
            "base_dim": 96,
            "dim_mult": [1, 2, 4, 4],
            "dropout": 0.0,
            "num_res_blocks": 2,
            "temperal_downsample": [False, True, True],
            "z_dim": 16
        }
        with open(os.path.join(vae_dir, "config.json"), "w") as f:
            json.dump(vae_config, f)
        
        # Symlink the safetensors file into the local dir
        link_path = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(link_path):
            os.symlink(vae_local, link_path)
        
        _vae = AutoencoderKLQwenImage.from_pretrained(
            vae_dir, torch_dtype=torch.bfloat16
        )
        _vae = _vae.to(mm.vae_device())
        _vae.eval()
        
        # ── Optimize VRAM for 15GB Colab GPUs ──
        # The Qwen 3D VAE is extremely memory heavy. Tiling is required
        # to prevent CUDA OOM when encoding/decoding high-res images.
        try:
            _vae.enable_slicing()
            _vae.enable_tiling()
            print("  ✅ VAE slicing & tiling enabled")
        except Exception as e:
            print(f"  ⚠️ Could not enable VAE tiling: {e}")
            
        print("  ✅ Qwen VAE loaded via diffusers")
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

def _vae_encode(image_tensor):
    """Encode image tensor [B,H,W,C] 0-1 → latent using diffusers VAE"""
    # ComfyUI tensor is [B, H, W, C] but diffusers needs [B, C, T, H, W] (3D VAE)
    x = image_tensor.permute(0, 3, 1, 2).to(_vae.device, dtype=_vae.dtype)
    x = x * 2.0 - 1.0  # normalize to [-1, 1]
    x = x.unsqueeze(2)  # add temporal dim: [B, C, H, W] → [B, C, 1, H, W]
    with torch.no_grad():
        latent = _vae.encode(x).latent_dist.sample()
    # Squeeze temporal dim and return as ComfyUI-compatible latent dict
    latent = latent.squeeze(2)  # [B, C, 1, H, W] → [B, C, H, W]
    return {"samples": latent.float().cpu()}

def _vae_decode(latent_dict):
    """Decode latent dict → image tensor [B,H,W,C] 0-1"""
    # Latent from sampler is [B, C, H, W]. Diffusers decode takes 4D latents directly.
    latent = latent_dict["samples"].to(_vae.device, dtype=_vae.dtype)
    with torch.no_grad():
        decoded = _vae.decode(latent).sample
    # The output image from decode is 5D [B, C, T, H, W]. We squeeze T=1.
    if len(decoded.shape) == 5:
        decoded = decoded.squeeze(2)
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded.clamp(0, 1)
    # Convert back to ComfyUI format [B, H, W, C]
    return decoded.permute(0, 2, 3, 1).float().cpu()

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
    latent = _vae_encode(img_tensor)

    # Qwen-Image-Edit UNET requires 64 channels (4x 16-channel latents).
    # We must provide the source image latent as conditioning.
    cond_latent = latent["samples"].clone()
    
    # ComfyUI uses "noise_mask" and additional unet conditioning to pass these
    # through to models that need extra 'in_channels' (like 64).
    # Since we are using standard KSampler, we wrap the conditioning payload
    # inside the pos/neg conditioning blocks just like unCLIP does.
    
    # Actually, ComfyUI handles InstructPix2Pix natively inside Sampler if we flag it.
    # The correct way in Comfy is to attach `concat_latent_image` to the conditionings.
    cond_dict = {"concat_latent_image": cond_latent, "concat_mask": torch.ones((1, 1, cond_latent.shape[2], cond_latent.shape[3]))}
    
    # Append to prompt conditionings
    pos[0][1].update(cond_dict)
    neg[0][1].update(cond_dict)

    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = _vae_decode(samples).detach()
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
    latent_base = _vae_encode(filled_tensor)
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0)

    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]

    latent = n["SetLatentNoiseMask"].set_mask(latent_base, mask_tensor)[0]
    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]

    decoded = _vae_decode(samples).detach()
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
