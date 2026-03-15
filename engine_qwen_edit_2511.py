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
        "euler", "simple", pos, neg, latent, denoise=1.0
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

    # Pass the full image to maintain aspect ratio and body context
    crop_pil = _resize_to_multiple(original, multiple=64, max_dim=1024)
    cw, ch = crop_pil.size

    mask_pil = Image.fromarray(mask_combined).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    pixels = _pil_to_tensor(crop_pil)
    # [H, W] mask -> [1, 1, H, W]
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    # 1) Modulo 8 precision cropping (ComfyUI InpaintModelConditioning)
    # The VAE compresses exactly 8x8 pixels into 1 latent unit. 
    # If the image dimensions aren't perfectly divisible by 8, the latent padding 
    # will permanently desynchronize the spatial alignment of the mask!
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        mask_tensor = mask_tensor[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

    # 2) Qwen-Image-Edit needs the raw pristine image for conditioning.
    # It does NOT use an erased hole-punch like traditional SDXL Inpaint models!
    latent_raw = _vae_encode(pixels)
    
    cond_latent = latent_raw["samples"].clone()

    # 2) We pass the precise spatial mask to `concat_mask` so the Instruct UNet 
    # knows exactly which object to modify visually.
    import torch.nn.functional as F
    latent_mask = F.interpolate(mask_tensor, size=(cond_latent.shape[2], cond_latent.shape[3]), mode='bilinear')

    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]

    # Structure Qwen-Image-Edit inputs:
    cond_dict = {"concat_latent_image": cond_latent, "concat_mask": latent_mask}
    
    pos[0][1].update(cond_dict)
    neg[0][1].update(cond_dict)

    # 3) We CANNOT use SetLatentNoiseMask! If we do, the unmasked area stays at 0 noise, 
    # creating a mathematically sharp discontinuous noise hole that blinds Qwen's attention.
    # We must use smooth, global noise and rely purely on our post-process 
    # pixel-space alpha blend to protect exactly the background.
    latent = latent_raw
    
    # 4) ALWAYS force denoise = 1.0. Instruct models will produce blurry smears 
    # if their diffusion trajectory is interrupted midway (like 0.75 denoise).
    denoise = 1.0

    samples = n["KSampler"].sample(
        _unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=1.0
    )[0]

    decoded = _vae_decode(samples).detach()
    result_np = np.array(decoded * 255, dtype=np.uint8)[0]
    result_pil = Image.fromarray(result_np).resize(original.size, Image.LANCZOS)

    # Composite back using full image bounds
    result = np.array(original).copy()
    mask_composite = Image.fromarray(mask_combined)
    mask_float = np.array(mask_composite).astype(np.float32)[:, :, None] / 255.0

    mask_blur = Image.fromarray((mask_float[:, :, 0] * 255).astype(np.uint8))
    mask_blur = mask_blur.filter(ImageFilter.GaussianBlur(3))
    mask_float = np.array(mask_blur).astype(np.float32)[:, :, None] / 255.0

    old_region = result.astype(np.float32)
    new_region = np.array(result_pil).astype(np.float32)
    blended = new_region * mask_float + old_region * (1 - mask_float)
    result = blended.clip(0, 255).astype(np.uint8)

    return Image.fromarray(result)
