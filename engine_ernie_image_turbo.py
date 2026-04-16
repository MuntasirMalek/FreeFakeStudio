# ============================================================
#  ERNIE-Image-Turbo  —  Generate-only engine (GGUF)
#  Uses: UnetLoaderGGUF + Ministral 3.3B CLIP + Flux2 VAE
#  8 inference steps, CFG 1.0, euler/simple
# ============================================================
import gc, torch, numpy as np
from PIL import Image

_loaded = False
_unet = None
_clip = None
_vae = None
_nodes = {}

# ── Node references (set once) ─────────────────────────────
def _get_nodes():
    global _nodes
    if not _nodes:
        import sys
        sys.path.insert(0, "/content/ComfyUI")
        from nodes import NODE_CLASS_MAPPINGS

        # Standard ComfyUI nodes
        _nodes = {
            "CLIPLoader":       NODE_CLASS_MAPPINGS["CLIPLoader"](),
            "VAELoader":        NODE_CLASS_MAPPINGS["VAELoader"](),
            "CLIPTextEncode":   NODE_CLASS_MAPPINGS["CLIPTextEncode"](),
            "KSampler":         NODE_CLASS_MAPPINGS["KSampler"](),
            "VAEDecode":        NODE_CLASS_MAPPINGS["VAEDecode"](),
            "EmptyLatentImage": NODE_CLASS_MAPPINGS["EmptyLatentImage"](),
        }

        # ComfyUI-GGUF node for loading GGUF diffusion model
        try:
            import nodes
            if hasattr(nodes, "init_extra_nodes"):
                nodes.init_extra_nodes()
            from nodes import NODE_CLASS_MAPPINGS as ALL_NODES
            if "UnetLoaderGGUF" in ALL_NODES:
                _nodes["UnetLoaderGGUF"] = ALL_NODES["UnetLoaderGGUF"]()
        except Exception:
            pass

        if "UnetLoaderGGUF" not in _nodes:
            try:
                from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
                _nodes["UnetLoaderGGUF"] = UnetLoaderGGUF()
            except ImportError:
                raise RuntimeError(
                    "ComfyUI-GGUF custom nodes not found! "
                    "Install them: git clone https://github.com/city96/ComfyUI-GGUF.git "
                    "/content/ComfyUI/custom_nodes/ComfyUI-GGUF"
                )

    return _nodes

# ── Load / Unload ──────────────────────────────────────────
def load():
    global _loaded, _unet, _clip, _vae
    if _loaded:
        return
    n = _get_nodes()
    print("⏳ Loading ERNIE-Image-Turbo (GGUF Q4_K_M)...")
    with torch.inference_mode():
        _unet = n["UnetLoaderGGUF"].load_unet("ernie-image-turbo-Q4_K_M.gguf")[0]
        _clip = n["CLIPLoader"].load_clip("ministral-3-3b.safetensors", type="ernie_image")[0]
        _vae  = n["VAELoader"].load_vae("flux2-vae.safetensors")[0]
    _loaded = True
    print("✅ ERNIE-Image-Turbo loaded!")

def unload():
    global _loaded, _unet, _clip, _vae
    _unet = None
    _clip = None
    _vae = None
    _loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🗑️ ERNIE-Image-Turbo unloaded")

def is_loaded():
    return _loaded

# ── Generate ───────────────────────────────────────────────
@torch.inference_mode()
def generate(prompt, negative, width, height, seed, cfg, denoise, steps=8):
    n = _get_nodes()
    pos = n["CLIPTextEncode"].encode(_clip, prompt)[0]
    neg = n["CLIPTextEncode"].encode(_clip, negative)[0]
    latent = n["EmptyLatentImage"].generate(width, height, batch_size=1)[0]
    samples = n["KSampler"].sample(
        _unet, seed, min(steps, 8), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = n["VAEDecode"].decode(_vae, samples)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
