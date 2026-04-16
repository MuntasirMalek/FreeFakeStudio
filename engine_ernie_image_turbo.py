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
        if "/content/ComfyUI" not in sys.path:
            sys.path.insert(0, "/content/ComfyUI")
        from nodes import NODE_CLASS_MAPPINGS

        # Import ComfyUI-GGUF nodes via importlib (same approach as Qwen engine)
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

        if "UnetLoaderGGUF" not in all_nodes:
            raise RuntimeError(
                "ComfyUI-GGUF custom nodes not found! "
                "Install them: git clone https://github.com/city96/ComfyUI-GGUF.git "
                "/content/ComfyUI/custom_nodes/ComfyUI-GGUF"
            )

        _nodes = {
            "UnetLoaderGGUF":   all_nodes["UnetLoaderGGUF"](),
            "CLIPLoader":       all_nodes["CLIPLoader"](),
            "VAELoader":        all_nodes["VAELoader"](),
            "CLIPTextEncode":   all_nodes["CLIPTextEncode"](),
            "KSampler":         all_nodes["KSampler"](),
            "VAEDecode":        all_nodes["VAEDecode"](),
            "EmptyLatentImage": all_nodes["EmptyLatentImage"](),
        }

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
