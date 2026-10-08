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

def _fix_comfy_app_namespace(comfy_dir):
    comfy_app = os.path.join(comfy_dir, "app")
    if os.path.exists(comfy_app):
        import sys, types, importlib.util
        if "app" in sys.modules:
            if not hasattr(sys.modules["app"], "__path__"):
                sys.modules["app"].__path__ = [comfy_app]
            elif comfy_app not in sys.modules["app"].__path__:
                sys.modules["app"].__path__.append(comfy_app)
        else:
            app_pkg = types.ModuleType("app")
            app_pkg.__path__ = [comfy_app]
            sys.modules["app"] = app_pkg

        gov_file = os.path.join(comfy_app, "governance.py")
        if os.path.exists(gov_file):
            try:
                spec = importlib.util.spec_from_file_location("app.governance", gov_file)
                if spec and spec.loader:
                    gov_mod = importlib.util.module_from_spec(spec)
                    sys.modules["app.governance"] = gov_mod
                    spec.loader.exec_module(gov_mod)
                    if "app" in sys.modules:
                        setattr(sys.modules["app"], "governance", gov_mod)
            except Exception:
                pass

# ── Node references (set once) ─────────────────────────────
def _get_nodes():
    global _nodes
    if not _nodes:
        import sys, os
        comfy_dir = os.environ.get("COMFY_DIR", "/content/ComfyUI")
        for p in [comfy_dir, "/content/ComfyUI", "/kaggle/working/ComfyUI", os.path.abspath("./ComfyUI"), os.path.join(os.path.dirname(__file__), "ComfyUI")]:
            if os.path.exists(p):
                if p not in sys.path:
                    sys.path.insert(0, p)
                comfy_dir = p
                break
        _fix_comfy_app_namespace(comfy_dir)
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
                f"ComfyUI-GGUF custom nodes not found! "
                f"Install them: git clone https://github.com/city96/ComfyUI-GGUF.git "
                f"{os.path.join(comfy_dir, 'custom_nodes', 'ComfyUI-GGUF')}"
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
    print("⏳ Loading ERNIE-Image-Turbo (GGUF Q6_K)...")
    with torch.inference_mode():
        _unet = n["UnetLoaderGGUF"].load_unet("ernie-image-turbo-Q6_K.gguf")[0]
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
