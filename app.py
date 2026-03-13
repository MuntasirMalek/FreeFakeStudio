# ============================================================
#  Z-Fooocus — Fooocus-style UI for Z-Image Turbo
#  Generate · Img2Img · Inpaint — All using ONE model
#  Built for Google Colab T4 (15GB VRAM)
# ============================================================

import os, random, time, sys, gc
import torch
import numpy as np
from PIL import Image, ImageFilter
import re, uuid
import gradio as gr

sys.path.insert(0, "/content/ComfyUI")
from nodes import NODE_CLASS_MAPPINGS

# ============================================================
#  Load ComfyUI Nodes
# ============================================================
UNETLoader       = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader       = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader        = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode   = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler_Node    = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode        = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode        = NODE_CLASS_MAPPINGS["VAEEncode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
SetLatentNoiseMask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()

# ============================================================
#  Load Z-Image Turbo FP8 (one model for everything)
# ============================================================
print("⏳ Loading Z-Image Turbo FP8...")
with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae  = VAELoader.load_vae("ae.safetensors")[0]
print("✅ Z-Image Turbo loaded!")

# ============================================================
#  Helpers
# ============================================================
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_save_path(prefix="img"):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)[:20]
    uid = uuid.uuid4().hex[:6]
    return os.path.join(SAVE_DIR, f"{safe}_{uid}.png")

def make_seed(seed):
    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 2**63)
    return int(seed)

def pil_to_tensor(img):
    """PIL Image → ComfyUI tensor (B, H, W, C) float32 0-1"""
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

# ============================================================
#  Fooocus-style inpaint helpers
#  (simplified from Fooocus/modules/inpaint_worker.py)
# ============================================================
def compute_crop_region(mask_np, padding=0.15):
    """Find bounding box of mask and add padding (Fooocus-style)."""
    indices = np.where(mask_np > 0)
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        return None  # no mask painted
    a, b = np.min(indices[0]), np.max(indices[0])
    c, d = np.min(indices[1]), np.max(indices[1])
    # Add padding around the mask (Fooocus uses 1.15x)
    h_center, h_half = (b + a) // 2, (b - a) // 2
    w_center, w_half = (d + c) // 2, (d - c) // 2
    size = int(max(h_half, w_half) * (1.0 + padding))
    a = max(0, h_center - size)
    b = min(mask_np.shape[0], h_center + size + 1)
    c = max(0, w_center - size)
    d = min(mask_np.shape[1], w_center + size + 1)
    return (a, b, c, d)

def fooocus_fill(image_np, mask_np):
    """Fooocus-style iterative blur fill for masked region."""
    current = image_np.copy()
    raw = image_np.copy()
    area = np.where(mask_np < 127)
    store = raw[area]
    for k, repeats in [(512, 2), (256, 2), (128, 4), (64, 4), (33, 8), (15, 8), (5, 16), (3, 16)]:
        for _ in range(repeats):
            pil_img = Image.fromarray(current)
            pil_img = pil_img.filter(ImageFilter.BoxBlur(k))
            current = np.array(pil_img)
            current[area] = store
    return current

def resize_to_multiple(img, multiple=64, max_dim=1024):
    """Resize image so both dimensions are multiples of 64, max 1024."""
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = int(w * scale) // multiple * multiple
    new_h = int(h * scale) // multiple * multiple
    new_w = max(multiple, new_w)
    new_h = max(multiple, new_h)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ============================================================
#  Tab 1: Generate (text → image)
# ============================================================
@torch.inference_mode()
def generate_image(prompt, negative, aspect_ratio, seed, steps, cfg, denoise):
    seed = make_seed(seed)
    width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]

    pos = CLIPTextEncode.encode(clip, prompt)[0]
    neg = CLIPTextEncode.encode(clip, negative)[0]
    latent = EmptyLatentImage.generate(width, height, batch_size=1)[0]
    samples = KSampler_Node.sample(
        unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    path = get_save_path("gen")
    Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(path)
    return path, path, str(seed)

# ============================================================
#  Tab 2: Img2Img (image + prompt → restyled image)
# ============================================================
@torch.inference_mode()
def img2img(input_image, prompt, negative, seed, steps, cfg, denoise):
    if input_image is None:
        raise gr.Error("Upload an image first!")
    seed = make_seed(seed)

    input_image = resize_to_multiple(input_image)
    img_tensor = pil_to_tensor(input_image)

    pos = CLIPTextEncode.encode(clip, prompt)[0]
    neg = CLIPTextEncode.encode(clip, negative)[0]
    latent = VAEEncode.encode(vae, img_tensor)[0]
    samples = KSampler_Node.sample(
        unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    path = get_save_path("i2i")
    Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(path)
    return path, path, str(seed)

# ============================================================
#  Tab 3: Inpaint (Fooocus-style: crop → fill → denoise → composite)
# ============================================================
@torch.inference_mode()
def inpaint(editor_data, prompt, negative, seed, steps, cfg, denoise):
    if editor_data is None:
        raise gr.Error("Upload and paint on an image first!")
    seed = make_seed(seed)

    # --- Extract image and mask from Gradio ImageEditor ---
    if isinstance(editor_data, dict):
        bg = editor_data.get("background")
        layers = editor_data.get("layers", [])
        composite = editor_data.get("composite")
    else:
        raise gr.Error("Unexpected input format.")

    if bg is None:
        raise gr.Error("No background image found.")
    if not isinstance(bg, Image.Image):
        bg = Image.fromarray(bg)

    original = bg.convert("RGB")
    orig_w, orig_h = original.size

    # Build binary mask from painted layers
    mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for layer in layers:
        if not isinstance(layer, Image.Image):
            layer = Image.fromarray(layer)
        layer = layer.resize((orig_w, orig_h))
        arr = np.array(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            mask_combined = np.maximum(mask_combined, arr[:, :, 3])
        elif arr.ndim == 3:
            mask_combined = np.maximum(mask_combined, np.mean(arr[:, :, :3], axis=2).astype(np.uint8))
        elif arr.ndim == 2:
            mask_combined = np.maximum(mask_combined, arr)

    if np.sum(mask_combined > 0) == 0:
        raise gr.Error("You didn't paint anything! Paint over the area you want to edit.")

    # --- Fooocus-style inpaint pipeline ---
    # Step 1: Find crop region around mask
    crop = compute_crop_region(mask_combined)
    if crop is None:
        raise gr.Error("No mask detected.")
    a, b, c, d = crop

    # Step 2: Crop to interested area
    cropped_img = np.array(original)[a:b, c:d]
    cropped_mask = mask_combined[a:b, c:d]

    # Step 3: Resize cropped region
    crop_pil = Image.fromarray(cropped_img)
    crop_pil = resize_to_multiple(crop_pil, multiple=64, max_dim=1024)
    cw, ch = crop_pil.size

    # Resize mask to match
    mask_pil = Image.fromarray(cropped_mask).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    # Step 4: Fooocus fill (blur out masked area for clean inpaint)
    filled = fooocus_fill(np.array(crop_pil), mask_resized)
    filled_pil = Image.fromarray(filled)

    # Step 5: VAE encode the filled image
    filled_tensor = pil_to_tensor(filled_pil)
    latent = VAEEncode.encode(vae, filled_tensor)[0]

    # Step 6: Apply mask to latent (only denoise masked region)
    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0)
    latent = SetLatentNoiseMask.set_mask(latent, mask_tensor)[0]

    # Step 7: KSampler — regenerate masked area
    pos = CLIPTextEncode.encode(clip, prompt)[0]
    neg = CLIPTextEncode.encode(clip, negative)[0]
    samples = KSampler_Node.sample(
        unet, seed, int(steps), float(cfg),
        "euler", "simple", pos, neg, latent, denoise=float(denoise)
    )[0]

    # Step 8: VAE decode
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    result_crop = np.array(decoded * 255, dtype=np.uint8)[0]
    result_crop_pil = Image.fromarray(result_crop).resize((d - c, b - a), Image.LANCZOS)

    # Step 9: Composite back onto original (Fooocus-style)
    result = np.array(original).copy()
    result_crop_np = np.array(result_crop_pil)
    mask_composite = Image.fromarray(cropped_mask).resize((d - c, b - a), Image.LANCZOS)
    mask_float = np.array(mask_composite).astype(np.float32)[:, :, None] / 255.0

    # Feather the edges for smooth blending
    mask_blur = Image.fromarray((mask_float[:, :, 0] * 255).astype(np.uint8))
    mask_blur = mask_blur.filter(ImageFilter.GaussianBlur(3))
    mask_float = np.array(mask_blur).astype(np.float32)[:, :, None] / 255.0

    old_region = result[a:b, c:d].astype(np.float32)
    new_region = result_crop_np.astype(np.float32)
    blended = new_region * mask_float + old_region * (1 - mask_float)
    result[a:b, c:d] = blended.clip(0, 255).astype(np.uint8)

    path = get_save_path("inpaint")
    Image.fromarray(result).save(path)
    return path, path, str(seed)


# ============================================================
#  UI — Fooocus-Style
# ============================================================

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]
DEFAULT_NEG = "low quality, blurry, pixelated, noise, watermark, text, logo"

CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1400px !important;
}
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8em;
    font-weight: 800;
    margin: 10px 0 0 0;
}
.subtitle {
    text-align: center; color: #888; margin-bottom: 15px; font-size: 1.05em;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="Z-Fooocus") as demo:

    gr.HTML("""
    <div>
        <h1 class="main-title">⚡ Z-Fooocus</h1>
        <p class="subtitle">Generate · Img2Img · Inpaint — Powered by Z-Image Turbo</p>
    </div>
    """)

    with gr.Tabs():

        # ─── GENERATE ─────────────────────────────────────────
        with gr.Tab("🖼️ Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gen_prompt = gr.Textbox(
                        label="✨ Prompt", lines=4,
                        placeholder="Describe the image you want...",
                        value="A cinematic portrait of an astronaut riding a white horse across a golden wheat field at sunset, 8K, ultra detailed"
                    )
                    with gr.Row():
                        gen_aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="📐 Aspect Ratio")
                        gen_seed = gr.Number(value=0, label="🎲 Seed (0=random)", precision=0)
                    gen_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        gen_steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
                        gen_cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                        gen_denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                        gen_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                with gr.Column(scale=1):
                    gen_out = gr.Image(label="Result", height=512)
                    gen_dl = gr.File(label="📥 Download")
                    gen_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)
            gen_btn.click(generate_image,
                [gen_prompt, gen_neg, gen_aspect, gen_seed, gen_steps, gen_cfg, gen_denoise],
                [gen_dl, gen_out, gen_seed_out])

        # ─── IMG2IMG ──────────────────────────────────────────
        with gr.Tab("✏️ Img2Img"):
            gr.Markdown("Upload a photo and describe how to restyle it. Lower denoise = more faithful to original.")
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_img = gr.Image(type="pil", label="📸 Upload Photo", height=400)
                    i2i_prompt = gr.Textbox(label="✨ Prompt", lines=2,
                        placeholder="e.g., oil painting style, vibrant colors")
                    i2i_btn = gr.Button("✨ Transform", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        i2i_steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
                        i2i_cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                        i2i_denoise = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Denoise (lower = more original)")
                        i2i_seed = gr.Number(value=0, label="🎲 Seed (0=random)", precision=0)
                        i2i_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                with gr.Column(scale=1):
                    i2i_out = gr.Image(label="Result", height=512)
                    i2i_dl = gr.File(label="📥 Download")
                    i2i_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)
            i2i_btn.click(img2img,
                [i2i_img, i2i_prompt, i2i_neg, i2i_seed, i2i_steps, i2i_cfg, i2i_denoise],
                [i2i_dl, i2i_out, i2i_seed_out])

        # ─── INPAINT ─────────────────────────────────────────
        with gr.Tab("🎨 Inpaint"):
            gr.Markdown("Upload a photo, **paint over the area** you want to change, then describe what should replace it. Only the painted area gets modified!")
            with gr.Row():
                with gr.Column(scale=1):
                    inp_editor = gr.ImageEditor(
                        label="🖌️ Upload & Paint",
                        type="pil", height=450,
                        brush=gr.Brush(colors=["#ffffff"], default_size=30),
                        eraser=gr.Eraser(default_size=20),
                    )
                    inp_prompt = gr.Textbox(label="✨ What should the painted area become?", lines=2,
                        placeholder="e.g., a blue sky with clouds")
                    inp_btn = gr.Button("🎨 Inpaint", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        inp_steps = gr.Slider(4, 25, value=12, step=1, label="Steps")
                        inp_cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                        inp_denoise = gr.Slider(0.1, 1.0, value=0.80, step=0.05, label="Denoise")
                        inp_seed = gr.Number(value=0, label="🎲 Seed (0=random)", precision=0)
                        inp_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                with gr.Column(scale=1):
                    inp_out = gr.Image(label="Result", height=512)
                    inp_dl = gr.File(label="📥 Download")
                    inp_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)
            inp_btn.click(inpaint,
                [inp_editor, inp_prompt, inp_neg, inp_seed, inp_steps, inp_cfg, inp_denoise],
                [inp_dl, inp_out, inp_seed_out])

    gr.Markdown("""
    ---
    <p style="text-align:center; color:#999; font-size:0.85em;">
    ⚡ Z-Fooocus · <a href="https://github.com/Tongyi-MAI/Z-Image">Z-Image</a>
    · Inpaint inspired by <a href="https://github.com/lllyasviel/Fooocus">Fooocus</a>
    · Based on <a href="https://github.com/NeuralFalconYT/Z-Image-Colab">NeuralFalconYT</a>
    </p>
    """)

demo.launch(share=True, debug=True)
