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
import cv2

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
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

# ============================================================
#  Fooocus-style inpaint helpers
# ============================================================
def compute_crop_region(mask_np, padding=0.30):
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

def fooocus_fill(image_np, mask_np):
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
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    new_w = int(w * scale) // multiple * multiple
    new_h = int(h * scale) // multiple * multiple
    new_w = max(multiple, new_w)
    new_h = max(multiple, new_h)
    return img.resize((new_w, new_h), Image.LANCZOS)

def enhance_inpaint_prompt(prompt):
    """Like Fooocus — auto-add quality boosters to inpaint prompts."""
    boosters = "photorealistic, high quality, detailed, natural lighting, 8K"
    if any(b in prompt.lower() for b in ["realistic", "quality", "detailed", "8k", "4k"]):
        return prompt  # user already added quality terms
    return f"{prompt}, {boosters}"

# ============================================================
#  Auto-mask helpers (rembg + OpenCV face detection)
# ============================================================
_rembg_session = None
def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session

def auto_mask_background(image_pil):
    """Returns mask where background=255 (white), person=0 (black)."""
    from rembg import remove
    session = get_rembg_session()
    # rembg returns RGBA where alpha=person
    result = remove(image_pil, session=session, only_mask=True)
    mask_np = np.array(result)  # person=255, bg=0
    bg_mask = 255 - mask_np     # invert: bg=255, person=0
    return bg_mask

def auto_mask_except_face(image_pil):
    """Returns mask where everything=255 EXCEPT face region=0."""
    from rembg import remove
    img_np = np.array(image_pil.convert("RGB"))

    # Detect faces with OpenCV
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    h, w = img_np.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255  # start with everything masked

    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            # Add padding around face (30% extra)
            pad_w = int(fw * 0.3)
            pad_h = int(fh * 0.3)
            x1 = max(0, fx - pad_w)
            y1 = max(0, fy - pad_h)
            x2 = min(w, fx + fw + pad_w)
            y2 = min(h, fy + fh + pad_h)
            # Create elliptical mask for natural face shape
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            axes_x = (x2 - x1) // 2
            axes_y = (y2 - y1) // 2
            cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 0, -1)
    else:
        # Fallback: if no face found, just mask background
        mask = auto_mask_background(image_pil)

    return mask

def generate_auto_mask(image_pil, mask_mode):
    """Generate auto mask and return as a preview image."""
    if image_pil is None:
        return None
    if mask_mode == "🏞️ Background Only":
        mask_np = auto_mask_background(image_pil)
    elif mask_mode == "🎭 Everything Except Face":
        mask_np = auto_mask_except_face(image_pil)
    else:
        return None

    # Create red overlay preview
    img_np = np.array(image_pil.convert("RGB"))
    overlay = img_np.copy()
    mask_bool = mask_np > 127
    overlay[mask_bool] = (overlay[mask_bool] * 0.4 + np.array([255, 50, 50]) * 0.6).astype(np.uint8)
    return Image.fromarray(overlay)


# ============================================================
#  Tab 1: Generate
# ============================================================
@torch.inference_mode()
def generate_image(prompt, negative, aspect_ratio, seed, cfg, denoise, num_images):
    seed = make_seed(seed)
    width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]
    num = int(num_images)

    pos = CLIPTextEncode.encode(clip, prompt)[0]
    neg = CLIPTextEncode.encode(clip, negative)[0]

    images, paths = [], []
    current_seed = seed
    for i in range(num):
        latent = EmptyLatentImage.generate(width, height, batch_size=1)[0]
        samples = KSampler_Node.sample(
            unet, current_seed, 8, float(cfg),
            "euler", "simple", pos, neg, latent, denoise=float(denoise)
        )[0]
        decoded = VAEDecode.decode(vae, samples)[0].detach()
        path = get_save_path("gen")
        img = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
        img.save(path)
        images.append(img)
        paths.append(path)
        current_seed += 1

    return images, paths, str(seed)

# ============================================================
#  Tab 2: Img2Img
# ============================================================
@torch.inference_mode()
def img2img(input_image, prompt, negative, seed, cfg, denoise, num_images):
    if input_image is None:
        raise gr.Error("Upload an image first!")
    seed = make_seed(seed)
    num = int(num_images)

    input_image = resize_to_multiple(input_image)
    img_tensor = pil_to_tensor(input_image)

    pos = CLIPTextEncode.encode(clip, prompt)[0]
    neg = CLIPTextEncode.encode(clip, negative)[0]
    latent = VAEEncode.encode(vae, img_tensor)[0]

    images, paths = [], []
    current_seed = seed
    for i in range(num):
        samples = KSampler_Node.sample(
            unet, current_seed, 8, float(cfg),
            "euler", "simple", pos, neg, latent, denoise=float(denoise)
        )[0]
        decoded = VAEDecode.decode(vae, samples)[0].detach()
        path = get_save_path("i2i")
        img = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
        img.save(path)
        images.append(img)
        paths.append(path)
        current_seed += 1

    return images, paths, str(seed)

# ============================================================
#  Tab 3: Inpaint — supports manual paint + auto-mask modes
# ============================================================
@torch.inference_mode()
def inpaint(editor_data, inp_image, prompt, negative, seed, cfg, denoise, num_images, mask_mode, auto_mask_data):
    seed = make_seed(seed)
    num = int(num_images)

    # --- Determine image and mask based on mode ---
    if mask_mode == "🖌️ Manual Paint":
        if editor_data is None:
            raise gr.Error("Upload and paint on an image first!")
        if isinstance(editor_data, dict):
            bg = editor_data.get("background")
            layers = editor_data.get("layers", [])
        else:
            raise gr.Error("Unexpected input format.")

        if bg is None:
            raise gr.Error("No background image found.")
        if not isinstance(bg, Image.Image):
            bg = Image.fromarray(bg)

        # Check if user actually painted anything
        manual_mask = np.zeros((bg.size[1], bg.size[0]), dtype=np.uint8)
        for layer in layers:
            if not isinstance(layer, Image.Image):
                layer = Image.fromarray(layer)
            layer = layer.resize(bg.size)
            arr = np.array(layer)
            if arr.ndim == 3 and arr.shape[2] == 4:
                manual_mask = np.maximum(manual_mask, arr[:, :, 3])
            elif arr.ndim == 3:
                manual_mask = np.maximum(manual_mask, np.mean(arr[:, :, :3], axis=2).astype(np.uint8))
            elif arr.ndim == 2:
                manual_mask = np.maximum(manual_mask, arr)

        has_manual_paint = np.sum(manual_mask > 0) > 0

        if has_manual_paint:
            # User painted their own mask — use it
            # Use the stored original image if available (editor bg might have overlay)
            if auto_mask_data and auto_mask_data.get("original") is not None:
                original = auto_mask_data["original"]
            else:
                original = bg.convert("RGB")
            mask_combined = manual_mask
        elif auto_mask_data and auto_mask_data.get("mask") is not None:
            # User didn't paint but there's a stored auto-mask — use it
            original = auto_mask_data["original"]
            mask_combined = auto_mask_data["mask"]
            # Resize to match if needed
            if mask_combined.shape[:2] != (original.size[1], original.size[0]):
                mask_combined = np.array(Image.fromarray(mask_combined).resize(original.size, Image.NEAREST))
        else:
            raise gr.Error("Paint the areas you want to change, or use auto-mask first!")

        orig_w, orig_h = original.size

    else:
        # Auto-mask mode (direct, no editing)
        if inp_image is None:
            raise gr.Error("Upload an image first!")
        if not isinstance(inp_image, Image.Image):
            inp_image = Image.fromarray(inp_image)
        original = inp_image.convert("RGB")
        orig_w, orig_h = original.size

        if mask_mode == "🏞️ Background Only":
            mask_combined = auto_mask_background(original)
        elif mask_mode == "🎭 Everything Except Face":
            mask_combined = auto_mask_except_face(original)
        else:
            raise gr.Error("Unknown mask mode.")

    # --- Fooocus-style inpaint pipeline ---
    crop = compute_crop_region(mask_combined)
    if crop is None:
        raise gr.Error("No mask detected.")
    a, b, c, d = crop

    cropped_img = np.array(original)[a:b, c:d]
    cropped_mask = mask_combined[a:b, c:d]

    crop_pil = Image.fromarray(cropped_img)
    crop_pil = resize_to_multiple(crop_pil, multiple=64, max_dim=1024)
    cw, ch = crop_pil.size

    mask_pil = Image.fromarray(cropped_mask).resize((cw, ch), Image.NEAREST)
    mask_resized = np.array(mask_pil)

    filled = fooocus_fill(np.array(crop_pil), mask_resized)
    filled_pil = Image.fromarray(filled)

    filled_tensor = pil_to_tensor(filled_pil)
    latent_base = VAEEncode.encode(vae, filled_tensor)[0]

    mask_tensor = torch.from_numpy(mask_resized.astype(np.float32) / 255.0).unsqueeze(0)

    prompt_enhanced = enhance_inpaint_prompt(prompt)
    pos = CLIPTextEncode.encode(clip, prompt_enhanced)[0]
    neg = CLIPTextEncode.encode(clip, negative)[0]

    images, paths = [], []
    current_seed = seed
    for i in range(num):
        latent = SetLatentNoiseMask.set_mask(latent_base, mask_tensor)[0]
        samples = KSampler_Node.sample(
            unet, current_seed, 8, float(cfg),
            "euler", "simple", pos, neg, latent, denoise=float(denoise)
        )[0]

        decoded = VAEDecode.decode(vae, samples)[0].detach()
        result_crop = np.array(decoded * 255, dtype=np.uint8)[0]
        result_crop_pil = Image.fromarray(result_crop).resize((d - c, b - a), Image.LANCZOS)

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

        path = get_save_path("inpaint")
        Image.fromarray(result).save(path)
        images.append(Image.fromarray(result))
        paths.append(path)
        current_seed += 1

    return images, paths, str(seed)


# ============================================================
#  UI
# ============================================================

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]
DEFAULT_NEG = "low quality, blurry, pixelated, noise, watermark, text, logo"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1400px !important;
}
.main-title {
    text-align: center; font-size: 2.8em; font-weight: 800;
    margin: 10px 0 0 0; letter-spacing: -0.02em;
}
.main-title span {
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center; color: #9ca3af; margin-bottom: 15px; font-size: 1.05em;
}
.footer-link { color: #60a5fa !important; text-decoration: none; }
.footer-link:hover { text-decoration: underline; }

/* Hide the move/pan tool from ImageEditor toolbar */
.image-editor .toolbar button[aria-label="Move"],
.image-editor .toolbar button:has(svg path[d*="M13"]):last-of-type {
    display: none !important;
}
"""

zfooocus_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

def toggle_inpaint_inputs(mode):
    """Show/hide editor vs image upload based on mask mode."""
    if mode == "🖌️ Manual Paint":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def edit_mask_manually(image, mask_mode):
    """Store auto-mask and load image with overlay into editor for refinement."""
    if image is None:
        raise gr.Error("Upload an image first!")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    original = image.convert("RGB")
    w, h = original.size

    # Generate the auto mask
    if mask_mode == "🏞️ Background Only":
        mask_np = auto_mask_background(original)
    elif mask_mode == "🎭 Everything Except Face":
        mask_np = auto_mask_except_face(original)
    else:
        raise gr.Error("Select an auto mask mode first.")

    # Store auto-mask + original in state
    auto_mask_data = {"mask": mask_np, "original": original}

    # Create a visual overlay on the image showing the mask
    # Darken masked areas so user can see what will be changed
    img_np = np.array(original).copy()
    mask_bool = mask_np > 127
    # Tint masked areas slightly red so user knows what's masked
    overlay = img_np.astype(np.float32)
    overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 180, 0, 255)  # red
    overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.4  # dim green
    overlay[mask_bool, 2] = overlay[mask_bool, 2] * 0.4  # dim blue
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))

    editor_value = {"background": overlay_pil, "layers": [], "composite": overlay_pil}
    return (
        editor_value,                    # inp_editor (with overlay)
        "🖌️ Manual Paint",               # switch mode
        gr.update(visible=True),          # show editor
        gr.update(visible=False),         # hide image upload
        gr.update(visible=False),         # hide mask preview
        gr.update(visible=False),         # hide edit mask button
        auto_mask_data,                   # store in State
    )

def preview_auto_mask(image, mask_mode):
    """Generate and show auto-mask preview."""
    if image is None or mask_mode == "🖌️ Manual Paint":
        return None
    return generate_auto_mask(image, mask_mode)

with gr.Blocks(theme=zfooocus_theme, css=CSS, title="Z-Fooocus") as demo:

    gr.HTML("""
    <div>
        <h1 class="main-title">⚡ <span>Z-Fooocus</span></h1>
        <p class="subtitle">Generate · Img2Img · Inpaint — Powered by Z-Image Turbo</p>
    </div>
    """)

    with gr.Tabs():

        # ─── GENERATE ─────────────────────────────────────────
        with gr.Tab("🖼️ Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gen_prompt = gr.Textbox(
                        label="Prompt", lines=4,
                        placeholder="Describe the image you want...",
                        value="A cinematic portrait of an astronaut riding a white horse across a golden wheat field at sunset, 8K, ultra detailed"
                    )
                    with gr.Row():
                        gen_aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
                        gen_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                    gen_num = gr.Slider(1, 16, value=2, step=1, label="Number of Images")
                    gen_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        gen_cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                        gen_denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                        gen_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                with gr.Column(scale=1):
                    gen_gallery = gr.Gallery(label="Results", columns=2, height=520,
                                             object_fit="contain", show_download_button=True,
                                             show_fullscreen_button=True)
                    gen_dl = gr.File(label="Download All", file_count="multiple")
                    gen_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)

            gen_btn.click(lambda: ([], None, ""), outputs=[gen_gallery, gen_dl, gen_seed_out]).then(
                generate_image,
                [gen_prompt, gen_neg, gen_aspect, gen_seed, gen_cfg, gen_denoise, gen_num],
                [gen_gallery, gen_dl, gen_seed_out])

        # ─── IMG2IMG ──────────────────────────────────────────
        with gr.Tab("✏️ Img2Img"):
            gr.Markdown("Upload a photo and describe how to restyle it. Lower denoise = more faithful to original.")
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_img = gr.Image(type="pil", label="Upload Photo", height=400, sources=["upload"])
                    i2i_prompt = gr.Textbox(label="Prompt", lines=2,
                        placeholder="e.g., oil painting style, vibrant colors")
                    i2i_num = gr.Slider(1, 16, value=2, step=1, label="Number of Images")
                    i2i_btn = gr.Button("✨ Transform", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        i2i_cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                        i2i_denoise = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Denoise (lower = more original)")
                        i2i_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                        i2i_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                    i2i_clear = gr.ClearButton([i2i_img], value="🗑️ Clear Image")
                with gr.Column(scale=1):
                    i2i_gallery = gr.Gallery(label="Results", columns=2, height=520,
                                              object_fit="contain", show_download_button=True,
                                              show_fullscreen_button=True)
                    i2i_dl = gr.File(label="Download All", file_count="multiple")
                    i2i_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)

            i2i_btn.click(lambda: ([], None, ""), outputs=[i2i_gallery, i2i_dl, i2i_seed_out]).then(
                img2img,
                [i2i_img, i2i_prompt, i2i_neg, i2i_seed, i2i_cfg, i2i_denoise, i2i_num],
                [i2i_gallery, i2i_dl, i2i_seed_out])

        # ─── INPAINT ─────────────────────────────────────────
        with gr.Tab("🎨 Inpaint"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Mask mode selector
                    inp_mask_mode = gr.Radio(
                        choices=["🖌️ Manual Paint", "🏞️ Background Only", "🎭 Everything Except Face"],
                        value="🖌️ Manual Paint",
                        label="Mask Mode",
                    )

                    # Manual paint editor (shown by default)
                    inp_editor = gr.ImageEditor(
                        label="Upload & Paint Mask",
                        type="pil", height=450,
                        brush=gr.Brush(colors=["#ffffff"], default_size=40),
                        eraser=gr.Eraser(default_size=30),
                        sources=["upload"],
                        transforms=[],
                        visible=True,
                    )

                    # Auto-mask image upload (hidden by default)
                    inp_image = gr.Image(type="pil", label="Upload Photo", height=400, visible=False, sources=["upload"])
                    # Auto-mask preview
                    inp_mask_preview = gr.Image(label="🔴 Mask Preview (red = will be changed)", height=300, visible=False, interactive=False)
                    # Button to send auto-mask to manual editor for refinement
                    inp_edit_mask_btn = gr.Button("✏️ Edit This Mask Manually", variant="secondary", visible=False)

                    inp_prompt = gr.Textbox(label="What should the masked area become?", lines=2,
                        placeholder="e.g., a tropical beach background")
                    inp_num = gr.Slider(1, 16, value=2, step=1, label="Number of Images")
                    inp_btn = gr.Button("🎨 Inpaint", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        inp_cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                        inp_denoise = gr.Slider(0.1, 1.0, value=0.60, step=0.05, label="Denoise (lower = more realistic)")
                        inp_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                        inp_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                    inp_clear = gr.ClearButton([inp_editor, inp_image, inp_mask_preview], value="🗑️ Clear All")
                with gr.Column(scale=1):
                    inp_gallery = gr.Gallery(label="Results", columns=2, height=520,
                                              object_fit="contain", show_download_button=True,
                                              show_fullscreen_button=True)
                    inp_send_btn = gr.Button("🖌️ Send to Paint Editor for Touch-up", variant="secondary")
                    inp_dl = gr.File(label="Download All", file_count="multiple")
                    inp_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)
                    # Hidden state to store selected image index
                    inp_selected_idx = gr.State(value=0)
                    # Hidden state to store auto-mask data
                    inp_auto_mask_state = gr.State(value=None)

            # Toggle visibility based on mask mode
            inp_mask_mode.change(
                toggle_inpaint_inputs,
                [inp_mask_mode],
                [inp_editor, inp_image, inp_mask_preview, inp_edit_mask_btn]
            )

            # Auto-generate mask preview when image uploaded in auto mode
            inp_image.change(preview_auto_mask, [inp_image, inp_mask_mode], [inp_mask_preview])
            inp_mask_mode.change(preview_auto_mask, [inp_image, inp_mask_mode], [inp_mask_preview])

            # Edit mask manually: store auto-mask in State, load overlay into editor
            inp_edit_mask_btn.click(
                edit_mask_manually,
                [inp_image, inp_mask_mode],
                [inp_editor, inp_mask_mode, inp_editor, inp_image, inp_mask_preview, inp_edit_mask_btn, inp_auto_mask_state]
            )

            # Track which gallery image is selected
            def on_gallery_select(evt: gr.SelectData):
                return evt.index

            inp_gallery.select(on_gallery_select, outputs=[inp_selected_idx])

            # Send selected result to paint editor for manual refinement
            def send_to_editor(gallery_images, selected_idx):
                if not gallery_images or len(gallery_images) == 0:
                    raise gr.Error("No results to send! Run inpaint first.")
                idx = int(selected_idx) if selected_idx is not None else 0
                idx = min(idx, len(gallery_images) - 1)
                img_data = gallery_images[idx]
                # Gallery items can be tuples (image, caption) or just images
                if isinstance(img_data, tuple):
                    img_data = img_data[0]
                if isinstance(img_data, str):
                    img_data = Image.open(img_data)
                # Return: editor with image as background, switch to manual mode, hide auto inputs
                editor_value = {"background": img_data, "layers": [], "composite": img_data}
                return (
                    editor_value,                    # inp_editor
                    "🖌️ Manual Paint",               # inp_mask_mode
                    gr.update(visible=True),          # inp_editor visible
                    gr.update(visible=False),         # inp_image hidden
                    gr.update(visible=False),         # inp_mask_preview hidden
                )

            inp_send_btn.click(
                send_to_editor,
                [inp_gallery, inp_selected_idx],
                [inp_editor, inp_mask_mode, inp_editor, inp_image, inp_mask_preview]
            )

            # Inpaint button
            inp_btn.click(lambda: ([], None, ""), outputs=[inp_gallery, inp_dl, inp_seed_out]).then(
                inpaint,
                [inp_editor, inp_image, inp_prompt, inp_neg, inp_seed, inp_cfg, inp_denoise, inp_num, inp_mask_mode, inp_auto_mask_state],
                [inp_gallery, inp_dl, inp_seed_out])

    gr.Markdown("""
    ---
    <p style="text-align:center; color:#6b7280; font-size:0.85em;">
    ⚡ Z-Fooocus ·
    <a class="footer-link" href="https://github.com/Tongyi-MAI/Z-Image">Z-Image</a> ·
    <a class="footer-link" href="https://github.com/lllyasviel/Fooocus">Fooocus</a> ·
    <a class="footer-link" href="https://github.com/NeuralFalconYT/Z-Image-Colab">NeuralFalconYT</a>
    </p>
    """)

demo.launch(share=True, debug=True)
