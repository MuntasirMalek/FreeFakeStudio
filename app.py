# ============================================================
#  FreeFakeStudio — Multi-Model AI Image Studio
#  Models: Z-Image Turbo · FLUX.2-klein · Qwen-Image-Edit
#  Built for Google Colab T4 (15GB VRAM)
# ============================================================

import os, random, time, sys, gc
import torch
import numpy as np
from PIL import Image
import re, uuid
import gradio as gr
import cv2

# ── Engine imports (lazy) ──────────────────────────────────
import engine_z_image
import engine_flux_klein_9b
import engine_flux_klein_4b
import engine_qwen_edit_2511

# ── Model Manager ─────────────────────────────────────────
_current_model = None

MODEL_GEN = ["⚡ Z-Image Turbo", "🔮 FLUX.2-klein 9B", "🌊 FLUX.2-klein 4B"]
MODEL_EDIT = ["🔮 FLUX.2-klein 9B", "🌊 FLUX.2-klein 4B", "🎨 Qwen-Image-Edit"]

_ENGINE_MAP = {
    "⚡ Z-Image Turbo": engine_z_image,
    "🔮 FLUX.2-klein 9B":  engine_flux_klein_9b,
    "🌊 FLUX.2-klein 4B":  engine_flux_klein_4b,
    "🎨 Qwen-Image-Edit": engine_qwen_edit_2511,
}

def _ensure_model(model_name):
    global _current_model
    engine = _ENGINE_MAP[model_name]
    if engine.is_loaded():
        return engine

    # Unload current model first
    if _current_model and _current_model != model_name:
        old_engine = _ENGINE_MAP[_current_model]
        old_engine.unload()

    engine.load()
    _current_model = model_name
    return engine

# ── Helpers ────────────────────────────────────────────────
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

# ── Auto-mask helpers ──────────────────────────────────────
_rembg_session = None

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session

def auto_mask_background(image_pil):
    from rembg import remove
    session = get_rembg_session()
    result = remove(image_pil, session=session, only_mask=True)
    mask_np = np.array(result)
    return 255 - mask_np

def auto_mask_except_face(image_pil):
    img_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    h, w = img_np.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            pad_w, pad_h = int(fw * 0.3), int(fh * 0.3)
            x1, y1 = max(0, fx - pad_w), max(0, fy - pad_h)
            x2, y2 = min(w, fx + fw + pad_w), min(h, fy + fh + pad_h)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = (x2 - x1) // 2, (y2 - y1) // 2
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 0, -1)
    else:
        mask = auto_mask_background(image_pil)
    return mask

def generate_auto_mask(image_pil, mask_mode):
    if image_pil is None:
        return None
    if mask_mode == "🏞️ Background Only":
        mask_np = auto_mask_background(image_pil)
    elif mask_mode == "🎭 Everything Except Face":
        mask_np = auto_mask_except_face(image_pil)
    else:
        return None
    img_np = np.array(image_pil.convert("RGB")).copy()
    img_np[mask_np > 127] = [255, 255, 255]
    return Image.fromarray(img_np)


# =====================================================================
#  Tab Functions
# =====================================================================

# ── GENERATE ───────────────────────────────────────────────
def generate_image(model_name, prompt, negative, aspect_ratio,
                   seed, cfg, denoise, num_images, steps):
    seed = make_seed(seed)
    w, h = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]
    engine = _ensure_model(model_name)

    paths = []
    for i in range(int(num_images)):
        img = engine.generate(prompt, negative, w, h,
                              seed + i, cfg, denoise, int(steps))
        path = get_save_path("gen")
        img.save(path)
        paths.append(path)
    return paths, paths, str(seed)

# ── IMG2IMG ────────────────────────────────────────────────
def do_img2img(model_name, input_image, prompt, negative,
               seed, cfg, denoise, num_images, steps):
    if input_image is None:
        raise gr.Error("Upload an image first!")
    seed = make_seed(seed)
    engine = _ensure_model(model_name)

    paths = []
    for i in range(int(num_images)):
        img = engine.img2img(input_image, prompt, negative,
                             seed + i, cfg, denoise, int(steps))
        path = get_save_path("i2i")
        img.save(path)
        paths.append(path)
    return paths, paths, str(seed)

# ── INPAINT ────────────────────────────────────────────────
def do_inpaint(model_name, editor_data, inp_image, prompt, negative,
               seed, cfg, denoise, num_images, mask_mode,
               auto_mask_data, steps):
    seed = make_seed(seed)
    engine = _ensure_model(model_name)

    # Determine image and mask
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
            if auto_mask_data and auto_mask_data.get("original") is not None:
                orig_data = auto_mask_data["original"]
                if isinstance(orig_data, str):
                    original = Image.open(orig_data).convert("RGB")
                elif isinstance(orig_data, np.ndarray):
                    original = Image.fromarray(orig_data).convert("RGB")
                elif isinstance(orig_data, Image.Image):
                    original = orig_data.convert("RGB")
                else:
                    original = bg.convert("RGB")
                stored_mask = auto_mask_data.get("mask")
                if stored_mask is not None:
                    stored_mask = np.array(stored_mask, dtype=np.uint8)
                    if stored_mask.shape[:2] != manual_mask.shape[:2]:
                        stored_mask = np.array(Image.fromarray(stored_mask).resize(
                            (manual_mask.shape[1], manual_mask.shape[0]), Image.NEAREST))
                    mask_combined = np.maximum(stored_mask, manual_mask)
                else:
                    mask_combined = manual_mask
            else:
                original = bg.convert("RGB")
                mask_combined = manual_mask
        elif auto_mask_data and auto_mask_data.get("mask") is not None:
            orig_data = auto_mask_data["original"]
            if isinstance(orig_data, str):
                original = Image.open(orig_data).convert("RGB")
            elif isinstance(orig_data, np.ndarray):
                original = Image.fromarray(orig_data).convert("RGB")
            elif isinstance(orig_data, Image.Image):
                original = orig_data.convert("RGB")
            else:
                raise gr.Error("Could not load original image.")
            mask_data = auto_mask_data["mask"]
            mask_combined = np.array(mask_data, dtype=np.uint8)
            if mask_combined.shape[:2] != (original.size[1], original.size[0]):
                mask_combined = np.array(Image.fromarray(mask_combined).resize(original.size, Image.NEAREST))
        else:
            raise gr.Error("Paint the areas you want to change, or use auto-mask first!")
    else:
        if inp_image is None:
            raise gr.Error("Upload an image first!")
        if not isinstance(inp_image, Image.Image):
            inp_image = Image.fromarray(inp_image)
        original = inp_image.convert("RGB")
        if mask_mode == "🏞️ Background Only":
            mask_combined = auto_mask_background(original)
        elif mask_mode == "🎭 Everything Except Face":
            mask_combined = auto_mask_except_face(original)
        else:
            raise gr.Error("Unknown mask mode.")

    paths = []
    for i in range(int(num_images)):
        img = engine.inpaint(original, mask_combined, prompt, negative,
                             seed + i, cfg, denoise, int(steps))
        path = get_save_path("inpaint")
        img.save(path)
        paths.append(path)
    return paths, paths, str(seed)


# =====================================================================
#  UI
# =====================================================================
ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
]
DEFAULT_NEG = "low quality, blurry, pixelated, noise, watermark, text, logo"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.gradio-container {
    font-family: 'Inter', -apple-system, sans-serif !important;
    max-width: 1400px !important;
}
.main-title { text-align:center; font-size:2.8em; font-weight:800; margin:0 0 4px 0; }
.main-title span {
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { text-align:center; color:#9ca3af; margin-bottom:15px; font-size:1.05em; }
.footer-link { color:#60a5fa !important; text-decoration:none; }

/* Prevent scrollbars from interrupting brush strokes in ImageEditor */
.image-editor-container, .image-editor-container .wrapper, .svelte-1p1fptw {
    overflow: hidden !important;
}

/* Hide the hand/pan tool button from ImageEditor toolbar */
button[aria-label="Pan"], button[aria-label="Move"] {
    display: none !important;
}

/* Gallery shift+click hint */
.gallery-item { cursor: pointer; }
"""

# JavaScript: override brush max size and fix preview
JS_CUSTOM = """
<script>
// Fix "Open in New Tab" — override navigator.share so Gradio's gallery
// share button opens the image in a new tab instead of downloading it.
(function() {
    var _origCanShare = navigator.canShare ? navigator.canShare.bind(navigator) : null;
    var _origShare = navigator.share ? navigator.share.bind(navigator) : null;

    navigator.canShare = function(data) {
        if (data && data.files && data.files.length > 0 &&
            data.files[0].type && data.files[0].type.startsWith('image/')) {
            return true;
        }
        return _origCanShare ? _origCanShare(data) : false;
    };

    navigator.share = async function(data) {
        if (data && data.files && data.files.length > 0 &&
            data.files[0].type && data.files[0].type.startsWith('image/')) {
            var url = URL.createObjectURL(data.files[0]);
            window.open(url, '_blank');
            return;
        }
        if (_origShare) return _origShare(data);
    };
})();

// Override brush/eraser size slider max from default ~100 to 300
function boostBrushMax() {
    document.querySelectorAll('input[type="range"]').forEach(function(slider) {
        if (parseFloat(slider.max) > 10 && parseFloat(slider.max) <= 110) {
            var parent = slider.closest('.image-editor, .image_editor');
            if (!parent) {
                var labels = slider.closest('.block, .wrap, div');
                if (labels && labels.querySelector('canvas')) parent = labels;
            }
            if (parent || slider.closest('[data-testid]')) {
                slider.max = 300;
                slider.setAttribute('max', '300');
            }
        }
    });
}
// Run periodically to catch dynamically created editors
setInterval(boostBrushMax, 2000);
setTimeout(boostBrushMax, 1000);
</script>
"""

zfooocus_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

# ── Inpaint UI helpers ─────────────────────────────────────
def toggle_inpaint_inputs(mode):
    if mode == "🖌️ Manual Paint":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def edit_mask_manually(image, mask_mode):
    if image is None:
        raise gr.Error("Upload an image first!")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    original = image.convert("RGB")
    w, h = original.size
    if mask_mode == "🏞️ Background Only":
        mask_np = auto_mask_background(original)
    elif mask_mode == "🎭 Everything Except Face":
        mask_np = auto_mask_except_face(original)
    else:
        raise gr.Error("Select an auto mask mode first.")
    auto_mask_data = {"mask": mask_np, "original": original}
    # Put the clean original as the background and the mask as an erasable RGBA layer
    mask_layer = np.zeros((h, w, 4), dtype=np.uint8)
    mask_layer[:, :, 0] = 255  # white R
    mask_layer[:, :, 1] = 255  # white G
    mask_layer[:, :, 2] = 255  # white B
    mask_layer[:, :, 3] = mask_np  # alpha = mask (visible where masked)
    mask_layer_pil = Image.fromarray(mask_layer, "RGBA")
    editor_value = {"background": original, "layers": [mask_layer_pil], "composite": original}
    return (editor_value, "🖌️ Manual Paint",
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False), auto_mask_data)

def preview_auto_mask(image, mask_mode):
    if image is None or mask_mode == "🖌️ Manual Paint":
        return None
    return generate_auto_mask(image, mask_mode)




# ── Build UI ───────────────────────────────────────────────
with gr.Blocks(theme=zfooocus_theme, css=CSS, title="FreeFakeStudio") as demo:

    gr.HTML(JS_CUSTOM)

    with gr.Tabs():

        # ═══════════════════════════════════════════════════
        # GENERATE
        # ═══════════════════════════════════════════════════
        with gr.Tab("🖼️ Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gen_model = gr.Dropdown(MODEL_GEN, value="⚡ Z-Image Turbo",
                                            label="Model")
                    gen_prompt = gr.Textbox(
                        label="Prompt", lines=4,
                        placeholder="Describe the image you want...",
                        value="A cinematic portrait of an astronaut riding a white horse across a golden wheat field at sunset, 8K, ultra detailed"
                    )
                    with gr.Row():
                        gen_aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
                        gen_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                    gen_num = gr.Slider(1, 16, value=2, step=1, label="Number of Images")
                    gen_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                    gen_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        gen_cfg = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="CFG")
                        gen_denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                        gen_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                with gr.Column(scale=1):
                    gr.HTML('<h1 class="main-title">🎭 <span>FreeFakeStudio</span></h1>')
                    gen_gallery = gr.Gallery(label="Results", columns=2, height=520,
                                             object_fit="contain", show_download_button=True,
                                             show_fullscreen_button=True, preview=True)
                    gen_dl = gr.File(label="Download All", file_count="multiple")
                    gen_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)


            gen_btn.click(
                lambda: ([], None, ""),
                outputs=[gen_gallery, gen_dl, gen_seed_out]).then(
                generate_image,
                [gen_model, gen_prompt, gen_neg, gen_aspect, gen_seed, gen_cfg, gen_denoise, gen_num, gen_steps],
                [gen_gallery, gen_dl, gen_seed_out])

        # ═══════════════════════════════════════════════════
        # IMG2IMG
        # ═══════════════════════════════════════════════════
        with gr.Tab("✏️ Img2Img"):
            with gr.Row():
                with gr.Column(scale=1):
                    i2i_model = gr.Dropdown(MODEL_EDIT, value="🌊 FLUX.2-klein 4B",
                                             label="Model")
                    i2i_img = gr.Image(type="pil", label="Upload Photo", sources=["upload"])
                    i2i_prompt = gr.Textbox(label="Prompt / Edit Instruction", lines=2,
                        placeholder="e.g., change the dress to a red saree")
                    i2i_num = gr.Slider(1, 16, value=2, step=1, label="Number of Images")
                    i2i_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    i2i_btn = gr.Button("✨ Transform", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        i2i_cfg = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="CFG")
                        i2i_denoise = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="Denoise")
                        i2i_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                        i2i_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)

                with gr.Column(scale=1):
                    gr.HTML('<h1 class="main-title">🎭 <span>FreeFakeStudio</span></h1>')
                    i2i_gallery = gr.Gallery(label="Results", columns=2, height=520,
                                              object_fit="contain", show_download_button=True,
                                              show_fullscreen_button=True, preview=True)
                    i2i_dl = gr.File(label="Download All", file_count="multiple")
                    i2i_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)


            i2i_btn.click(
                lambda: ([], None, ""),
                outputs=[i2i_gallery, i2i_dl, i2i_seed_out]).then(
                do_img2img,
                [i2i_model, i2i_img, i2i_prompt, i2i_neg, i2i_seed, i2i_cfg, i2i_denoise, i2i_num, i2i_steps],
                [i2i_gallery, i2i_dl, i2i_seed_out])

        # ═══════════════════════════════════════════════════
        # INPAINT
        # ═══════════════════════════════════════════════════
        with gr.Tab("🎨 Inpaint"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp_model = gr.Dropdown(MODEL_EDIT, value="🌊 FLUX.2-klein 4B",
                                             label="Model")
                    inp_mask_mode = gr.Radio(
                        choices=["🖌️ Manual Paint", "🏞️ Background Only", "🎭 Everything Except Face"],
                        value="🖌️ Manual Paint", label="Mask Mode",
                    )
                    inp_editor = gr.ImageEditor(
                        label="Upload & Paint Mask", type="pil",
                        canvas_size=(2048, 2048),
                        brush=gr.Brush(colors=["#ffffff"], default_size=40, default_color="#ffffff"),
                        eraser=gr.Eraser(default_size=40),
                        sources=["upload"], transforms=[],
                        layers=False, visible=True,
                    )
                    inp_image = gr.Image(type="pil", label="Upload Photo",
                                         visible=False, sources=["upload"])
                    inp_mask_preview = gr.Image(label="Mask Preview (white = will be changed)",
                                                 height=300, visible=False, interactive=False)
                    inp_edit_mask_btn = gr.Button("✏️ Edit This Mask Manually",
                                                   variant="secondary", visible=False)
                    inp_prompt = gr.Textbox(label="What should the masked area become?", lines=2,
                        placeholder="e.g., a tropical beach background")
                    inp_num = gr.Slider(1, 16, value=2, step=1, label="Number of Images")
                    inp_steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                    inp_btn = gr.Button("🎨 Inpaint", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Advanced", open=False):
                        inp_cfg = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="CFG")
                        inp_denoise = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Denoise")
                        inp_seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                        inp_neg = gr.Textbox(DEFAULT_NEG, label="Negative Prompt", lines=2)
                    inp_clear = gr.ClearButton([inp_editor, inp_image, inp_mask_preview], value="🗑️ Clear All")
                with gr.Column(scale=1):
                    gr.HTML('<h1 class="main-title">🎭 <span>FreeFakeStudio</span></h1>')
                    inp_gallery = gr.Gallery(label="Results", columns=2, height=520,
                                              object_fit="contain", show_download_button=True,
                                              show_fullscreen_button=True, preview=True)
                    inp_send_btn = gr.Button("🖌️ Send to Paint Editor for Touch-up", variant="secondary")
                    inp_dl = gr.File(label="Download All", file_count="multiple")
                    inp_seed_out = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)
                    inp_selected_idx = gr.State(value=0)
                    inp_auto_mask_state = gr.State(value=None)

            # Events

            inp_mask_mode.change(toggle_inpaint_inputs, [inp_mask_mode],
                                 [inp_editor, inp_image, inp_mask_preview, inp_edit_mask_btn])
            inp_image.change(preview_auto_mask, [inp_image, inp_mask_mode], [inp_mask_preview])
            inp_mask_mode.change(preview_auto_mask, [inp_image, inp_mask_mode], [inp_mask_preview])
            inp_edit_mask_btn.click(edit_mask_manually, [inp_image, inp_mask_mode],
                [inp_editor, inp_mask_mode, inp_editor, inp_image, inp_mask_preview, inp_edit_mask_btn, inp_auto_mask_state])

            def on_gallery_select(evt: gr.SelectData):
                return evt.index
            inp_gallery.select(on_gallery_select, outputs=[inp_selected_idx])

            def send_to_editor(gallery_images, selected_idx):
                if not gallery_images:
                    raise gr.Error("No results to send!")
                idx = min(int(selected_idx or 0), len(gallery_images) - 1)
                img_data = gallery_images[idx]
                if isinstance(img_data, tuple):
                    img_data = img_data[0]
                if isinstance(img_data, str):
                    img_data = Image.open(img_data)
                editor_value = {"background": img_data, "layers": [], "composite": img_data}
                return (editor_value, "🖌️ Manual Paint",
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
            inp_send_btn.click(send_to_editor, [inp_gallery, inp_selected_idx],
                [inp_editor, inp_mask_mode, inp_editor, inp_image, inp_mask_preview])

            def do_inpaint_wrapper(model_name, editor_data, inp_image, prompt, negative,
                                   seed, cfg, denoise, num_images, mask_mode,
                                   auto_mask_data, steps):
                return do_inpaint(model_name, editor_data, inp_image, prompt, negative,
                                 seed, cfg, denoise, num_images, mask_mode,
                                 auto_mask_data, steps)

            inp_btn.click(
                lambda: ([], None, ""),
                outputs=[inp_gallery, inp_dl, inp_seed_out]).then(
                do_inpaint_wrapper,
                [inp_model, inp_editor, inp_image, inp_prompt, inp_neg, inp_seed,
                 inp_cfg, inp_denoise, inp_num, inp_mask_mode, inp_auto_mask_state, inp_steps],
                [inp_gallery, inp_dl, inp_seed_out])

            def optimize_cfg_for_model(model_name):
                # FLUX-type models (Qwen, FLUX) work best with CFG=1.0 in ComfyUI
                # (guidance is embedded in the model, not applied via CFG)
                return gr.update(value=1.0)
                
            i2i_model.change(optimize_cfg_for_model, inputs=[i2i_model], outputs=[i2i_cfg])
            inp_model.change(optimize_cfg_for_model, inputs=[inp_model], outputs=[inp_cfg])

    gr.Markdown("""
    ---
    <p style="text-align:center; color:#6b7280; font-size:0.85em;">
    🎭 FreeFakeStudio — Open-source Multi-Model AI Image Studio
    </p>
    """)

# ── Load first available model on startup ──────────────────
import glob as _gl
_model_files = {
    "⚡ Z-Image Turbo":   "z-image-turbo-fp8-e4m3fn.safetensors",
    "🌊 FLUX.2-klein 4B": "flux-2-klein-4b.safetensors",
    "🔮 FLUX.2-klein 9B": "flux-2-klein-9b-kv-fp8.safetensors",
}
_diff_dir = "/content/ComfyUI/models/diffusion_models"
for _name, _file in _model_files.items():
    if os.path.exists(os.path.join(_diff_dir, _file)):
        try:
            _ENGINE_MAP[_name].load()
            _current_model = _name
            break
        except Exception as e:
            print(f"⚠️ Failed to load {_name}: {e}")
else:
    print("⚠️ No pre-loaded model — select one from the dropdown to load on first use")

demo.launch(share=True, debug=True)
