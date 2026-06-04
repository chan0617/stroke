"""
stroke_app.py — Sticker Edge Cleanup & 4K Upscaler
----------------------------------------------------
- ALL processing is LOCAL (OpenCV, Pillow, NumPy)
- NO external API calls, NO fees, NO cloud services
- Gradio analytics disabled
"""

import os, sys, subprocess

# Block all Gradio analytics before import — no data leaves the machine
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"]  = "False"

# ── Auto-install missing packages ─────────────────────────────────────────────
_DEPS = {"gradio": "gradio>=4.0", "PIL": "pillow", "cv2": "opencv-python", "numpy": "numpy"}
for _mod, _pkg in _DEPS.items():
    try:
        __import__(_mod)
    except ImportError:
        print(f"Installing {_pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "-q"])

import time, traceback
from pathlib import Path

import gradio as gr
import numpy as np
import cv2
from PIL import Image

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Remove green / colour fringe
# ══════════════════════════════════════════════════════════════════════════════

def remove_fringe(rgba: np.ndarray) -> np.ndarray:
    """
    Despill #00FF7F (spring green) and general colour contamination
    from semi-transparent edge pixels.
    """
    img = rgba.astype(np.float32)
    r, g, b, a = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]

    max_rb       = np.maximum(r, b)
    green_excess = np.clip(g - max_rb, 0, 255)
    edge         = (a > 1)   & (a < 240)
    interior     = (a >= 240)
    despill      = np.clip(green_excess / 128.0, 0, 1)

    new_g = g.copy()
    new_g[edge]     = np.clip(g[edge]     - green_excess[edge]     * despill[edge],   0, 255)
    new_g[interior] = np.clip(g[interior] - green_excess[interior] * 0.4,             0, 255)

    very_green = (g > 180) & (g > r * 1.8) & (g > b * 1.4)
    kill  = np.clip(green_excess / 200.0, 0, 1)
    new_a = a.copy()
    new_a[edge & very_green] = np.clip(
        a[edge & very_green] * (1.0 - kill[edge & very_green] * 0.85), 0, 255
    )

    result = img.copy()
    result[:,:,1] = new_g
    result[:,:,3] = new_a
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Smooth jagged alpha edges
# ══════════════════════════════════════════════════════════════════════════════

def smooth_alpha(alpha: np.ndarray) -> np.ndarray:
    """Morphological cleanup + Gaussian anti-aliasing on the alpha channel."""
    af     = alpha.astype(np.float32) / 255.0
    binary = (alpha > 128).astype(np.uint8) * 255
    k3     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3)
    opened = cv2.morphologyEx(closed,  cv2.MORPH_OPEN,  k3)
    smooth = cv2.GaussianBlur(opened.astype(np.float32) / 255.0, (5, 5), 1.0)
    edge   = (af > 0.04) & (af < 0.96)
    result = af.copy()
    result[edge]      = smooth[edge] * 0.65 + af[edge] * 0.35
    result[af >= 0.96] = 1.0
    return np.clip(result * 255, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3+4 — Per-character outline (white 2 px + black 1.5 px)
# ══════════════════════════════════════════════════════════════════════════════

def _alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Porter-Duff 'over' composite (both RGBA uint8)."""
    b, t  = bottom.astype(np.float32) / 255, top.astype(np.float32) / 255
    ta, ba = t[:,:,3:4], b[:,:,3:4]
    oa    = ta + ba * (1 - ta)
    safe  = np.where(oa > 0, oa, 1.0)
    orgb  = (t[:,:,:3]*ta + b[:,:,:3]*ba*(1-ta)) / safe
    out   = np.empty_like(bottom, dtype=np.float32)
    out[:,:,:3] = orgb
    out[:,:,3]  = oa[:,:,0]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def add_outline_per_character(
    img: np.ndarray,
    white_px: float = 2.0,
    black_px: float = 1.5,
    min_area: int   = 50,
) -> np.ndarray:
    """
    KEY FIX: uses cv2.connectedComponents to find every separate character
    in the image, then applies a precise white+black outline to EACH ONE
    individually via cv2.distanceTransform.

    Outline stack (outside → inside):
        black (1.5 px) → white (2 px) → character
    """
    h, w  = img.shape[:2]
    alpha = img[:,:,3]
    binary = (alpha > 30).astype(np.uint8)

    # Find separate character blobs
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    total_px = white_px + black_px   # 3.5 px — outer edge of black outline
    result   = np.zeros((h, w, 4), dtype=np.uint8)

    for label_id in range(1, num_labels):          # 0 = background
        comp = (labels == label_id).astype(np.uint8) * 255

        # Skip isolated noise pixels
        if int(comp.sum()) // 255 < min_area:
            continue

        # Precise distance (in pixels) from this component's boundary outward
        dist = cv2.distanceTransform(
            cv2.bitwise_not(comp), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        layer = np.zeros((h, w, 4), dtype=np.uint8)

        # --- Black outline ring ---
        black_zone = (dist > 0) & (dist <= total_px)
        layer[black_zone] = [0, 0, 0, 255]

        # --- White outline ring (overwrites inner part of black) ---
        white_zone = (dist > 0) & (dist <= white_px)
        layer[white_zone] = [255, 255, 255, 255]

        # --- Anti-alias the outermost black edge ---
        aa_zone = (dist > total_px - 0.5) & (dist <= total_px + 1.0)
        fade    = 1.0 - np.clip((dist - (total_px - 0.5)) / 1.0, 0, 1)
        layer[:,:,3][aa_zone] = np.clip(fade[aa_zone] * 255, 0, 255).astype(np.uint8)

        # --- Original character pixels on top ---
        char_px = comp > 0
        layer[char_px] = img[char_px]

        # Merge this character's layer into the overall result
        result = _alpha_over(result, layer)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — 4K Upscale + sharpen
# ══════════════════════════════════════════════════════════════════════════════

def upscale_4k(img: np.ndarray) -> np.ndarray:
    h, w   = img.shape[:2]
    MAX_DIM = 3840

    if max(h, w) < MAX_DIM:
        scale = MAX_DIM / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        up = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    else:
        up = img.copy()

    rgb   = up[:,:,:3].copy()
    alpha = up[:,:,3].copy()

    # Unsharp mask sharpening
    blur      = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(rgb.astype(np.float32), 1.6, blur, -0.6, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Bilateral filter: smooth noise, preserve edges
    denoised  = cv2.bilateralFilter(sharpened, d=5, sigmaColor=45, sigmaSpace=45)
    rgb_final = cv2.addWeighted(sharpened, 0.55, denoised, 0.45, 0)

    result = np.zeros_like(up)
    result[:,:,:3] = rgb_final
    result[:,:,3]  = alpha
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_image(input_path, progress=gr.Progress(track_tqdm=False)):
    if input_path is None:
        return None, None, None, None, "No image uploaded.", ""

    try:
        t0 = time.time()

        progress(0.05, desc="Loading image …")
        img = np.array(Image.open(input_path).convert("RGBA"))

        progress(0.20, desc="Removing colour fringe …")
        img = remove_fringe(img)

        progress(0.35, desc="Smoothing alpha edges …")
        img[:,:,3] = smooth_alpha(img[:,:,3])

        progress(0.50, desc="Adding per-character outline …")
        cleaned = add_outline_per_character(img, white_px=2.0, black_px=1.5)

        progress(0.70, desc="Upscaling to 4K …")
        upscaled = upscale_4k(cleaned)

        progress(0.90, desc="Saving outputs …")
        ts = int(time.time())
        c_path = str(OUTPUT_DIR / f"cleaned_{ts}.png")
        u_path = str(OUTPUT_DIR / f"4k_{ts}.png")

        Image.fromarray(cleaned,  "RGBA").save(c_path, "PNG", compress_level=1)
        Image.fromarray(upscaled, "RGBA").save(u_path, "PNG", compress_level=1)

        c_kb    = os.path.getsize(c_path)  / 1024
        u_kb    = os.path.getsize(u_path)  / 1024
        elapsed = time.time() - t0

        c_info = f"Resolution: {cleaned.shape[1]}×{cleaned.shape[0]} px  |  {c_kb:.0f} KB"
        u_info = (f"Resolution: {upscaled.shape[1]}×{upscaled.shape[0]} px  |  "
                  f"{u_kb:.0f} KB  |  {elapsed:.1f} s")

        progress(1.0, desc="Done!")
        return cleaned, upscaled, c_path, u_path, c_info, u_info

    except Exception as e:
        msg = f"Error: {e}\n\n{traceback.format_exc()}"
        print(msg)
        return None, None, None, None, msg, ""


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Stroke") as demo:

    gr.Markdown("# Stroke — Sticker Edge Cleanup & 4K Upscaler")
    gr.Markdown(
        "> **Local processing only** — OpenCV + Pillow + NumPy.  "
        "No external API. No cloud. No fees."
    )

    upload  = gr.Image(
        label="Upload PNG (transparent background)",
        type="filepath", image_mode="RGBA", height=300,
    )
    run_btn = gr.Button("⚡  Process", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Cleaned + Outline")
            cleaned_out  = gr.Image(label="Cleaned PNG", type="numpy",
                                    image_mode="RGBA", height=380)
            cleaned_info = gr.Textbox(label="Info", interactive=False, lines=1)
            dl_cleaned   = gr.File(label="⬇  Download Cleaned PNG")

        with gr.Column():
            gr.Markdown("### 4K Upscaled")
            upscaled_out  = gr.Image(label="4K PNG", type="numpy",
                                     image_mode="RGBA", height=380)
            upscaled_info = gr.Textbox(label="Info", interactive=False, lines=1)
            dl_4k         = gr.File(label="⬇  Download 4K PNG")

    run_btn.click(
        fn=process_image,
        inputs=[upload],
        outputs=[cleaned_out, upscaled_out, dl_cleaned, dl_4k,
                 cleaned_info, upscaled_info],
        show_progress="full",
    )

    gr.Markdown("---\nOutputs are also saved to `./output/` automatically.")


if __name__ == "__main__":
    demo.launch(inbrowser=True)
