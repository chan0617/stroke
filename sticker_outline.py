"""
sticker_outline.py
------------------
Cleans up sticker PNGs (removes green #00FF7F fringe, smooths jagged edges)
and adds a white (2px) + black (1.5px) outline stack.

Usage:
    python sticker_outline.py [input_dir] [output_dir]
    Defaults: input/ → output/

Dependencies:
    pip install pillow opencv-python numpy
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# Step 1 — Remove green (#00FF7F) colour fringe
# ─────────────────────────────────────────────

def remove_green_fringe(rgba: np.ndarray) -> np.ndarray:
    """
    Despill spring-green (#00FF7F = R0 G255 B127) contamination from
    semi-transparent edge pixels.

    Strategy:
    - On edge pixels (0 < alpha < 240): reduce the green channel so it
      doesn't exceed max(R, B), then shrink alpha proportionally to
      how "purely green" the pixel is.
    - On fully-opaque interior pixels: mild despill only (rarely needed).
    """
    img = rgba.astype(np.float32)
    r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]

    max_rb = np.maximum(r, b)                    # what green "should" cap at
    green_excess = np.clip(g - max_rb, 0, 255)  # how much green over-shoots

    edge_mask     = (a > 1)  & (a < 240)
    interior_mask = (a >= 240)

    # --- despill green channel on edges ---
    despill_ratio = np.clip(green_excess / 128.0, 0.0, 1.0)
    new_g = g.copy()
    new_g[edge_mask] = np.clip(
        g[edge_mask] - green_excess[edge_mask] * despill_ratio[edge_mask],
        0, 255
    )
    # mild despill on opaque interior
    new_g[interior_mask] = np.clip(
        g[interior_mask] - green_excess[interior_mask] * 0.4,
        0, 255
    )

    # --- shrink alpha for near-pure-green edge pixels ---
    very_green = (g > 180) & (g > r * 1.8) & (g > b * 1.4)
    new_a = a.copy()
    green_alpha_kill = np.clip(green_excess / 200.0, 0.0, 1.0)
    new_a[edge_mask & very_green] = np.clip(
        a[edge_mask & very_green] * (1.0 - green_alpha_kill[edge_mask & very_green] * 0.85),
        0, 255
    )

    result = img.copy()
    result[:, :, 1] = new_g
    result[:, :, 3] = new_a
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Step 2 — Smooth / anti-alias alpha edges
# ─────────────────────────────────────────────

def smooth_alpha_edges(alpha: np.ndarray) -> np.ndarray:
    """
    1. Morphological close → fill tiny holes inside the character.
    2. Morphological open  → remove isolated noise specks.
    3. Gaussian blur on the cleaned mask for smooth anti-aliased edges.
    4. Blend smoothed result with original, keeping fully-opaque interior intact.
    """
    alpha_f = alpha.astype(np.float32) / 255.0
    binary  = (alpha > 128).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  k3)

    # Gaussian for anti-aliasing
    morph_f  = opened.astype(np.float32) / 255.0
    smooth_f = cv2.GaussianBlur(morph_f, (5, 5), 1.0)

    # Only modify pixels in the "edge zone" (not fully opaque or transparent)
    edge_zone     = (alpha_f > 0.04) & (alpha_f < 0.96)
    interior_zone = alpha_f >= 0.96

    result = alpha_f.copy()
    result[edge_zone]     = smooth_f[edge_zone] * 0.65 + alpha_f[edge_zone] * 0.35
    result[interior_zone] = 1.0   # don't touch solid interior

    return np.clip(result * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Steps 3 & 4 — Build outline layers
# ─────────────────────────────────────────────

def build_outline_layer(
    alpha: np.ndarray,
    dilate_px: float,
    color_rgb: tuple,
) -> np.ndarray:
    """
    Dilate the alpha mask by `dilate_px` pixels, fill with `color_rgb`,
    return an RGBA numpy array.

    The returned layer covers the *entire* dilated area (including where the
    character sits). When composited correctly (below the character), only the
    ring beyond the character edge is visible.
    """
    h, w = alpha.shape
    radius = max(1, int(np.ceil(dilate_px)))
    ksize  = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    binary  = (alpha > 30).astype(np.uint8) * 255
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Smooth the outline edge for a clean anti-aliased look
    dilated_f = dilated.astype(np.float32) / 255.0
    dilated_f = cv2.GaussianBlur(dilated_f, (3, 3), 0.7)
    dilated_f = np.clip(dilated_f, 0, 1)

    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[:, :, 0] = color_rgb[0]
    layer[:, :, 1] = color_rgb[1]
    layer[:, :, 2] = color_rgb[2]
    layer[:, :, 3] = (dilated_f * 255).astype(np.uint8)
    return layer


# ─────────────────────────────────────────────
# Step 5 — Alpha composite (Porter-Duff "over")
# ─────────────────────────────────────────────

def alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Composite `top` over `bottom`, both RGBA uint8 numpy arrays."""
    b = bottom.astype(np.float32) / 255.0
    t = top.astype(np.float32)    / 255.0

    ta = t[:, :, 3:4]
    ba = b[:, :, 3:4]

    out_a   = ta + ba * (1.0 - ta)
    safe_a  = np.where(out_a > 0, out_a, 1.0)  # avoid div/0
    out_rgb = (t[:, :, :3] * ta + b[:, :, :3] * ba * (1.0 - ta)) / safe_a

    out = np.empty_like(bottom, dtype=np.float32)
    out[:, :, :3] = out_rgb
    out[:, :, 3]  = out_a[:, :, 0]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Main processing pipeline
# ─────────────────────────────────────────────

def process_sticker(input_path: str, output_path: str) -> bool:
    name = os.path.basename(input_path)
    print(f"  [{name}] loading …")

    try:
        pil = Image.open(input_path).convert("RGBA")
    except Exception as exc:
        print(f"  [{name}] ERROR — could not open file: {exc}")
        return False

    img = np.array(pil)
    print(f"  [{name}] size {img.shape[1]}×{img.shape[0]}")

    # 1. Remove green fringing
    print(f"  [{name}] removing green fringe …")
    img = remove_green_fringe(img)

    # 2. Smooth jagged alpha edges
    print(f"  [{name}] smoothing alpha edges …")
    img[:, :, 3] = smooth_alpha_edges(img[:, :, 3])

    alpha = img[:, :, 3]

    # 3. White outline — dilate 2 px from character edge
    print(f"  [{name}] building white outline (2 px) …")
    white_layer = build_outline_layer(alpha, dilate_px=2.0, color_rgb=(255, 255, 255))

    # 4. Black outline — dilate 3.5 px from character edge (= 2 + 1.5 px outside white)
    print(f"  [{name}] building black outline (1.5 px outside white) …")
    black_layer = build_outline_layer(alpha, dilate_px=3.5, color_rgb=(0, 0, 0))

    # 5. Composite bottom → top: black / white / character
    print(f"  [{name}] compositing layers …")
    composite = alpha_over(black_layer, white_layer)
    composite = alpha_over(composite,  img)

    # 6. Save
    try:
        Image.fromarray(composite, "RGBA").save(output_path, "PNG", compress_level=1)
        print(f"  [{name}] saved → {output_path}")
        return True
    except Exception as exc:
        print(f"  [{name}] ERROR — could not save: {exc}")
        return False


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    input_dir  = sys.argv[1] if len(sys.argv) > 1 else "input"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    if not os.path.isdir(input_dir):
        print(f"Input directory not found: '{input_dir}'")
        print("Usage: python sticker_outline.py [input_dir] [output_dir]")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    png_files = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(".png")
    )

    if not png_files:
        print(f"No PNG files found in '{input_dir}'")
        sys.exit(0)

    print(f"Found {len(png_files)} PNG file(s)  |  {input_dir} → {output_dir}")
    print("=" * 60)

    ok, fail = 0, 0
    for fname in png_files:
        result = process_sticker(
            os.path.join(input_dir, fname),
            os.path.join(output_dir, fname),
        )
        print()
        if result:
            ok += 1
        else:
            fail += 1

    print("=" * 60)
    print(f"Done — {ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
