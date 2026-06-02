#!/usr/bin/env python3
"""
Render the Web World Wide hero globe as a seamless, looping GIF that mirrors the
website: a blue sky with drifting pixel-art clouds and the wireframe globe
spinning in front.

Faithful to the live site:
  * Sky gradient      -> site/src/styles/sky.css
  * Pixel-art clouds  -> site/src/scripts/clouds.ts (same SHAPES + colors)
  * Wireframe globe   -> site/src/scripts/globe-nav.ts (14 meridians, 9
                         parallels, fov 34, cam z 3.4, uniform black lines)

Perfect loop, no visible cut: the GIF spans one shared period of N frames in
which (a) the globe rotates exactly 2*pi/14 -- the wireframe is invariant under
that rotation, so frame N == frame 0 -- and (b) the clouds pan exactly one frame
width and wrap, so they also return to start. The last frame therefore flows
seamlessly into the first.

The globe is drawn at 4x supersample for clean anti-aliasing on the sky; the
clouds are drawn crisp (nearest-neighbor) like the site's `image-rendering:
pixelated`. The scene is framed as a rounded-rect card.

Also emits a small static transparent wireframe (globe.png) for the website
project-card icon.

Requires: Pillow, numpy.  Outputs profile/assets/globe.gif + globe.png.
"""
from __future__ import annotations
import math
import os
import random
import numpy as np
from PIL import Image, ImageDraw

# --- globe geometry (mirrors globe-nav.ts) -----------------------------------
MERIDIANS = 14
PARALLELS = 9
LINE_SEGS = 96
CAM_DIST = 3.4
FOV_DEG = 34.0
TILT_X_DEG = 12.0        # slight forward tilt so it reads as a globe

# --- scene / output ----------------------------------------------------------
OUT = 340                # final GIF size (px, square)
SS = 4                   # globe supersample factor
FRAMES = 36              # frames across one shared loop period
GLOBE_FILL = 0.82        # globe diameter as fraction of the card
LINE_W = 1.7 * SS        # globe line width at supersample scale
CORNER = 30              # rounded-card corner radius (px)
SEED = 7                 # deterministic cloud layout

# --- colors (brand / site) ---------------------------------------------------
SKY_TOP = (20, 80, 200)     # #1450c8  --sky-deep
SKY_MID = (30, 104, 240)    # #1e68f0  --sky
SKY_BOT = (46, 122, 242)    # #2e7af2
GLOBE_RGB = (8, 12, 28)     # near-black wireframe (matches site's black lines)
CLOUD_OPACITY = 0.72        # site .sky-puffs opacity

HALF_FOV = math.radians(FOV_DEG) / 2.0
TAN_HALF = math.tan(HALF_FOV)
TILT_X = math.radians(TILT_X_DEG)

# --- pixel-art cloud shapes (ported verbatim from clouds.ts) -----------------
SHAPES = [
    ["0002220002220000022000",
     "0022112202211220222110",
     "0211111211111111111112",
     "2111111111111111111112",
     "2111111111111111111112",
     "0222222222222222222220"],
    ["00022200002200",
     "00211122022112",
     "02111111111112",
     "02111111111112",
     "02222222222220"],
    ["00022000022000022000022000",
     "02211220221122022112202112",
     "21111111111111111111111111",
     "02222222222222222222222220"],
    ["0002220000",
     "0022112200",
     "0211111120",
     "2111111112",
     "2111111112",
     "0211111120",
     "0022222200"],
]
# outline colors, light -> deep blue (clouds.ts DEFAULT_LAYERS)
CLOUD_OUTLINES = [(212, 223, 240), (192, 208, 236), (168, 200, 240), (136, 180, 232)]


# ============================ sky ============================================
def make_sky(size: int) -> Image.Image:
    """Vertical gradient matching sky.css (#1450c8 -> #1e68f0@38% -> #2e7af2)."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        f = y / (size - 1)
        if f <= 0.38:
            t = f / 0.38
            c = [SKY_TOP[i] + (SKY_MID[i] - SKY_TOP[i]) * t for i in range(3)]
        else:
            t = (f - 0.38) / 0.62
            c = [SKY_MID[i] + (SKY_BOT[i] - SKY_MID[i]) * t for i in range(3)]
        arr[y, :] = c
    return Image.fromarray(arr, "RGB")


# ============================ clouds =========================================
def cloud_sprite(shape, scale, outline) -> Image.Image:
    """Render one pixel-art puff crisp (1px cells scaled nearest)."""
    h, w = len(shape), len(shape[0])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for y in range(h):
        for x in range(w):
            ch = shape[y][x]
            if ch == "1":
                px[x, y] = (255, 255, 255, 255)
            elif ch == "2":
                px[x, y] = (*outline, 255)
    return img.resize((w * scale, h * scale), Image.NEAREST)


def build_clouds(size: int):
    """Deterministic puff layout. Returns list of (sprite, x, y, alpha)."""
    rng = random.Random(SEED)
    layers = [   # (count, scale_min, scale_max, op_min, op_max, top_min, top_max)
        (3, 2, 3, 0.30, 0.50, 0.00, 0.30),
        (4, 3, 5, 0.50, 0.70, 0.02, 0.62),
        (4, 5, 7, 0.70, 0.85, 0.08, 0.86),
        (2, 8, 11, 0.85, 0.95, 0.16, 0.92),
    ]
    puffs = []
    for li, (count, smin, smax, omin, omax, tmin, tmax) in enumerate(layers):
        outline = CLOUD_OUTLINES[li]
        for _ in range(count):
            shape = rng.choice(SHAPES)
            scale = rng.randint(smin, smax)
            sprite = cloud_sprite(shape, scale, outline)
            x = rng.uniform(0, size)
            y = rng.uniform(tmin, tmax) * size
            alpha = rng.uniform(omin, omax)
            puffs.append((sprite, x, y, alpha))
    return puffs


def paste_clouds(base: Image.Image, puffs, pan: float, size: int) -> None:
    """Composite drifting puffs onto `base`, wrapping seamlessly across width."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for sprite, x0, y, alpha in puffs:
        sp = sprite
        if alpha < 0.999:  # pre-multiply layer alpha into the sprite's alpha
            a = sp.split()[3].point(lambda v: int(v * alpha))
            sp = sp.copy(); sp.putalpha(a)
        x = (x0 - pan) % size
        for dx in (x - size, x, x + size):  # wrap copies on both edges
            layer.alpha_composite(sp, (int(round(dx)), int(round(y))))
    # global cloud-layer opacity, like .sky-puffs { opacity: .72 }
    a = layer.split()[3].point(lambda v: int(v * CLOUD_OPACITY))
    layer.putalpha(a)
    base.alpha_composite(layer)


# ============================ globe ==========================================
def build_lines():
    lines = []
    for i in range(MERIDIANS):
        a = (i / MERIDIANS) * math.pi * 2
        j = np.arange(LINE_SEGS + 1)
        t = (j / LINE_SEGS) * math.pi
        lines.append(np.stack(
            [np.sin(t) * math.cos(a), np.cos(t), np.sin(t) * math.sin(a)], axis=1))
    for i in range(1, PARALLELS):
        phi = (i / PARALLELS) * math.pi
        y, r = math.cos(phi), math.sin(phi)
        j = np.arange(LINE_SEGS + 1)
        t = (j / LINE_SEGS) * math.pi * 2
        lines.append(np.stack(
            [r * np.cos(t), np.full_like(t, y), r * np.sin(t)], axis=1))
    return lines


def rot_y(p, th):
    c, s = math.cos(th), math.sin(th)
    return np.stack([p[:, 0] * c + p[:, 2] * s, p[:, 1],
                     -p[:, 0] * s + p[:, 2] * c], axis=1)


def tilt_x(p, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.stack([p[:, 0], p[:, 1] * c - p[:, 2] * s,
                     p[:, 1] * s + p[:, 2] * c], axis=1)


def project(p, size, fill):
    d = CAM_DIST - p[:, 2]
    nx = (p[:, 0] / d) / TAN_HALF
    ny = (p[:, 1] / d) / TAN_HALF
    px = size / 2 + nx * (size / 2) * fill
    py = size / 2 - ny * (size / 2) * fill
    return np.stack([px, py], axis=1)


def render_globe(lines, theta, size, color, line_w, fill):
    """Wireframe on a transparent layer at `size`, uniform color (site look)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = line_w / 2.0
    for pts in lines:
        xy = project(tilt_x(rot_y(pts, theta), TILT_X), size, fill)
        draw.line([tuple(pt) for pt in xy], fill=(*color, 255),
                  width=int(round(line_w)), joint="curve")
        # round caps so the polyline ends/joints stay smooth
        x0, y0 = xy[0]; x1, y1 = xy[-1]
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=(*color, 255))
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=(*color, 255))
    return img


# ============================ framing / export ===============================
def rounded_mask(size: int, radius: int) -> Image.Image:
    m = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        (0, 0, size * SS - 1, size * SS - 1), radius=radius * SS, fill=255)
    return m.resize((size, size), Image.LANCZOS)


def to_paletted(frame: Image.Image, mask: Image.Image) -> Image.Image:
    """RGB scene -> P mode; pixels outside the rounded card are transparent."""
    pal = frame.convert("RGB").quantize(
        colors=255, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    transp = mask.point(lambda v: 255 if v < 128 else 0).convert("1")
    pal.paste(255, transp)
    pal.info["transparency"] = 255
    return pal


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "profile", "assets")
    os.makedirs(out_dir, exist_ok=True)

    lines = build_lines()
    sky = make_sky(OUT)
    puffs = build_clouds(OUT)
    mask = rounded_mask(OUT, CORNER)
    period = (2 * math.pi) / MERIDIANS

    frames = []
    for f in range(FRAMES):
        scene = sky.copy().convert("RGBA")
        paste_clouds(scene, puffs, pan=f / FRAMES * OUT, size=OUT)
        globe = render_globe(lines, f / FRAMES * period, OUT * SS,
                             GLOBE_RGB, LINE_W, GLOBE_FILL).resize(
            (OUT, OUT), Image.LANCZOS)
        scene.alpha_composite(globe)
        frames.append(to_paletted(scene.convert("RGB"), mask))

    gif_path = os.path.join(out_dir, "globe.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, disposal=2, transparency=255,
                   optimize=True)

    # static transparent wireframe for the website project-card icon
    icon = render_globe(lines, 0.0, OUT * SS, (30, 104, 240), 2.2 * SS,
                        GLOBE_FILL).resize((OUT, OUT), Image.LANCZOS)
    png_path = os.path.join(out_dir, "globe.png")
    icon.save(png_path)

    print(f"wrote {gif_path} ({os.path.getsize(gif_path)//1024} KB, "
          f"{OUT}px, {FRAMES} frames)")
    print(f"wrote {png_path} ({os.path.getsize(png_path)//1024} KB)")


if __name__ == "__main__":
    main()
