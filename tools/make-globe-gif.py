#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "numpy"]
# ///
"""
Render the Web World Wide hero globe as a seamless, looping GIF that mirrors the
website: a blue sky with drifting pixel-art clouds and the pixel-art Earth
spinning in front, with the hero's ink outline and offset shadow.

Faithful to the live site:
  * Sky gradient      -> site/src/styles/sky.css
  * Pixel-art clouds  -> site/src/scripts/clouds.ts (same SHAPES + colors)
  * Pixel-art globe   -> site/src/lib/pixel-globe.ts (a straight port of
                         globeSampler: same Natural Earth map, orthographic
                         projection, 3x3 majority vote and small-island pass)
                         seen as in scripts/globe-nav.ts (64 px across, viewed
                         from 5 deg N, Atlantic facing at rest, turning west to
                         east) and styled like .wordmark-globe (4px ink border,
                         8px ink offset shadow).

The map itself is read from the site's generated site/src/lib/world-map.ts,
so the GIF always matches the website. Pass a path to that file (or to a
webworldwide-website checkout) as the first argument; with no argument the
script fetches it from the site repo's main branch.

Perfect loop, no visible cut: the GIF spans one full turn of the globe in
FRAMES frames while the clouds pan exactly one frame width and wrap, so the
last frame flows straight into the first.

Globe pixels are scaled by an exact integer (nearest neighbour, like the
site's `image-rendering: pixelated`); only the disc's rim is anti-aliased.
The scene is framed as a rounded-rect card.

Also emits a static transparent globe (globe.png) for the website
project-card icon.

Run:  uv run tools/make-globe-gif.py [path/to/world-map.ts | path/to/site-checkout]
Outputs profile/assets/globe.gif + globe.png.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

WORLD_MAP_URL = ("https://raw.githubusercontent.com/WebWorldWide/webworldwide-website/"
                 "main/site/src/lib/world-map.ts")

# --- globe (mirrors globe-nav.ts / pixel-globe.ts) ---------------------------
GLOBE_PX = 64            # globe resolution across (the hero's HERO_GLOBE_PX)
VIEW_LAT = 5.0           # seen from a little north
START_LON = -30.0        # Atlantic facing at rest
SUB = 3                  # sub-samples per pixel side (majority vote)
PALETTE = np.array([     # ocean, land, desert, tundra, ice
    (0x1e, 0x68, 0xf0),
    (0x39, 0xbe, 0x00),
    (0xc6, 0xd7, 0x6f),
    (0xd7, 0xd7, 0xd7),
    (0xff, 0xff, 0xff),
], dtype=np.uint8)
CODE = {".": 0, "g": 1, "d": 2, "t": 3, "i": 4}

# --- scene / output ----------------------------------------------------------
OUT = 340                # final GIF size (px, square)
SS = 4                   # supersample factor for the rim / card corners
FRAMES = 160             # frames per full turn (= one loop)
FRAME_MS = 50            # 160 x 50ms = an 8s turn
PX_SCALE = 4             # screen px per globe px -> 256px globe
BORDER = 4               # .wordmark-globe border (px)
SHADOW = 8               # .wordmark-globe box-shadow offset (px)
CORNER = 30              # rounded-card corner radius (px)
SEED = 7                 # deterministic cloud layout

# --- colors (brand / site) ---------------------------------------------------
SKY_TOP = (20, 80, 200)     # #1450c8  --sky-deep
SKY_MID = (30, 104, 240)    # #1e68f0  --sky
SKY_BOT = (46, 122, 242)    # #2e7af2
INK = (14, 41, 96)          # #0e2960  --ink
CLOUD_OPACITY = 0.72        # site .sky-puffs opacity

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


# ============================ world map ======================================
def load_world_map(arg: str | None):
    """Return (map[H, W] class indices, islands[(lat, lon)]) from world-map.ts."""
    if arg is None:
        with urllib.request.urlopen(WORLD_MAP_URL, timeout=30) as r:
            src = r.read().decode()
    else:
        path = arg
        if os.path.isdir(path):
            path = os.path.join(path, "site", "src", "lib", "world-map.ts")
        with open(path, encoding="utf-8") as f:
            src = f.read()
    w = int(re.search(r"MAP_W\s*=\s*(\d+)", src).group(1))
    h = int(re.search(r"MAP_H\s*=\s*(\d+)", src).group(1))
    rle = "".join(re.findall(r"'([^']*)'", re.search(r"MAP_RLE\s*=([^;]*);", src).group(1)))
    out = np.zeros(w * h, dtype=np.uint8)
    i = 0
    for cls, run in re.findall(r"([.gdti])(\d*)", rle):
        n = int(run) if run else 1
        out[i:i + n] = CODE[cls]
        i += n
    assert i == w * h, f"world map decoded to {i} cells, expected {w * h}"
    nums = [float(v) for v in re.findall(
        r"-?\d+(?:\.\d+)?", re.search(r"ISLANDS\s*=\s*\[([^\]]*)\]", src).group(1))]
    islands = list(zip(nums[0::2], nums[1::2]))
    return out.reshape(h, w), islands


# ============================ globe ==========================================
def globe_sampler(world, islands, n: int, lat: float):
    """Port of pixel-globe.ts globeSampler: returns lon -> classes[n, n]."""
    map_h, map_w = world.shape
    flat = world.reshape(-1)
    per_deg = map_w / 360
    lat0 = math.radians(lat)
    cos_l, sin_l = math.cos(lat0), math.sin(lat0)
    k = SUB * SUB
    s = np.arange(k)
    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = ((ii[..., None] + ((s % SUB) + 0.5) / SUB) / n) * 2 - 1
    y = 1 - ((jj[..., None] + ((s // SUB) + 0.5) / SUB) / n) * 2
    r2 = x * x + y * y
    scale = np.where(r2 > 0.998, np.sqrt(r2 / 0.998), 1.0)
    x, y = x / scale, y / scale
    z = np.sqrt(np.maximum(0, 1 - x * x - y * y))
    la = np.degrees(np.arcsin(np.clip(y * cos_l + z * sin_l, -1, 1)))
    row_of = np.clip(np.floor((90 - la) * per_deg), 0, map_h - 1).astype(np.int64)
    lon_of = (np.degrees(np.arctan2(x, z * cos_l - y * sin_l)) + 180).astype(np.float32)
    row_of = row_of.reshape(n * n, k)
    lon_of = lon_of.reshape(n * n, k)
    isl_cell = [min(map_h - 1, math.floor((90 - a) * per_deg)) * map_w
                + min(map_w - 1, math.floor((b + 180) * per_deg)) for a, b in islands]
    pix = np.arange(n * n)

    def sample(lon: float) -> np.ndarray:
        col = np.floor((lon_of + np.float32(lon)) * per_deg).astype(np.int64) % map_w
        cls = flat[row_of * map_w + col]                     # (P, K)
        counts = np.zeros((n * n, len(PALETTE)), dtype=np.int16)
        best = np.zeros(n * n, dtype=np.int64)
        for q in range(k):                                    # same tie-break as JS
            c = cls[:, q]
            counts[pix, c] += 1
            best = np.where(counts[pix, c] > counts[pix, best], c, best)
        out = best.astype(np.uint8)
        for (ila, ilo), cell in zip(islands, isl_cell):
            la_r, dl = math.radians(ila), math.radians(ilo - lon)
            if sin_l * math.sin(la_r) + cos_l * math.cos(la_r) * math.cos(dl) <= 0.05:
                continue
            gx = math.cos(la_r) * math.sin(dl)
            gy = cos_l * math.sin(la_r) - sin_l * math.cos(la_r) * math.cos(dl)
            i = min(n - 1, math.floor((gx + 1) / 2 * n))
            j = min(n - 1, math.floor((1 - gy) / 2 * n))
            p = j * n + i
            if out[p] == 0:
                c = flat[cell]
                out[p] = 1 if c == 0 else c
        return out.reshape(n, n)

    return sample


def disc_mask(size: int, cx: float, cy: float, r: float) -> Image.Image:
    """Anti-aliased filled circle as an L mask (supersampled)."""
    m = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(m).ellipse(((cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS),
                              fill=255)
    return m.resize((size, size), Image.LANCZOS)


class GlobeLayer:
    """Draws the styled globe (shadow, ink border, pixel disc) onto a scene."""

    def __init__(self, size: int):
        d = GLOBE_PX * PX_SCALE
        self.size = size
        # centre the globe + its offset shadow as one unit
        self.ox = (size - d - SHADOW) // 2
        self.oy = self.ox
        c = d / 2
        self.ink = Image.new("RGBA", (size, size), (*INK, 255))
        self.shadow = disc_mask(size, self.ox + c + SHADOW, self.oy + c + SHADOW, c + BORDER)
        self.border = disc_mask(size, self.ox + c, self.oy + c, c + BORDER)
        disc = disc_mask(d, c, c, c)
        self.disc = Image.new("L", (size, size), 0)
        self.disc.paste(disc, (self.ox, self.oy))

    def draw(self, scene: Image.Image, classes: np.ndarray) -> None:
        rgb = PALETTE[classes]
        art = Image.fromarray(rgb, "RGB").resize(
            (GLOBE_PX * PX_SCALE,) * 2, Image.NEAREST)
        full = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        full.paste(art, (self.ox, self.oy))
        scene.paste(self.ink, (0, 0), self.shadow)
        scene.paste(self.ink, (0, 0), self.border)
        scene.paste(full, (0, 0), self.disc)


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
            sp = sp.copy()
            sp.putalpha(a)
        x = (x0 - pan) % size
        for dx in (x - size, x, x + size):  # wrap copies on both edges
            layer.alpha_composite(sp, (int(round(dx)), int(round(y))))
    # global cloud-layer opacity, like .sky-puffs { opacity: .72 }
    a = layer.split()[3].point(lambda v: int(v * CLOUD_OPACITY))
    layer.putalpha(a)
    base.alpha_composite(layer)


# ============================ framing / export ===============================
def rounded_mask(size: int, radius: int) -> Image.Image:
    m = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        (0, 0, size * SS - 1, size * SS - 1), radius=radius * SS, fill=255)
    return m.resize((size, size), Image.LANCZOS)


def build_palette(sky: Image.Image, puffs) -> Image.Image:
    """One shared palette for every frame (no per-frame flicker, no dither
    noise on the flat pixel art): sky rows + cloud/globe/ink colours."""
    probe = sky.copy().convert("RGBA")
    paste_clouds(probe, puffs, pan=0, size=OUT)
    sw = Image.new("RGB", (OUT, 8))
    for i, c in enumerate([*map(tuple, PALETTE), INK]):
        sw.paste(c, (i * 8, 0, i * 8 + 8, 8))
    combo = Image.new("RGB", (OUT, OUT + 8))
    combo.paste(probe.convert("RGB"), (0, 0))
    combo.paste(sw, (0, OUT))
    return combo.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE)


def to_paletted(frame: Image.Image, pal_img: Image.Image, mask: Image.Image) -> Image.Image:
    """RGB scene -> P mode on the shared palette; outside the card is transparent."""
    pal = frame.convert("RGB").quantize(palette=pal_img, dither=Image.NONE)
    transp = mask.point(lambda v: 255 if v < 128 else 0).convert("1")
    pal.paste(255, transp)
    pal.info["transparency"] = 255
    return pal


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "profile", "assets")
    os.makedirs(out_dir, exist_ok=True)

    world, islands = load_world_map(sys.argv[1] if len(sys.argv) > 1 else None)
    sample = globe_sampler(world, islands, GLOBE_PX, VIEW_LAT)

    sky = make_sky(OUT)
    puffs = build_clouds(OUT)
    mask = rounded_mask(OUT, CORNER)
    globe = GlobeLayer(OUT)
    pal_img = build_palette(sky, puffs)

    frames = []
    for f in range(FRAMES):
        scene = sky.copy().convert("RGBA")
        paste_clouds(scene, puffs, pan=f / FRAMES * OUT, size=OUT)
        # west to east, as seen from space: features drift right
        globe.draw(scene, sample(START_LON - f / FRAMES * 360))
        frames.append(to_paletted(scene, pal_img, mask))

    gif_path = os.path.join(out_dir, "globe.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, disposal=2, transparency=255,
                   optimize=False)

    # static transparent globe (same styling) for the website project-card icon
    icon = Image.new("RGBA", (OUT, OUT), (0, 0, 0, 0))
    globe.draw(icon, sample(START_LON))
    png_path = os.path.join(out_dir, "globe.png")
    icon.save(png_path, optimize=True)

    print(f"wrote {gif_path} ({os.path.getsize(gif_path) // 1024} KB, "
          f"{OUT}px, {FRAMES} frames)")
    print(f"wrote {png_path} ({os.path.getsize(png_path) // 1024} KB)")


if __name__ == "__main__":
    main()
