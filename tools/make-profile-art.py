#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "numpy"]
# ///
"""
Art for the Web World Wide GitHub org profile (profile/README.md), drawn in
the website's own visual language.

Design contract (design-ui skill, kept small):
  * Purpose: in one glance, know this is Web World Wide and reach the projects.
  * Art direction: the site's sky as a wide editorial banner. A big stacked
    Archivo Black wordmark on the left, the spinning pixel-art Earth on the
    right as the one moving subject, a small tracked VT323 tagline. Quiet text
    navigation sits under it so the banner reads larger.
  * Typography: the site's existing system, not new fonts: Archivo Black
    (display), VT323 (utility labels). Fonts are fetched as TTF from the
    Fontsource CDN and cached in tools/.cache/.
  * Surface: site tokens. Ink #0e2960 borders and hard offset shadows, cherry
    #ec5a2b markers, sky gradient, pixel clouds.
  * Repeated detail: section heads reuse the site's eyebrow ("01 - BLOG"):
    cherry bar, tracked pixel caps, under GitHub's own h2 rule. Light and dark
    variants (served with <picture>).

Outputs (profile/assets/):
  banner.gif                       hero; the globe turns once per ~19s loop
  eyebrow-<name>-{light,dark}.png  section heads

Faithful to the live site:
  * Sky gradient      -> site/src/styles/sky.css
  * Pixel-art clouds  -> site/src/scripts/clouds.ts (same SHAPES + colors)
  * Pixel-art globe   -> site/src/lib/pixel-globe.ts (straight port of
                         globeSampler, verified pixel-identical) seen as in
                         scripts/globe-nav.ts, styled like .wordmark-globe.
  * Wordmark shadow   -> .wordmark text-shadow in styles/home.css
  * Eyebrow           -> .eyebrow in styles/home.css

The world map is read from the site's generated site/src/lib/world-map.ts.
Pass a path to that file or to a webworldwide-website checkout; with no
argument it is fetched from the site repo's main branch.

Images are drawn at 2x and shown at half size. GitHub and browsers cache
/raw/ images hard: ship a visual change under a new filename.

Run:  uv run tools/make-profile-art.py [path/to/world-map.ts | path/to/site-checkout]
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "profile", "assets")
CACHE = os.path.join(HERE, ".cache")

WORLD_MAP_URL = ("https://raw.githubusercontent.com/WebWorldWide/webworldwide-website/"
                 "main/site/src/lib/world-map.ts")
FONT_URL = "https://cdn.jsdelivr.net/fontsource/fonts/{}@latest/latin-400-normal.ttf"

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

# --- banner (2x pixels) ------------------------------------------------------
W, H = 1760, 600         # shown at ~880x300
CARD_BORDER = 6          # 3px ink border
CARD_SHADOW = 14         # 7px hard ink shadow, like the site's cards
CARD_RADIUS = 14
PX_SCALE = 7             # screen px per globe px -> 448px globe
G_BORDER = 8             # .wordmark-globe border (4px)
G_SHADOW = 16            # .wordmark-globe shadow (8px)
FRAMES = 240             # frames per full turn (= one loop)
FRAME_MS = 80            # 240 x 80ms = a ~19s turn (the site's is 24s)
SEED = 11                # deterministic cloud layout
KEEP_CLEAR = [(60, 50, 700, 590), (930, 40, 1450, 560)]  # wordmark, globe
AA = 4                   # supersample for masks

# --- colors (site tokens) ----------------------------------------------------
SKY_TOP = (20, 80, 200)     # #1450c8  --sky-deep
SKY_MID = (30, 104, 240)    # #1e68f0  --sky
SKY_BOT = (46, 122, 242)    # #2e7af2
INK = (14, 41, 96)          # #0e2960  --ink
CHERRY = (236, 90, 43)      # #ec5a2b  --cherry
WHITE = (255, 255, 255)
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



class GlobeLayer:
    """Styled globe (ink shadow, ink border, pixel disc) at a fixed spot."""

    def __init__(self, cx: int, cy: int):
        d = GLOBE_PX * PX_SCALE
        self.ox, self.oy = cx - d // 2, cy - d // 2
        c = d / 2
        self.ink = Image.new("RGBA", (W, H), (*INK, 255))
        self.shadow = disc_mask_wh(self.ox + c + G_SHADOW, self.oy + c + G_SHADOW, c + G_BORDER)
        self.border = disc_mask_wh(self.ox + c, self.oy + c, c + G_BORDER)
        self.disc = disc_mask_wh(self.ox + c, self.oy + c, c)

    def draw_frame(self, scene: Image.Image) -> None:
        scene.paste(self.ink, (0, 0), self.shadow)
        scene.paste(self.ink, (0, 0), self.border)

    def draw_globe(self, scene: Image.Image, classes: np.ndarray) -> None:
        art = Image.fromarray(PALETTE[classes], "RGB").resize(
            (GLOBE_PX * PX_SCALE,) * 2, Image.NEAREST)
        full = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        full.paste(art, (self.ox, self.oy))
        scene.paste(full, (0, 0), self.disc)


def disc_mask_wh(cx: float, cy: float, r: float) -> Image.Image:
    m = Image.new("L", (W * AA, H * AA), 0)
    ImageDraw.Draw(m).ellipse(((cx - r) * AA, (cy - r) * AA, (cx + r) * AA, (cy + r) * AA),
                              fill=255)
    return m.resize((W, H), Image.LANCZOS)


# ============================ fonts ==========================================
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.ttf")
    if not os.path.exists(path):
        with urllib.request.urlopen(FONT_URL.format(name), timeout=30) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    return ImageFont.truetype(path, size)


def tracked(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill, tracking: float) -> float:
    """Draw text with letter-spacing (em); returns the end x."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += fnt.getlength(ch) + tracking * fnt.size
    return x


def tracked_width(text: str, fnt, tracking: float) -> float:
    return sum(fnt.getlength(ch) + tracking * fnt.size for ch in text) - tracking * fnt.size


# ============================ sky ============================================
def make_sky(w: int, h: int) -> Image.Image:
    """Vertical gradient matching sky.css (#1450c8 -> #1e68f0@38% -> #2e7af2)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        f = y / (h - 1)
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



def paste_clouds(base: Image.Image) -> None:
    """A still, deterministic layer of puffs (far small/faint, near big/bright)."""
    rng = random.Random(SEED)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layers = [   # (count, scale_min, scale_max, op_min, op_max, top_min, top_max)
        (5, 3, 4, 0.30, 0.50, 0.02, 0.40),
        (5, 4, 6, 0.50, 0.70, 0.05, 0.75),
        (4, 6, 9, 0.70, 0.85, 0.30, 0.90),
        (2, 10, 13, 0.85, 0.95, 0.55, 0.95),
    ]
    for li, (count, smin, smax, omin, omax, tmin, tmax) in enumerate(layers):
        for _ in range(count):
            sp = cloud_sprite(rng.choice(SHAPES), rng.randint(smin, smax), CLOUD_OUTLINES[li])
            alpha = rng.uniform(omin, omax)
            sp.putalpha(sp.split()[3].point(lambda v, a=alpha: int(v * a)))
            for _ in range(200):  # keep the wordmark and the globe clear
                x = int(rng.uniform(-0.05, 1.0) * base.size[0])
                y = int(rng.uniform(tmin, tmax) * base.size[1])
                r = (x, y, x + sp.size[0], y + sp.size[1])
                if not any(r[0] < b[2] and r[2] > b[0] and r[1] < b[3] and r[3] > b[1]
                           for b in KEEP_CLEAR):
                    break
            layer.alpha_composite(sp, (x, y))
    layer.putalpha(layer.split()[3].point(lambda v: int(v * CLOUD_OPACITY)))
    base.alpha_composite(layer)


# ============================ banner =========================================
def wordmark(scene: Image.Image, x: int, y: int) -> int:
    """Stacked white wordmark with the site's ink text-shadow; returns bottom y."""
    fnt = font("archivo-black", 150)
    lead = 138
    d = ImageDraw.Draw(scene)
    lines = ["Web", "World", "Wide"]
    # .wordmark: 2px 0 ink, 0 2px ink, 2px 2px ink, 4px 4px ink/45%  (x2.5 here)
    soft = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    ds = ImageDraw.Draw(soft)
    for i, t in enumerate(lines):
        ds.text((x + 10, y + i * lead + 10), t, font=fnt, fill=(*INK, 115))
    scene.alpha_composite(soft)
    for i, t in enumerate(lines):
        for dx, dy in ((5, 0), (0, 5), (5, 5)):
            d.text((x + dx, y + i * lead + dy), t, font=fnt, fill=(*INK, 255))
        d.text((x, y + i * lead), t, font=fnt, fill=(*WHITE, 255))
    return y + (len(lines) - 1) * lead + fnt.getbbox("W")[3]


def tagline(scene: Image.Image, x: int, y: int) -> None:
    """Cherry bar + tracked pixel caps, like the site's eyebrow on the sky."""
    d = ImageDraw.Draw(scene)
    bar_w, bar_h, sh = 60, 16, 6
    fnt = font("vt323", 52)
    top = fnt.getbbox("M")[1]
    cap = fnt.getbbox("M")[3] - top
    by = y + top + (cap - bar_h) // 2
    d.rectangle((x + sh, by + sh, x + bar_w + sh - 1, by + bar_h + sh - 1), fill=INK)
    d.rectangle((x, by, x + bar_w - 1, by + bar_h - 1), fill=CHERRY)
    tx = x + bar_w + 26
    tracked(d, (tx + 3, y + 3), "MY WORLD ON THE WEB", fnt, INK, 0.22)
    tracked(d, (tx, y), "MY WORLD ON THE WEB", fnt, WHITE, 0.22)


def card_masks():
    """Rounded card + its hard offset shadow, inside a WxH canvas."""
    cw, ch = W - CARD_SHADOW, H - CARD_SHADOW

    def rr(x0, y0):
        m = Image.new("L", (W * AA, H * AA), 0)
        ImageDraw.Draw(m).rounded_rectangle(
            (x0 * AA, y0 * AA, (x0 + cw) * AA - 1, (y0 + ch) * AA - 1),
            radius=CARD_RADIUS * AA, fill=255)
        return m.resize((W, H), Image.LANCZOS)

    def inner():
        b = CARD_BORDER
        m = Image.new("L", (W * AA, H * AA), 0)
        ImageDraw.Draw(m).rounded_rectangle(
            (b * AA, b * AA, (cw - b) * AA - 1, (ch - b) * AA - 1),
            radius=(CARD_RADIUS - b // 2) * AA, fill=255)
        return m.resize((W, H), Image.LANCZOS)

    return rr(0, 0), rr(CARD_SHADOW, CARD_SHADOW), inner()


def make_banner(sample) -> None:
    card, shadow, inner = card_masks()
    cw, ch = W - CARD_SHADOW, H - CARD_SHADOW

    sky = make_sky(W, H).convert("RGBA")
    paste_clouds(sky)
    globe = GlobeLayer(cx=1190, cy=ch // 2 - 6)
    globe.draw_frame(sky)
    bottom = wordmark(sky, 96, 66)
    tagline(sky, 102, bottom + 34)

    # flatten: shadow, ink card, sky inside the border
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ink = Image.new("RGBA", (W, H), (*INK, 255))
    base.paste(ink, (0, 0), shadow)
    base.paste(ink, (0, 0), card)
    base.paste(sky, (0, 0), inner)
    alpha = Image.composite(Image.new("L", (W, H), 255), Image.new("L", (W, H), 0),
                            Image.composite(card, shadow, card))
    outside = alpha.point(lambda v: 255 if v < 128 else 0).convert("1")

    # one shared palette for every frame, built from a frame + exact swatches
    probe = base.copy()
    globe.draw_globe(probe, sample(START_LON))
    sw = Image.new("RGB", (W, 8))
    for i, c in enumerate([*map(tuple, PALETTE), INK, CHERRY, WHITE]):
        sw.paste(c, (i * 8, 0, i * 8 + 8, 8))
    combo = Image.new("RGB", (W, H + 8))
    combo.paste(probe.convert("RGB"), (0, 0))
    combo.paste(sw, (0, H))
    pal = combo.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE)

    frames = []
    for f in range(FRAMES):
        scene = base.copy()
        # west to east, as seen from space: features drift right
        globe.draw_globe(scene, sample(START_LON - f / FRAMES * 360))
        q = scene.convert("RGB").quantize(palette=pal, dither=Image.NONE)
        q.paste(255, outside)
        q.info["transparency"] = 255
        frames.append(q)

    path = os.path.join(OUT_DIR, "banner.gif")
    # disposal=1 + optimize: Pillow stores only the changed (globe) rectangle
    # after the first frame, so the wide banner costs about what the globe does.
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_MS,
                   loop=0, disposal=1, transparency=255, optimize=True)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB, {W}x{H}, {FRAMES} frames)")


# ============================ eyebrows =======================================
EYEBROWS = {"projects": "01 \u2014 PROJECTS", "elsewhere": "02 \u2014 ELSEWHERE"}


def make_eyebrows() -> None:
    """Site eyebrow: cherry bar with ink shadow, tracked VT323 caps.
    light: ink text on GitHub's white; dark: white text with ink shadow."""
    fnt = font("vt323", 64)
    bar_w, bar_h, sh = 68, 18, 6
    top = fnt.getbbox("M")[1]
    cap = fnt.getbbox("M")[3] - top
    for key, text in EYEBROWS.items():
        tw = tracked_width(text, fnt, 0.2)
        w = int(bar_w + 30 + tw + 12)
        h = int(cap + 12)
        for mode in ("light", "dark"):
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            by = (cap - bar_h) // 2
            d.rectangle((sh, by + sh, bar_w + sh - 1, by + bar_h + sh - 1),
                        fill=(*INK, 255) if mode == "light" else (0, 0, 0, 160))
            d.rectangle((0, by, bar_w - 1, by + bar_h - 1), fill=(*CHERRY, 255))
            tx = bar_w + 30
            if mode == "dark":
                tracked(d, (tx + 4, -top + 4), text, fnt, (*INK, 255), 0.2)
                tracked(d, (tx, -top), text, fnt, (*WHITE, 255), 0.2)
            else:
                tracked(d, (tx, -top), text, fnt, (*INK, 255), 0.2)
            p = os.path.join(OUT_DIR, f"eyebrow-{key}-{mode}.png")
            im.save(p, optimize=True)
            print(f"wrote {p} ({w}x{h})")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    world, islands = load_world_map(sys.argv[1] if len(sys.argv) > 1 else None)
    sample = globe_sampler(world, islands, GLOBE_PX, VIEW_LAT)
    make_banner(sample)
    make_eyebrows()


if __name__ == "__main__":
    main()
