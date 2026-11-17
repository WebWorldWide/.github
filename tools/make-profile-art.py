#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "numpy", "cairosvg"]
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
  banner-v2.gif                    hero; the globe turns and the clouds drift
                                   one width per ~19s loop
  link-*.png                       nav pills (the site's top-bar pill)
  social-*.png                     social tiles (the site's Socials tiles,
                                   same Simple Icons brand glyphs)
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
TAGLINE_BAND = (470, 570)  # clouds never cross the small tagline (y range)
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



def build_puffs():
    """Deterministic puffs (far small/faint -> near big/bright): (sprite, x, y)."""
    rng = random.Random(SEED)
    layers = [   # (count, scale_min, scale_max, op_min, op_max, top_min, top_max)
        (5, 3, 4, 0.30, 0.50, 0.02, 0.40),
        (5, 4, 6, 0.50, 0.70, 0.05, 0.75),
        (4, 6, 9, 0.70, 0.85, 0.20, 0.90),
        (2, 10, 13, 0.85, 0.95, 0.40, 0.95),
    ]
    puffs = []
    for li, (count, smin, smax, omin, omax, tmin, tmax) in enumerate(layers):
        for k in range(count):
            sp = cloud_sprite(rng.choice(SHAPES), rng.randint(smin, smax), CLOUD_OUTLINES[li])
            alpha = rng.uniform(omin, omax) * CLOUD_OPACITY
            sp.putalpha(sp.split()[3].point(lambda v, a=alpha: int(v * a)))
            for _ in range(200):
                y = int(rng.uniform(tmin, tmax) * H)
                if y + sp.size[1] < TAGLINE_BAND[0] or y > TAGLINE_BAND[1]:
                    break
            # spread each layer evenly across the width so the drift never
            # opens a long empty stretch
            x = (k + rng.uniform(0.1, 0.9)) / count * W
            puffs.append((sp, x, y))
    return puffs


def draw_clouds(scene: Image.Image, puffs, pan: float) -> None:
    """Puffs drifted left by `pan` px, wrapping seamlessly across W."""
    for sp, x0, y in puffs:
        x = (x0 - pan) % W
        for dx in (x - W, x):
            if dx < W and dx + sp.size[0] > 0:
                scene.alpha_composite(sp, (int(round(dx)), y))


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
    puffs = build_puffs()
    globe = GlobeLayer(cx=1190, cy=ch // 2 - 6)
    ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))     # globe ink ring + shadow
    globe.draw_frame(ring)
    text = Image.new("RGBA", (W, H), (0, 0, 0, 0))     # wordmark + tagline
    bottom = wordmark(text, 96, 66)
    tagline(text, 102, bottom + 34)
    ink = Image.new("RGBA", (W, H), (*INK, 255))
    frame_bg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    frame_bg.paste(ink, (0, 0), shadow)
    frame_bg.paste(ink, (0, 0), card)
    alpha = Image.composite(card, shadow, card)
    outside = alpha.point(lambda v: 255 if v < 128 else 0).convert("1")

    def render(f: int) -> Image.Image:
        scene = sky.copy()
        draw_clouds(scene, puffs, f / FRAMES * W)          # one width per loop
        scene.alpha_composite(ring)
        # west to east, as seen from space: features drift right
        globe.draw_globe(scene, sample(START_LON - f / FRAMES * 360))
        scene.alpha_composite(text)
        out = frame_bg.copy()
        out.paste(scene, (0, 0), inner)
        return out

    # one shared palette for every frame, built from a frame + exact swatches
    probe = render(0)
    sw = Image.new("RGB", (W, 8))
    for i, c in enumerate([*map(tuple, PALETTE), INK, CHERRY, WHITE, *CLOUD_OUTLINES]):
        sw.paste(c, (i * 8, 0, i * 8 + 8, 8))
    combo = Image.new("RGB", (W, H + 8))
    combo.paste(probe.convert("RGB"), (0, 0))
    combo.paste(sw, (0, H))
    pal = combo.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE)

    full = []
    for f in range(FRAMES):
        q = render(f).convert("RGB").quantize(palette=pal, dither=Image.NONE)
        q.paste(255, outside)
        full.append(np.asarray(q))

    # Transparency optimisation (what gifsicle -O2 does): every frame after
    # the first only stores the pixels that changed; the rest are the
    # transparent index, which LZW packs to almost nothing.
    frames = []
    prev = None
    for idx in full:
        if prev is None:
            out = idx
        else:
            out = np.where(idx == prev, 255, idx).astype(np.uint8)
        prev = idx
        im = Image.fromarray(out, "P")
        im.putpalette(pal.getpalette())
        im.info["transparency"] = 255
        frames.append(im)

    path = os.path.join(OUT_DIR, "banner-v2.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_MS,
                   loop=0, disposal=1, transparency=255, optimize=False)
    print(f"wrote {path} ({os.path.getsize(path) // 1024} KB, {W}x{H}, {FRAMES} frames)")
    return full, pal


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


# ============================ links ==========================================
CHERRY_STRONG = (176, 71, 8)   # #b04708: white text on it passes AA
INK_SOFT = (42, 74, 144)       # #2a4a90
PAPER = (255, 255, 255)

# Brand glyphs: Simple Icons paths, the same ones the site's Socials.astro uses.
GLYPHS = {
    "youtube": "M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z",
    "x": "M14.234 10.162 22.977 0h-2.072l-7.591 8.824L7.251 0H.258l9.168 13.343L.258 24H2.33l8.016-9.318L16.749 24h6.993zm-2.837 3.299-.929-1.329L3.076 1.56h3.182l5.965 8.532.929 1.329 7.754 11.09h-3.182z",
    "bluesky": "M5.202 2.857C7.954 4.922 10.913 9.11 12 11.358c1.087-2.247 4.046-6.436 6.798-8.501C20.783 1.366 24 .213 24 3.883c0 .732-.42 6.156-.667 7.037-.856 3.061-3.978 3.842-6.755 3.37 4.854.826 6.089 3.562 3.422 6.299-5.065 5.196-7.28-1.304-7.847-2.97-.104-.305-.152-.448-.153-.327 0-.121-.05.022-.153.327-.568 1.666-2.782 8.166-7.847 2.97-2.667-2.737-1.432-5.473 3.422-6.3-2.777.473-5.899-.308-6.755-3.369C.42 10.04 0 4.615 0 3.883c0-3.67 3.217-2.517 5.202-1.026",
    "mastodon": "M23.268 5.313c-.35-2.578-2.617-4.61-5.304-5.004C17.51.242 15.792 0 11.813 0h-.03c-3.98 0-4.835.242-5.288.309C3.882.692 1.496 2.518.917 5.127.64 6.412.61 7.837.661 9.143c.074 1.874.088 3.745.26 5.611.118 1.24.325 2.47.62 3.68.55 2.237 2.777 4.098 4.96 4.857 2.336.792 4.849.923 7.256.38.265-.061.527-.132.786-.213.585-.184 1.27-.39 1.774-.753a.057.057 0 0 0 .023-.043v-1.809a.052.052 0 0 0-.02-.041.053.053 0 0 0-.046-.01 20.282 20.282 0 0 1-4.709.545c-2.73 0-3.463-1.284-3.674-1.818a5.593 5.593 0 0 1-.319-1.433.053.053 0 0 1 .066-.054c1.517.363 3.072.546 4.632.546.376 0 .75 0 1.125-.01 1.57-.044 3.224-.124 4.768-.422.038-.008.077-.015.11-.024 2.435-.464 4.753-1.92 4.989-5.604.008-.145.03-1.52.03-1.67.002-.512.167-3.63-.024-5.545zm-3.748 9.195h-2.561V8.29c0-1.309-.55-1.976-1.67-1.976-1.23 0-1.846.79-1.846 2.35v3.403h-2.546V8.663c0-1.56-.617-2.35-1.848-2.35-1.112 0-1.668.668-1.67 1.977v6.218H4.822V8.102c0-1.31.337-2.35 1.011-3.12.696-.77 1.608-1.164 2.74-1.164 1.311 0 2.302.5 2.962 1.498l.638 1.06.638-1.06c.66-.999 1.65-1.498 2.96-1.498 1.13 0 2.043.395 2.74 1.164.675.77 1.012 1.81 1.012 3.12z",
    "reddit": "M12 0C5.373 0 0 5.373 0 12c0 3.314 1.343 6.314 3.515 8.485l-2.286 2.286C.775 23.225 1.097 24 1.738 24H12c6.627 0 12-5.373 12-12S18.627 0 12 0Zm4.388 3.199c1.104 0 1.999.895 1.999 1.999 0 1.105-.895 2-1.999 2-.946 0-1.739-.657-1.947-1.539v.002c-1.147.162-2.032 1.15-2.032 2.341v.007c1.776.067 3.4.567 4.686 1.363.473-.363 1.064-.58 1.707-.58 1.547 0 2.802 1.254 2.802 2.802 0 1.117-.655 2.081-1.601 2.531-.088 3.256-3.637 5.876-7.997 5.876-4.361 0-7.905-2.617-7.998-5.87-.954-.447-1.614-1.415-1.614-2.538 0-1.548 1.255-2.802 2.803-2.802.645 0 1.239.218 1.712.585 1.275-.79 2.881-1.291 4.64-1.365v-.01c0-1.663 1.263-3.034 2.88-3.207.188-.911.993-1.595 1.959-1.595Zm-8.085 8.376c-.784 0-1.459.78-1.506 1.797-.047 1.016.64 1.429 1.426 1.429.786 0 1.371-.369 1.418-1.385.047-1.017-.553-1.841-1.338-1.841Zm7.406 0c-.786 0-1.385.824-1.338 1.841.047 1.017.634 1.385 1.418 1.385.785 0 1.473-.413 1.426-1.429-.046-1.017-.721-1.797-1.506-1.797Zm-3.703 4.013c-.974 0-1.907.048-2.77.135-.147.015-.241.168-.183.305.483 1.154 1.622 1.964 2.953 1.964 1.33 0 2.47-.81 2.953-1.964.057-.137-.037-.29-.184-.305-.863-.087-1.795-.135-2.769-.135Z",
    "instagram": "M7.0301.084c-1.2768.0602-2.1487.264-2.911.5634-.7888.3075-1.4575.72-2.1228 1.3877-.6652.6677-1.075 1.3368-1.3802 2.127-.2954.7638-.4956 1.6365-.552 2.914-.0564 1.2775-.0689 1.6882-.0626 4.947.0062 3.2586.0206 3.6671.0825 4.9473.061 1.2765.264 2.1482.5635 2.9107.308.7889.72 1.4573 1.388 2.1228.6679.6655 1.3365 1.0743 2.1285 1.38.7632.295 1.6361.4961 2.9134.552 1.2773.056 1.6884.069 4.9462.0627 3.2578-.0062 3.668-.0207 4.9478-.0814 1.28-.0607 2.147-.2652 2.9098-.5633.7889-.3086 1.4578-.72 2.1228-1.3881.665-.6682 1.0745-1.3378 1.3795-2.1284.2957-.7632.4966-1.636.552-2.9124.056-1.2809.0692-1.6898.063-4.948-.0063-3.2583-.021-3.6668-.0817-4.9465-.0607-1.2797-.264-2.1487-.5633-2.9117-.3084-.7889-.72-1.4568-1.3876-2.1228C21.2982 1.33 20.628.9208 19.8378.6165 19.074.321 18.2017.1197 16.9244.0645 15.6471.0093 15.236-.005 11.977.0014 8.718.0076 8.31.0215 7.0301.0839m.1402 21.6932c-1.17-.0509-1.8053-.2453-2.2287-.408-.5606-.216-.96-.4771-1.3819-.895-.422-.4178-.6811-.8186-.9-1.378-.1644-.4234-.3624-1.058-.4171-2.228-.0595-1.2645-.072-1.6442-.079-4.848-.007-3.2037.0053-3.583.0607-4.848.05-1.169.2456-1.805.408-2.2282.216-.5613.4762-.96.895-1.3816.4188-.4217.8184-.6814 1.3783-.9003.423-.1651 1.0575-.3614 2.227-.4171 1.2655-.06 1.6447-.072 4.848-.079 3.2033-.007 3.5835.005 4.8495.0608 1.169.0508 1.8053.2445 2.228.408.5608.216.96.4754 1.3816.895.4217.4194.6816.8176.9005 1.3787.1653.4217.3617 1.056.4169 2.2263.0602 1.2655.0739 1.645.0796 4.848.0058 3.203-.0055 3.5834-.061 4.848-.051 1.17-.245 1.8055-.408 2.2294-.216.5604-.4763.96-.8954 1.3814-.419.4215-.8181.6811-1.3783.9-.4224.1649-1.0577.3617-2.2262.4174-1.2656.0595-1.6448.072-4.8493.079-3.2045.007-3.5825-.006-4.848-.0608M16.953 5.5864A1.44 1.44 0 1 0 18.39 4.144a1.44 1.44 0 0 0-1.437 1.4424M5.8385 12.012c.0067 3.4032 2.7706 6.1557 6.173 6.1493 3.4026-.0065 6.157-2.7701 6.1506-6.1733-.0065-3.4032-2.771-6.1565-6.174-6.1498-3.403.0067-6.156 2.771-6.1496 6.1738M8 12.0077a4 4 0 1 1 4.008 3.9921A3.9996 3.9996 0 0 1 8 12.0077",
    "threads": "M12.186 24h-.007c-3.581-.024-6.334-1.205-8.184-3.509C2.35 18.44 1.5 15.586 1.472 12.01v-.017c.03-3.579.879-6.43 2.525-8.482C5.845 1.205 8.6.024 12.18 0h.014c2.746.02 5.043.725 6.826 2.098 1.677 1.29 2.858 3.13 3.509 5.467l-2.04.569c-1.104-3.96-3.898-5.984-8.304-6.015-2.91.022-5.11.936-6.54 2.717C4.307 6.504 3.616 8.914 3.589 12c.027 3.086.718 5.496 2.057 7.164 1.43 1.783 3.631 2.698 6.54 2.717 2.623-.02 4.358-.631 5.8-2.045 1.647-1.613 1.618-3.593 1.09-4.798-.31-.71-.873-1.3-1.634-1.75-.192 1.352-.622 2.446-1.284 3.272-.886 1.102-2.14 1.704-3.73 1.79-1.202.065-2.361-.218-3.259-.801-1.063-.689-1.685-1.74-1.752-2.964-.065-1.19.408-2.285 1.33-3.082.88-.76 2.119-1.207 3.583-1.291a13.853 13.853 0 0 1 3.02.142c-.126-.742-.375-1.332-.75-1.757-.513-.586-1.308-.883-2.359-.89h-.029c-.844 0-1.992.232-2.721 1.32L7.734 7.847c.98-1.454 2.568-2.256 4.478-2.256h.044c3.194.02 5.097 1.975 5.287 5.388.108.046.216.094.321.142 1.49.7 2.58 1.761 3.154 3.07.797 1.82.871 4.79-1.548 7.158-1.85 1.81-4.094 2.628-7.277 2.65Zm1.003-11.69c-.242 0-.487.007-.739.021-1.836.103-2.98.946-2.916 2.143.067 1.256 1.452 1.839 2.784 1.767 1.224-.065 2.818-.543 3.086-3.71a10.5 10.5 0 0 0-2.215-.221z",
    # Feather-style RSS + a pen nib for the blog (simple, bold at small sizes)
    "rss": "M3.75 3a.75.75 0 0 0-.75.75v1.5c0 .414.336.75.75.75H4c6.075 0 11 4.925 11 11v.25c0 .414.336.75.75.75h1.5a.75.75 0 0 0 .75-.75V17C18 9.268 11.732 3 4 3h-.25ZM3 10.75a.75.75 0 0 1 .75-.75H4a7 7 0 0 1 7 7v.25a.75.75 0 0 1-.75.75h-1.5a.75.75 0 0 1-.75-.75V17a4 4 0 0 0-4-4h-.25a.75.75 0 0 1-.75-.75v-1.5ZM3 16a2 2 0 1 1 4 0 2 2 0 0 1-4 0Z",
}
GLYPH_VIEWBOX = {"rss": "0 0 21 21"}

SOCIALS = [  # key, name, handle, brand colour(s)
    ("youtube", "YouTube", "@web_world_wide", ["#ff0033"]),
    ("x", "X", "@thewebworldwide", ["#000000"]),
    ("bluesky", "Bluesky", "@web-world-wide.bsky.social", ["#0085ff"]),
    ("mastodon", "Mastodon", "@webworldwide@mastodon.social", ["#6364ff"]),
    ("reddit", "Reddit", "u/web_world_wide", ["#ff4500"]),
    ("instagram", "Instagram", "@thewebworldwide",
     ["#feda75", "#fa7e1e", "#d62976", "#962fbf", "#4f5bd5"]),
    ("threads", "Threads", "@thewebworldwide", ["#000000"]),
]


def glyph(key: str, size: int, color: str = "#fff") -> Image.Image:
    import cairosvg
    import io
    vb = GLYPH_VIEWBOX.get(key, "0 0 24 24")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{size}" '
           f'height="{size}"><path fill="{color}" d="{GLYPHS[key]}"/></svg>')
    return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert("RGBA")


def hexrgb(h: str):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def brand_fill(size, colors) -> Image.Image:
    """Solid brand colour, or a 135deg gradient through the stops."""
    w, h = size
    if len(colors) == 1:
        return Image.new("RGBA", size, (*hexrgb(colors[0]), 255))
    stops = np.array([hexrgb(c) for c in colors], np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    t = ((xx / max(w - 1, 1)) + (yy / max(h - 1, 1))) / 2 * (len(stops) - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, len(stops) - 2)
    f = (t - i0)[..., None]
    rgb = stops[i0] * (1 - f) + stops[i0 + 1] * f
    a = np.full((h, w, 1), 255, np.float32)
    return Image.fromarray(np.concatenate([rgb, a], 2).astype(np.uint8), "RGBA")


def rounded(size, r, scale=AA) -> Image.Image:
    w, h = size
    m = Image.new("L", (w * scale, h * scale), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * scale - 1, h * scale - 1), radius=r * scale,
                                        fill=255)
    return m.resize(size, Image.LANCZOS)


def boxed(w: int, h: int, r: int, border: int, shadow: int, fill_img: Image.Image,
          shadow_rgb=INK) -> Image.Image:
    """The site's object: fill inside an ink border, hard ink offset shadow."""
    im = Image.new("RGBA", (w + shadow, h + shadow), (0, 0, 0, 0))
    m = rounded((w, h), r)
    im.paste(Image.new("RGBA", (w, h), (*shadow_rgb, 255)), (shadow, shadow), m)
    im.paste(Image.new("RGBA", (w, h), (*INK, 255)), (0, 0), m)
    inner = rounded((w - 2 * border, h - 2 * border), max(r - border, 1))
    im.paste(fill_img.crop((0, 0, w - 2 * border, h - 2 * border)), (border, border), inner)
    return im


def social_tile(key, name, handle, colors) -> Image.Image:
    """The site's .elsewhere-tile at 2x: brand stripe, icon box, serif name,
    pixel handle, arrow."""
    w, h, b, sh, r = 392, 152, 6, 10, 12
    body = Image.new("RGBA", (w, h), (*PAPER, 255))
    body.paste(brand_fill((w, 16), colors), (0, 0))               # top stripe
    d = ImageDraw.Draw(body)
    # icon box: brand fill, 2px ink border, 3px ink shadow
    ib, ix, iy = 84, 22, 32
    box = boxed(ib, ib, 8, 4, 6, brand_fill((ib, ib), colors))
    body.alpha_composite(box, (ix, iy))
    g = glyph(key, 46)
    body.alpha_composite(g, (ix + (ib - 46) // 2, iy + (ib - 46) // 2))
    tx = ix + ib + 6 + 22
    serif = font("cormorant-garamond", 46)
    pix = font("vt323", 26)
    # handle wraps at "@", "." and "/" seams (like the site), two lines max
    maxw = w - b - tx - 52
    parts = re.split(r"(?=[@./])", handle.upper())
    parts = [p for p in parts if p]
    lines, cur = [], ""
    for prt in parts:
        if cur and tracked_width(cur + prt, pix, 0.08) > maxw:
            lines.append(cur)
            cur = prt
        else:
            cur += prt
    lines.append(cur)
    lines = lines[:2]
    block = 46 + 4 + 24 * len(lines)
    y0 = 16 + (h - 16 - b - block) // 2 - 4
    d.text((tx, y0 + 40), name, font=serif, fill=INK, anchor="ls")
    for k, ln in enumerate(lines):
        tracked(d, (tx, y0 + 50 + k * 24 - pix.getbbox("M")[1] + 2), ln, pix, INK_SOFT, 0.08)
    d.text((w - b - 40, h // 2 + 6), "→", font=font("vt323", 40), fill=CHERRY_STRONG,
           anchor="mm")
    tile = boxed(w, h, r, b, sh, body)
    # the site's 2px white inset line inside the border
    ImageDraw.Draw(tile).rounded_rectangle((b, b, w - b - 1, h - b - 1), radius=r - b,
                                           outline=(255, 255, 255, 255), width=3)
    tile.paste(brand_fill((w - 2 * b, 16), colors), (b, b),
               rounded((w - 2 * b, 16), 0))                       # stripe over inset
    return tile


def pill(label: str, icon: Image.Image | None, primary: bool) -> Image.Image:
    """The site's nav pill: VT323 caps, 3px ink border, hard ink shadow."""
    pix = font("vt323", 38)
    h, b, sh = 76, 6, 8
    tw = tracked_width(label, pix, 0.08)
    pad = 30
    iw = 0 if icon is None else icon.size[0] + 14
    w = int(pad + iw + tw + pad)
    bg = CHERRY_STRONG if primary else PAPER
    fg = WHITE if primary else INK
    body = Image.new("RGBA", (w, h), (*bg, 255))
    d = ImageDraw.Draw(body)
    x = pad - b
    if icon is not None:
        body.alpha_composite(icon, (x, (h - 2 * b - icon.size[1]) // 2))
        x += iw
    top = pix.getbbox("M")[1]
    cap = pix.getbbox("M")[3] - top
    tracked(d, (x, (h - 2 * b - cap) // 2 - top), label, pix, fg, 0.08)
    return boxed(w, h, h // 2, b, sh, body)


def globe_icon(size: int) -> Image.Image:
    """The site's favicon, scaled crisp."""
    fav = Image.open(os.path.join(OUT_DIR, "www-favicon.png")).convert("RGBA")
    return fav.resize((size, size), Image.LANCZOS)


def make_links() -> None:
    pills = [
        ("site", "WEBWORLDWIDE.ONLINE", globe_icon(40), True),
        ("blog", "BLOG", None, False),
        ("youtube", "YOUTUBE", glyph("youtube", 34, "#ff0033"), False),
        ("rss", "RSS", glyph("rss", 30, "#ec5a2b"), False),
    ]
    for key, label, icon, primary in pills:
        im = pill(label, icon, primary)
        p = os.path.join(OUT_DIR, f"link-{key}.png")
        im.save(p, optimize=True)
        print(f"wrote {p} ({im.size[0]}x{im.size[1]})")
    for key, name, handle, colors in SOCIALS:
        im = social_tile(key, name, handle, colors)
        p = os.path.join(OUT_DIR, f"social-{key}.png")
        im.save(p, optimize=True)
        print(f"wrote {p} ({im.size[0]}x{im.size[1]})")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    world, islands = load_world_map(sys.argv[1] if len(sys.argv) > 1 else None)
    sample = globe_sampler(world, islands, GLOBE_PX, VIEW_LAT)
    make_banner(sample)
    make_eyebrows()
    make_links()


if __name__ == "__main__":
    main()
