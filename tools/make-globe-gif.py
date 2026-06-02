#!/usr/bin/env python3
"""
Render the Web World Wide spinning wireframe globe as a seamless, looping GIF.

This is an offline port of the site's live WebGL globe
(webworldwide-website/site/src/scripts/globe-nav.ts): a parametric wireframe
sphere of 14 meridians + 9 parallels (96 segments / line), viewed through a
perspective camera (fov 34, z = 3.4) and rotated about the Y axis.

Seamless-loop trick: the wireframe is invariant under a 2*pi/14 rotation
(parallels are circles about Y; the 14 meridians permute onto themselves), so we
only animate one 1/14-turn slice -> a perfect loop in a handful of frames.

Lines are drawn back-to-front (painter's algorithm) with a depth-based color
fade (full brand sky-blue at the front, lightened toward the back) so the sphere
reads as 3D without needing per-pixel alpha. Rendered at 4x and downsampled for
clean anti-aliasing, then exported with a transparent background that reads on
both GitHub light and dark themes.

Requires: Pillow, numpy.  Outputs profile/assets/globe.gif and globe.png.
"""
from __future__ import annotations
import math
import os
import numpy as np
from PIL import Image, ImageDraw

# --- geometry (mirrors globe-nav.ts) -----------------------------------------
MERIDIANS = 14
PARALLELS = 9
LINE_SEGS = 96
CAM_DIST = 3.4          # camera.position.z
FOV_DEG = 34.0          # PerspectiveCamera vertical fov
TILT_X_DEG = 12.0       # slight forward tilt so it reads as a globe (site is 0)

# --- output -------------------------------------------------------------------
OUT = 360               # final GIF size (px)
SS = 4                  # supersample factor for anti-aliasing
SIZE = OUT * SS
FRAMES = 24             # frames across one 1/14 turn
MARGIN = 0.90           # fraction of half-frame the sphere fills
LINE_W = 2.4 * SS       # line width at render scale

# --- brand colors -------------------------------------------------------------
FRONT = np.array([30, 104, 240], dtype=float)    # #1e68f0 sky blue (near edge)
BACK = np.array([150, 196, 246], dtype=float)     # lightened blue (far side)

HALF_FOV = math.radians(FOV_DEG) / 2.0
TAN_HALF = math.tan(HALF_FOV)
TILT_X = math.radians(TILT_X_DEG)


def build_lines() -> list[np.ndarray]:
    """Return each meridian/parallel as an (N,3) array of unit-sphere points."""
    lines: list[np.ndarray] = []
    for i in range(MERIDIANS):
        a = (i / MERIDIANS) * math.pi * 2
        j = np.arange(LINE_SEGS + 1)
        t = (j / LINE_SEGS) * math.pi
        pts = np.stack(
            [np.sin(t) * math.cos(a), np.cos(t), np.sin(t) * math.sin(a)], axis=1
        )
        lines.append(pts)
    for i in range(1, PARALLELS):
        phi = (i / PARALLELS) * math.pi
        y = math.cos(phi)
        r = math.sin(phi)
        j = np.arange(LINE_SEGS + 1)
        t = (j / LINE_SEGS) * math.pi * 2
        pts = np.stack(
            [r * np.cos(t), np.full_like(t, y), r * np.sin(t)], axis=1
        )
        lines.append(pts)
    return lines


def rot_y(pts: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([x * c + z * s, y, -x * s + z * c], axis=1)


def tilt_x(pts: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.stack([x, y * c - z * s, y * s + z * c], axis=1)


def project(pts: np.ndarray):
    """Perspective-project to pixel coords; return (xy, depth_z_camera)."""
    z = pts[:, 2]
    d = CAM_DIST - z                       # distance along view axis (>0)
    ndc_x = (pts[:, 0] / d) / TAN_HALF
    ndc_y = (pts[:, 1] / d) / TAN_HALF
    px = SIZE / 2 + ndc_x * (SIZE / 2) * MARGIN
    py = SIZE / 2 - ndc_y * (SIZE / 2) * MARGIN
    return np.stack([px, py], axis=1), z


def render_frame(lines, theta: float) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    segments = []  # (depth, (x0,y0,x1,y1), color)
    for pts in lines:
        p = tilt_x(rot_y(pts, theta), TILT_X)
        xy, z = project(p)
        for k in range(len(p) - 1):
            zmid = (z[k] + z[k + 1]) / 2.0
            t = (zmid + 1.0) / 2.0          # 0 (back) .. 1 (front)
            col = BACK + (FRONT - BACK) * t
            color = (int(col[0]), int(col[1]), int(col[2]), 255)
            segments.append(
                (zmid, (xy[k, 0], xy[k, 1], xy[k + 1, 0], xy[k + 1, 1]), color)
            )

    segments.sort(key=lambda s: s[0])       # back first, front last
    r = LINE_W / 2.0
    for _, (x0, y0, x1, y1), color in segments:
        draw.line((x0, y0, x1, y1), fill=color, width=int(round(LINE_W)))
        # round caps so joints don't gap
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=color)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=color)

    return img.resize((OUT, OUT), Image.LANCZOS)


def to_paletted(frame: Image.Image) -> Image.Image:
    """RGBA -> P mode with a reserved transparent index (binary alpha)."""
    rgb = frame.convert("RGB")
    pal = rgb.quantize(colors=48, method=Image.MEDIANCUT, dither=Image.NONE)
    alpha = frame.split()[3]
    mask = alpha.point(lambda a: 255 if a >= 128 else 0)  # threshold 50%
    pal.paste(255, mask.point(lambda a: 255 if a == 0 else 0).convert("1"))
    pal.info["transparency"] = 255
    return pal


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "profile", "assets")
    os.makedirs(out_dir, exist_ok=True)

    lines = build_lines()
    period = (2 * math.pi) / MERIDIANS
    rgba = [render_frame(lines, k / FRAMES * period) for k in range(FRAMES)]

    frames = [to_paletted(f) for f in rgba]
    gif_path = os.path.join(out_dir, "globe.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        disposal=2,
        transparency=255,
        optimize=True,
    )
    png_path = os.path.join(out_dir, "globe.png")
    rgba[0].save(png_path)

    print(f"wrote {gif_path} ({os.path.getsize(gif_path) // 1024} KB, "
          f"{OUT}px, {FRAMES} frames)")
    print(f"wrote {png_path} ({os.path.getsize(png_path) // 1024} KB)")


if __name__ == "__main__":
    main()
