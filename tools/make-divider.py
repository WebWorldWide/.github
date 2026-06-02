#!/usr/bin/env python3
"""
Generate the slim brand gradient divider used between README sections:
sky (#1e68f0) -> cherry (#ec5a2b) -> mint (#28b083), with soft rounded ends and
a transparent background. Outputs profile/assets/divider.png.

Requires: Pillow, numpy.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

W, H = 1000, 8
SS = 4
STOPS = [(0.0, (30, 104, 240)),    # #1e68f0 sky
         (0.5, (236, 90, 43)),     # #ec5a2b cherry
         (1.0, (40, 176, 131))]    # #28b083 mint


def lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def gradient(w):
    row = np.zeros((w, 3), dtype=np.uint8)
    for x in range(w):
        f = x / (w - 1)
        for k in range(len(STOPS) - 1):
            f0, c0 = STOPS[k]; f1, c1 = STOPS[k + 1]
            if f0 <= f <= f1:
                row[x] = lerp(c0, c1, (f - f0) / (f1 - f0))
                break
    return row


def main():
    w, h = W * SS, H * SS
    bar = np.tile(gradient(w)[None, :, :], (h, 1, 1))
    img = Image.fromarray(bar, "RGB").convert("RGBA")
    # rounded-capsule alpha mask
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=255)
    img.putalpha(mask)
    img = img.resize((W, H), Image.LANCZOS)

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "profile", "assets", "divider.png")
    img.save(out)
    print(f"wrote {out} ({os.path.getsize(out)} bytes, {W}x{H})")


if __name__ == "__main__":
    main()
