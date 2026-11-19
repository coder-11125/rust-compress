#!/usr/bin/env python3
"""Generate SSIMULACRA2 verification pairs and their expected scores.

Writes pairs of small synthetic PNGs plus expected.csv (name,reference.png,distorted.png,score) into fixtures/ssim2.
Scores come from the pure-Python port py-ssimulacra2 (pip install ssimulacra2). Output is deterministic.

Run:  python tools/ssim2_vectors.py [--out fixtures/ssim2]
"""
import argparse
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage
import ssimulacra2


def plasma(h, w, rng):
    """Band-limited fractal texture, 3 channels in 0..1 (octave sizes scale with the image)."""
    m = max(h, w)
    octaves = ((m / 6, 1.0), (m / 20, 0.4), (2.0, 0.05 if m < 200 else 0.0))
    out = np.zeros((h, w, 3))
    for sigma, amp in octaves:
        n = ndimage.gaussian_filter(rng.standard_normal((h, w, 3)), (sigma, sigma, 0), mode="wrap")
        out += amp * n / (n.std() + 1e-9)
    out -= out.min()
    return out / out.max()


def base_image(w, h, seed, detail=True):
    """Gradient + plasma + shapes + text-like high-frequency edges."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    grad = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1), (xx + yy) / max(w + h - 2, 1)], axis=2)
    img = 0.35 + 0.35 * plasma(h, w, rng) + 0.3 * grad
    im = Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
    if detail:
        d = ImageDraw.Draw(im)
        for _ in range(min(12, max(2, w * h // 3000))):
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            x1, y1 = x0 + int(rng.integers(3, max(4, w // 4))), y0 + int(rng.integers(3, max(4, h // 4)))
            col = tuple(int(v) for v in rng.integers(0, 256, 3))
            (d.ellipse if rng.integers(0, 2) else d.rectangle)([x0, y0, x1, y1], fill=col)
        # text-like: short alternating 1px strokes
        for _ in range(max(1, h // 24)):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, max(1, w - 12)))
            for k in range(0, min(40, w - x), 2):
                hh = int(rng.integers(2, 7))
                d.line([x + k, y, x + k, y + hh], fill=(0, 0, 0) if rng.integers(0, 2) else (255, 255, 255))
    return im


def jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def webp(im, q):
    b = io.BytesIO()
    im.save(b, "WEBP", quality=q, method=4)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def noise(im, sigma, seed):
    a = np.asarray(im, dtype=np.float64)
    n = np.random.default_rng(seed).standard_normal(a.shape) * sigma
    return Image.fromarray(np.clip(a + n + 0.5, 0, 255).astype(np.uint8))


def posterize(im, bits):
    return ImageOps.posterize(im, bits)


def colour_shift(im):
    a = np.asarray(im, dtype=np.int32) + np.array([18, -10, 24])
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def brightness(im, d):
    a = np.asarray(im, dtype=np.int32) + d
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def sharpen(im):
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=300, threshold=0))


def translate(im):
    a = np.asarray(im)
    return Image.fromarray(np.roll(a, 1, axis=1))


def gray_rgb(im):
    return im.convert("L").convert("RGB")


def build():
    P = []  # (name, ref, dist)
    b = base_image(96, 72, 1)
    P.append(("identical_96x72", b, b.copy()))
    # sizes: each distorted with JPEG q50 (and a noise variant for tiny ones)
    sizes = [(8, 8), (8, 9), (9, 8), (15, 15), (16, 16), (17, 33), (31, 65), (64, 64), (100, 75), (129, 97),
             (256, 144), (320, 256), (513, 385)]
    for i, (w, h) in enumerate(sizes):
        r = base_image(w, h, 100 + i, detail=16 <= w < 100)
        if w >= 200:  # coarse levels keep the big PNGs small
            r = Image.fromarray(np.asarray(r) & 0xF0)
        # smooth big images keep the PNG small; small ones get visible noise
        if w >= 200:  # blurred then re-quantised, so the file stays small
            d = Image.fromarray(np.asarray(r.filter(ImageFilter.GaussianBlur(1.2))) & 0xF0)
        else:
            d = jpeg(r, 75) if w >= 64 else noise(r, 6, 200 + i)
        P.append((f"size_{w}x{h}", r, d))
    for q in (10, 30, 50, 70, 90):
        P.append((f"jpeg_q{q}", b, jpeg(b, q)))
    for q in (10, 50, 90):
        P.append((f"webp_q{q}", b, webp(b, q)))
    for r, tag in ((0.7, "0p7"), (1.5, "1p5"), (3, "3")):
        P.append((f"blur_r{tag}", b, b.filter(ImageFilter.GaussianBlur(r))))
    P.append(("noise_low", b, noise(b, 3, 7)))
    P.append(("noise_high", b, noise(b, 20, 8)))
    P.append(("posterize_4bit", b, posterize(b, 4)))
    P.append(("posterize_2bit", b, posterize(b, 2)))
    P.append(("colour_shift", b, colour_shift(b)))
    P.append(("brightness_plus20", b, brightness(b, 20)))
    P.append(("brightness_minus40", b, brightness(b, -40)))
    P.append(("sharpen_halos", b, sharpen(b)))
    P.append(("translate_1px", b, translate(b)))
    P.append(("different_image", b, base_image(96, 72, 999)))
    # grayscale-origin stored as RGB
    g = gray_rgb(b)
    P.append(("gray_identical", g, g.copy()))
    P.append(("gray_jpeg_q30", g, gray_rgb(jpeg(g, 30))))
    P.append(("gray_blur_r1p5", g, g.filter(ImageFilter.GaussianBlur(1.5))))
    P.append(("gray_noise_high", g, gray_rgb(noise(g, 20, 9))))
    P.append(("gray_vs_colour", g, b))
    # very dark / very bright
    dark = Image.fromarray((np.asarray(b, dtype=np.float64) * 0.08).astype(np.uint8))
    bright = Image.fromarray((255 - np.asarray(b, dtype=np.float64) * 0.08).astype(np.uint8))
    P.append(("dark_jpeg_q30", dark, jpeg(dark, 30)))
    P.append(("dark_noise_low", dark, noise(dark, 2, 10)))
    P.append(("bright_jpeg_q30", bright, jpeg(bright, 30)))
    P.append(("bright_noise_low", bright, noise(bright, 2, 11)))
    P.append(("black_vs_white", Image.new("RGB", (64, 64), (0, 0, 0)), Image.new("RGB", (64, 64), (255, 255, 255))))
    P.append(("flat_gray_vs_gray+1", Image.new("RGB", (64, 64), (128,) * 3), Image.new("RGB", (64, 64), (129,) * 3)))
    # extra distortion strengths on a second, high-detail base
    c = base_image(64, 64, 5)
    for q in (20, 60):
        P.append((f"detail_jpeg_q{q}", c, jpeg(c, q)))
    P.append(("detail_blur_r1", c, c.filter(ImageFilter.GaussianBlur(1))))
    P.append(("detail_noise_mid", c, noise(c, 10, 12)))
    P.append(("detail_webp_q30", c, webp(c, 30)))
    P.append(("detail_translate_1px", c, translate(c)))
    P.append(("detail_posterize_3bit", c, posterize(c, 3)))
    P.append(("detail_colour_shift", c, colour_shift(c)))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "fixtures", "ssim2"))
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):  # drop stale pairs from earlier runs
        if f.endswith(".png"):
            os.remove(os.path.join(out, f))
    rows = []
    ref_names = {}  # pairs sharing one reference object share one file
    for name, r, d in build():
        rp, dp = ref_names.setdefault(id(r), f"{name}.ref.png"), f"{name}.dist.png"
        for im, p in ((r, rp), (d, dp)):
            if not os.path.exists(os.path.join(out, p)):
                im.convert("RGB").save(os.path.join(out, p), "PNG", compress_level=9, optimize=True)
        score = ssimulacra2.compute_ssimulacra2(os.path.join(out, rp), os.path.join(out, dp))
        rows.append(f"{name},{rp},{dp},{float(score)!r}")
    with open(os.path.join(out, "expected.csv"), "w", newline="") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} pairs -> {out}")


if __name__ == "__main__":
    main()
