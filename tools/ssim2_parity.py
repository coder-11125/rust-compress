#!/usr/bin/env python3
"""Compare the Rust SSIMULACRA2 against the py-ssimulacra2 reference on real photos.

Each image in --src is downscaled so its long side is at most 1280, then paired with JPEG q30/60/90 and WebP
q30/60/90 versions made by Pillow. Every pair is scored by the Python reference and by --rust-cmd, a command
template with {ref} and {dist} placeholders that prints the score as the first number on stdout.

Run:  python tools/ssim2_parity.py --src DIR [--limit N] [--rust-cmd "target/release/examples/ssim2 {ref} {dist}"]
Needs: pip install numpy pillow scipy ssimulacra2
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VARIANTS = [("jpeg", 30), ("jpeg", 60), ("jpeg", 90), ("webp", 30), ("webp", 60), ("webp", 90)]
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def rust_score(template, ref, dist):
    cmd = [a.replace("{ref}", ref).replace("{dist}", dist) for a in shlex.split(template)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    m = NUM.search(out.stdout)
    if out.returncode != 0 or not m:
        raise RuntimeError(f"exit {out.returncode}: {(out.stderr or out.stdout).strip()[:200]}")
    return float(m.group(0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory of real images (read only)")
    ap.add_argument("--limit", type=int, default=0, help="use at most N images (0 = all)")
    ap.add_argument("--rust-cmd", default="target/release/examples/ssim2 {ref} {dist}",
                    help="command template with {ref} and {dist}; first number on stdout is the score")
    a = ap.parse_args()
    try:
        import ssimulacra2
    except ImportError:
        sys.exit("the reference needs `pip install ssimulacra2` (and numpy, scipy, pillow)")

    names = sorted(f for f in os.listdir(a.src) if os.path.splitext(f)[1].lower() in EXTS)
    if a.limit:
        names = names[: a.limit]
    if not names:
        sys.exit(f"no images found in {a.src}")

    exe = shlex.split(a.rust_cmd)[0]
    have_rust = shutil.which(exe) is not None
    if not have_rust:
        print(f"NOTE: Rust command not found ({exe}); printing reference scores only. Build the ssim2 example "
              f"or pass --rust-cmd.", file=sys.stderr)

    diffs = []
    print(f"{'pair':44} {'reference':>12} {'rust':>12} {'diff':>10}")
    with tempfile.TemporaryDirectory(prefix="ssim2_parity_") as tmp:
        for name in names:
            try:
                im = Image.open(os.path.join(a.src, name)).convert("RGB")
            except Exception as e:  # unreadable or unsupported format
                print(f"skip {name}: {e}", file=sys.stderr)
                continue
            s = 1280 / max(im.size)
            if s < 1:
                im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            ref = os.path.join(tmp, "ref.png")
            im.save(ref)
            for fmt, q in VARIANTS:
                dist = os.path.join(tmp, f"dist_{fmt}{q}.png")
                enc = os.path.join(tmp, "enc.jpg" if fmt == "jpeg" else "enc.webp")
                im.save(enc, "JPEG" if fmt == "jpeg" else "WEBP", quality=q)
                Image.open(enc).convert("RGB").save(dist)
                label = f"{name[:36]} {fmt}{q}"
                py = ssimulacra2.compute_ssimulacra2(ref, dist)
                if not have_rust:
                    print(f"{label:44} {py:12.6f} {'-':>12} {'-':>10}")
                    continue
                try:
                    rs = rust_score(a.rust_cmd, ref, dist)
                except Exception as e:
                    print(f"{label:44} {py:12.6f} {'ERROR':>12}  {e}")
                    continue
                diffs.append(abs(rs - py))
                print(f"{label:44} {py:12.6f} {rs:12.6f} {rs - py:+10.6f}")
    if diffs:
        print(f"\npairs {len(diffs)}  mean |diff| {sum(diffs) / len(diffs):.6f}  max |diff| {max(diffs):.6f}  "
              f">0.01: {sum(d > 0.01 for d in diffs)}  >0.1: {sum(d > 0.1 for d in diffs)}")
    elif have_rust:
        print("no pair scored by the Rust command")


if __name__ == "__main__":
    main()
