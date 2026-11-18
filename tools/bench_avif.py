#!/usr/bin/env python3
"""AVIF size at equal SSIMULACRA2: rc vs libavif (through Pillow) or any command, on any images.

Setup:  python3 -m venv env && env/bin/pip install numpy pillow ssimulacra2   # Pillow 11.3+ ships AVIF support
        cargo build --release -p rust-compress
Usage:  bench_avif.py WORKDIR --src PHOTOS [options]   (see --help)

The built-in reference is libavif's encoder as exposed by Pillow, with --ref-speed (0 slowest/best to 10 fastest,
default 6). To compare against `avifenc`, whose quality scale differs, pass --ref-cmd with {in}, {out} and {q}
and --ref-qs, e.g.  --ref-cmd 'avifenc -q {q} -s 6 {in} {out}' --ref-qs 20,30,40,50,60,70
Note that rc's AVIF encoder is much slower than libaom's fast presets; the script reports only size.
"""
import argparse, os
from functools import partial

import benchlib as B
from benchlib import Image


def rc_cmd(rc, q, extra, src, out, keep_icc):
    cmd = [rc, "-f", "avif", "-q", str(q), "--force", "--quiet"]
    if not keep_icc:
        cmd.append("--strip-icc")
    return [*cmd, "-o", out, *extra, src]


def ref_encode(png, q, out, template, speed):
    if template is None:
        Image.open(png).convert("RGB").save(out, "AVIF", quality=q, speed=speed)
    else:
        B.run(B.render(template, **{"in": png, "out": out, "q": q}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    B.add_common(ap, qs="25,35,45,55,65,75,85", targets="55,65,75,85", size=1024, limit=8)
    ap.add_argument("--ref-speed", type=int, default=6, help="libavif speed for the built-in reference (default 6)")
    ap.add_argument("--keep-icc", action="store_true")
    a = ap.parse_args()
    if a.ref_cmd:
        B.check_template(a.ref_cmd, "in", "out", "q")
    name = a.ref_name or (os.path.basename(a.ref_cmd.split()[0]) if a.ref_cmd else f"libavif speed {a.ref_speed} (Pillow)")
    rid = B.ref_id_for(a.ref_cmd) if a.ref_cmd else f"pil-avif-s{a.ref_speed}"
    B.quality_bench(a, fmt="avif", ext=".avif", rc=B.find_rc(a.rc), ref_name=name, ref_id=rid,
                    rc_cmd=partial(rc_cmd, keep_icc=a.keep_icc),
                    ref_encode=partial(ref_encode, template=a.ref_cmd, speed=a.ref_speed))


if __name__ == "__main__":
    main()
