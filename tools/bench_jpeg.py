#!/usr/bin/env python3
"""Lossy JPEG size at equal SSIMULACRA2: rc vs mozjpeg 4.0.3 (or any command), on any images.

Setup:  python3 -m venv env && env/bin/pip install numpy pillow ssimulacra2 jpeglib
        cargo build --release -p rust-compress
Usage:  bench_jpeg.py WORKDIR --src PHOTOS [--src MORE ...] [options]   (see --help)

The reference is mozjpeg through the `jpeglib` package. To compare against another encoder pass --ref-cmd with
{in} (a lossless PNG), {out} and {q} placeholders, e.g.  --ref-cmd 'magick {in} -quality {q} {out}'
Inputs are decoded and re-saved as lossless PNG first, so use pristine photographs. See benchlib.py for how the
comparison works.
"""
import argparse, os, sys
from functools import partial

import benchlib as B
from benchlib import np, Image


def rc_cmd(rc, q, extra, src, out, sub, keep_icc):
    cmd = [rc, "-f", "jpeg", "-q", str(q), "--force", "--quiet", "--subsampling", sub]
    if not keep_icc:
        cmd.append("--strip-icc")
    return [*cmd, "-o", out, *extra, src]


def ref_encode(png, q, out, template, sub):
    if template is None:
        import jpeglib
        jpeglib.version.set("mozjpeg403")
        im = jpeglib.from_spatial(np.array(Image.open(png).convert("RGB")))
        im.samp_factor = ((2, 2), (1, 1), (1, 1)) if sub == "420" else ((1, 1), (1, 1), (1, 1))
        im.write_spatial(out, qt=q)
    else:
        B.run(B.render(template, **{"in": png, "out": out, "q": q}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    B.add_common(ap, qs="25,35,45,55,65,75,85,92", targets="55,65,75,85")
    ap.add_argument("--subsampling", default="420", choices=["420", "444"])
    ap.add_argument("--keep-icc", action="store_true",
                    help="let rc keep ICC profiles (default strips them: most references write none)")
    a = ap.parse_args()
    if a.ref_cmd:
        B.check_template(a.ref_cmd, "in", "out", "q")
        name, rid = a.ref_name or os.path.basename(a.ref_cmd.split()[0]), B.ref_id_for(a.ref_cmd)
    else:
        try:
            import jpeglib  # noqa: F401
        except ImportError:
            sys.exit("the default reference needs `pip install jpeglib` (mozjpeg); or pass --ref-cmd")
        name, rid = a.ref_name or "mozjpeg 4.0.3", "mozjpeg-" + a.subsampling
    B.quality_bench(a, fmt="jpeg", ext=".jpg", rc=B.find_rc(a.rc), ref_name=name, ref_id=rid,
                    rc_cmd=partial(rc_cmd, sub=a.subsampling, keep_icc=a.keep_icc),
                    ref_encode=partial(ref_encode, template=a.ref_cmd, sub=a.subsampling))


if __name__ == "__main__":
    main()
