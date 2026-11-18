#!/usr/bin/env python3
"""WebP size: rc vs libwebp (through Pillow) or any command, on any images.

Lossy (default): size at equal SSIMULACRA2 over a quality sweep, against libwebp method 6.
--lossless:      size of rc's lossless WebP against libwebp lossless at method 6 / quality 100 (no quality
                 sweep: both are exact, so the comparison is bytes), with a pixel-exactness check on both.

Setup:  python3 -m venv env && env/bin/pip install numpy pillow ssimulacra2     # Pillow ships libwebp
        cargo build --release -p rust-compress
Usage:  bench_webp.py WORKDIR --src PHOTOS [--lossless] [options]   (see --help)

To compare against `cwebp` pass --ref-cmd with {in}, {out} and {q}, e.g.  --ref-cmd 'cwebp -q {q} -m 6 {in} -o {out}'
(lossy) or  --ref-cmd 'cwebp -lossless -z 9 {in} -o {out}'  (lossless; {q} is not needed there).
"""
import argparse, os, sys, time
from functools import partial

import benchlib as B
from benchlib import np, Image


def rc_cmd(rc, q, extra, src, out, keep_icc):
    cmd = [rc, "-f", "webp", "--lossy", "-q", str(q), "--force", "--quiet"]
    if not keep_icc:
        cmd.append("--strip-icc")
    return [*cmd, "-o", out, *extra, src]


def ref_encode(png, q, out, template, method):
    if template is None:
        Image.open(png).convert("RGB").save(out, "WEBP", quality=q, method=method)
    else:
        B.run(B.render(template, **{"in": png, "out": out, "q": q}))


def lossless(a):
    rc = B.find_rc(a.rc)
    _, files = B.corpus(a.work, B.sources(a), a.size, a.limit)
    tot_rc = tot_ref = 0
    diffs, speed = [], []
    for png in files:
        tag = os.path.splitext(os.path.basename(png))[0]
        out_rc, out_ref = os.path.join(a.work, "rc.webp"), os.path.join(a.work, "ref.webp")
        t0 = time.perf_counter()
        B.run([rc, "-f", "webp", "-l", str(a.level), "--force", "--quiet", "-o", out_rc, *a.rc_arg, png])
        t_rc = time.perf_counter() - t0
        t0 = time.perf_counter()
        if a.ref_cmd:
            B.run(B.render(a.ref_cmd, **{"in": png, "out": out_ref, "q": 100}))
        else:
            Image.open(png).convert("RGB").save(out_ref, "WEBP", lossless=True, quality=100, method=6)
        t_ref = time.perf_counter() - t0
        src = np.array(Image.open(png).convert("RGB"))
        ok = all(np.array_equal(src, np.array(Image.open(p).convert("RGB"))) for p in (out_rc, out_ref))
        r, m = os.path.getsize(out_rc), os.path.getsize(out_ref)
        tot_rc, tot_ref = tot_rc + r, tot_ref + m
        diffs.append(r / m - 1)
        speed.append(t_ref / t_rc)
        print(f"{tag.ljust(24)} rc {r:>9} B  ref {m:>9} B  {(r / m - 1) * 100:+6.1f}%  rc {t_rc:5.2f}s ref {t_ref:5.2f}s"
              f"  {'exact' if ok else 'PIXELS DIFFER'}", flush=True)
    print(f"TOTAL rc vs {a.ref_name or 'libwebp lossless m6'}: {(tot_rc / tot_ref - 1) * 100:+.1f}% size "
          f"(per file mean {np.mean(diffs) * 100:+.1f}%, worst {np.max(diffs) * 100:+.1f}%); "
          f"reference takes {np.mean(speed):.1f}x rc's wall time on average (positive size = rc larger)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    B.add_common(ap, qs="30,45,60,70,80,90", targets="55,65,75,85")
    ap.add_argument("--lossless", action="store_true", help="compare lossless sizes instead of equal quality")
    ap.add_argument("--level", type=int, default=7, help="rc effort for --lossless (default 7)")
    ap.add_argument("--ref-method", type=int, default=6, help="libwebp method for the built-in reference (0-6, default 6)")
    ap.add_argument("--keep-icc", action="store_true")
    a = ap.parse_args()
    if a.lossless:
        return lossless(a)
    if a.ref_cmd:
        B.check_template(a.ref_cmd, "in", "out", "q")
    name = a.ref_name or (os.path.basename(a.ref_cmd.split()[0]) if a.ref_cmd else f"libwebp m{a.ref_method} (Pillow)")
    rid = B.ref_id_for(a.ref_cmd) if a.ref_cmd else f"pil-webp-m{a.ref_method}"
    B.quality_bench(a, fmt="webp lossy", ext=".webp", rc=B.find_rc(a.rc), ref_name=name, ref_id=rid,
                    rc_cmd=partial(rc_cmd, keep_icc=a.keep_icc),
                    ref_encode=partial(ref_encode, template=a.ref_cmd, method=a.ref_method))


if __name__ == "__main__":
    main()
