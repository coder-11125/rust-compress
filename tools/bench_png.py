#!/usr/bin/env python3
"""PNG size: rc vs oxipng (through the pyoxipng package) or any command, on any PNG files.

Lossless (default): each PNG is optimized by rc (`-l LEVEL`) and by the reference at one or more settings; sizes,
wall time and pixel-exactness are reported per file, then totals. The source PNGs are used as they are (not
re-encoded), so screenshots, 16-bit and alpha images behave as they do in real use.
--lossy:            palette quantization. Size at equal SSIMULACRA2 over a sweep of palette sizes, against a command
                    you supply (e.g. pngquant, which no pip package bundles): --ref-cmd 'pngquant {c} --force --output {out} {in}'

Setup:  python3 -m venv env && env/bin/pip install numpy pillow ssimulacra2 pyoxipng
        cargo build --release -p rust-compress
Usage:  bench_png.py WORKDIR --src PNGS [--ref-levels 2,4] [--level 9] [--lossy --ref-cmd ...]   (see --help)

Fairness: rc is run with --keep-hidden-pixels (the invisible colours under fully transparent pixels stay exact, as
in oxipng's default) and oxipng strips only non-essential chunks, so neither side gets credit for rewriting
invisible pixels or dropping metadata the other keeps. Pass --allow-hidden-rewrite to let rc rewrite them (the
exactness check then ignores fully transparent pixels).
"""
import argparse, os, sys, time
from functools import partial

import benchlib as B
from benchlib import np, Image


def decode(path):
    """Pixels as an array for exactness checks, or None for modes Pillow cannot compare directly (16-bit)."""
    im = Image.open(path)
    if im.mode.startswith("I;16") or im.mode == "I":
        return None
    return np.array(im.convert("RGBA"))


def same_pixels(a, b, ignore_hidden):
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return False
    if ignore_hidden:
        keep = a[..., 3] > 0
        return bool(np.array_equal(a[keep], b[keep]) and np.array_equal(a[..., 3], b[..., 3]))
    return bool(np.array_equal(a, b))


def oxipng_optimize(data, level):
    import oxipng
    kw = dict(level=level, strip=oxipng.StripChunks.safe())
    if level >= 7:  # "max": level 6 with zopfli, like `oxipng -o max -Z`
        kw.update(level=6, deflate=oxipng.Deflaters.zopfli(15))
    return oxipng.optimize_from_memory(data, **kw)


def lossless(a):
    rc = B.find_rc(a.rc)
    files = B.find_files(B.sources(a), {".png"})
    files = files[: a.limit] if a.limit > 0 else files
    if not files:
        sys.exit("no PNG files found")
    levels = [int(x) for x in a.ref_levels.split(",")]
    if a.ref_cmd:
        B.check_template(a.ref_cmd, "in", "out")
        levels = [0]
    tot_in = tot_rc = 0
    tot_ref = {l: 0 for l in levels}
    wins = {l: 0 for l in levels}
    bad = 0
    os.makedirs(a.work, exist_ok=True)
    for png in files:
        tag = os.path.splitext(os.path.basename(png))[0]
        out = os.path.join(a.work, "rc.png")
        cmd = [rc, "-f", "png", "-l", str(a.level), "--force", "--quiet", "-o", out, *a.rc_arg]
        if not a.allow_hidden_rewrite:
            cmd.append("--keep-hidden-pixels")
        t0 = time.perf_counter()
        B.run([*cmd, png])
        t_rc = time.perf_counter() - t0
        src = decode(png)
        ok = same_pixels(src, decode(out), a.allow_hidden_rewrite)
        size_in, r = os.path.getsize(png), os.path.getsize(out)
        tot_in, tot_rc = tot_in + size_in, tot_rc + r
        line = f"{tag.ljust(22)} in {size_in:>9}  rc -l{a.level} {r:>9} ({t_rc:5.2f}s)"
        if ok is False:
            bad += 1
            line += "  PIXELS DIFFER"
        for l in levels:
            t0 = time.perf_counter()
            if a.ref_cmd:
                ref_out = os.path.join(a.work, "ref.png")
                B.run(B.render(a.ref_cmd, **{"in": png, "out": ref_out, "l": l}))
                m = os.path.getsize(ref_out)
            else:
                m = len(oxipng_optimize(open(png, "rb").read(), l))
            t_ref = time.perf_counter() - t0
            tot_ref[l] += m
            wins[l] += r <= m
            line += f" | ref{'' if a.ref_cmd else ' o' + str(l)} {m:>9} ({t_ref:5.2f}s) {(r / m - 1) * 100:+5.1f}%"
        print(line, flush=True)
    print(f"TOTAL ({len(files)} files): input {tot_in} B, rc -l{a.level} {tot_rc} B ({(tot_rc / tot_in - 1) * 100:+.1f}% vs input)")
    for l in levels:
        label = a.ref_name or ("custom command" if a.ref_cmd else f"oxipng -o{l if l < 7 else ' max -Z'}")
        print(f"  vs {label}: {(tot_rc / tot_ref[l] - 1) * 100:+.1f}% total size; rc equal or smaller on {wins[l]} of {len(files)} files"
              "  (positive = rc larger)")
    print("pixel exactness:", "all verified exact" if not bad else f"{bad} FILE(S) DIFFER", "(16-bit files are not compared)")


def rc_lossy_cmd(rc, q, extra, src, out):
    return [rc, "-f", "png", "--lossy", "--colors", str(q), "--force", "--quiet", "-o", out, *extra, src]


def ref_lossy_encode(png, q, out, template):
    B.run(B.render(template, **{"in": png, "out": out, "c": q, "q": q}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    B.add_common(ap, qs="256,128,96,64,48,32,24", targets="70,80,90", size=0, limit=0)
    ap.add_argument("--lossy", action="store_true", help="compare palette quantization at equal SSIMULACRA2 (needs --ref-cmd)")
    ap.add_argument("--level", type=int, default=9, help="rc effort for lossless mode (default 9)")
    ap.add_argument("--ref-levels", default="2,4", help="oxipng levels to compare against; 7 means `-o max -Z` (default 2,4)")
    ap.add_argument("--allow-hidden-rewrite", action="store_true")
    a = ap.parse_args()
    if not a.lossy:
        return lossless(a)
    if not a.ref_cmd:
        sys.exit("--lossy needs --ref-cmd with {in}, {out} and {c} (the palette size), e.g. "
                 "--ref-cmd 'pngquant {c} --force --output {out} {in}'")
    B.check_template(a.ref_cmd, "in", "out")
    B.quality_bench(a, fmt="png lossy", ext=".png", rc=B.find_rc(a.rc), ref_name=a.ref_name or os.path.basename(a.ref_cmd.split()[0]),
                    ref_id=B.ref_id_for(a.ref_cmd), rc_cmd=rc_lossy_cmd,
                    ref_encode=partial(ref_lossy_encode, template=a.ref_cmd))


if __name__ == "__main__":
    main()
