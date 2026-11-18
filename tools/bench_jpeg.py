#!/usr/bin/env python3
"""Size at equal SSIMULACRA2: rc vs any reference encoder, for lossy JPEG, WebP or AVIF.

For every image, rc and the reference encoder are run over a range of quality settings; the size each
needs to reach each target SSIMULACRA2 score is interpolated (linear in score, log in size) and the two
are compared. Positive percentages mean rc's file is larger. Works on Linux, macOS and Windows.

Setup (any Python 3 venv):
    python3 -m venv env && env/bin/pip install numpy pillow ssimulacra2     # + jpeglib for the default JPEG reference
    cargo build --release -p rust-compress

Usage:
    bench_jpeg.py WORKDIR --src PHOTOS [--src MORE ...] [options]

Inputs
    --src PATH         image file or directory (searched recursively); repeatable, or set RC_BENCH_SRC
                       (paths separated by os.pathsep: ':' on Unix, ';' on Windows)
    --limit N          use at most N images (default 12; 0 = all)
    --size PX          downscale so the longer side is at most PX (default 1280; 0 = keep full size)

What to compare
    --format F         jpeg (default) | webp | avif: the format rc writes (webp means lossy WebP)
    --subsampling S    jpeg only: 420 (default) | 444
    --rc PATH          rc binary (default: target/release/rc[.exe] next to this repository, else `rc` on PATH)
    --rc-arg ARG       extra argument for rc, repeatable (e.g. --rc-arg=--no-trellis)
    --keep-icc         let rc keep ICC profiles (default strips them: most reference encoders write none, and a
                       few hundred bytes of profile would skew small files)

Reference encoder (what rc is compared against)
    default            --format jpeg: mozjpeg 4.0.3 through the `jpeglib` Python package
    --ref-cmd CMD      any command line, run once per image and quality, with {in} (a lossless PNG), {out}
                       (the file to write) and {q} (the quality) substituted. Examples:
                         --ref-cmd 'cjpeg -quality {q} -outfile {out} {in}'          # needs a PPM-reading cjpeg
                         --ref-cmd 'cwebp -q {q} {in} -o {out}' --format webp
                         --ref-cmd 'avifenc -q {q} {in} {out}' --format avif --ref-qs 10,20,30,40,50,60
                         --ref-cmd 'magick {in} -quality {q} {out}'
    --ref-qs LIST      quality settings for the reference when its scale differs from rc's (default: --qs)
    --ref-name NAME    label in the report (default: the command's first word, or mozjpeg)

Other
    --qs LIST          rc's quality settings to sweep (default 25,35,45,55,65,75,85,92)
    --targets LIST     SSIMULACRA2 scores to compare at (default 55,65,75,85)
    --jobs N           images processed in parallel (default 4)

Inputs are decoded and re-saved as lossless PNG in WORKDIR before encoding, so use pristine photographs:
already heavily compressed JPEGs bring their artifacts into both encoders. Both encoders' outputs are decoded
with Pillow for scoring, so the format must be readable by your Pillow (AVIF needs Pillow 11.3+ or
pillow-avif-plugin). The reference side is cached in WORKDIR per command, so re-runs only redo rc.
"""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

try:
    import numpy as np
    from PIL import Image
    import ssimulacra2
except ImportError as e:
    sys.exit(f"missing Python package ({e.name}); install with: pip install numpy pillow ssimulacra2")

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
OUT_EXT = {"jpeg": ".jpg", "webp": ".webp", "avif": ".avif"}


def find_images(paths):
    found = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                found += [os.path.join(root, n) for n in names if os.path.splitext(n)[1].lower() in EXTS]
        elif os.path.isfile(p):
            found.append(p)
        else:
            sys.exit(f"--src path not found: {p}")
    return sorted(set(found))


def corpus(work, srcs, size, limit):
    """Decode the sources to lossless PNGs under WORKDIR/corpus-<hash>; returns (dir, png paths)."""
    files = find_images(srcs)
    if limit > 0:
        files = files[:limit]
    if not files:
        sys.exit("no images found in the given --src paths")
    key = hashlib.sha1(("\n".join(files) + f"|{size}").encode()).hexdigest()[:10]
    d = os.path.join(work, f"corpus-{key}")
    os.makedirs(d, exist_ok=True)
    outs, used = [], set()
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
        tag, n = stem, 1
        while tag in used:
            n += 1
            tag = f"{stem}-{n}"
        used.add(tag)
        out = os.path.join(d, tag + ".png")
        if not os.path.exists(out):
            im = Image.open(p).convert("RGB")
            if size > 0:
                im.thumbnail((size, size), Image.LANCZOS)
            im.save(out)  # no ICC profile is carried into the corpus
        outs.append(out)
    return d, outs


def score(ref, enc, scratch):
    """SSIMULACRA2 of an encoded file against the lossless reference, decoded through Pillow."""
    t = os.path.join(scratch, "dec.png")
    try:
        Image.open(enc).convert("RGB").save(t)
    except Exception as e:
        raise SystemExit(f"cannot decode {enc} with Pillow ({e}); is this format supported by your Pillow?")
    return ssimulacra2.compute_ssimulacra2(ref, t)


def rc_cmd(rc, fmt, q, sub, keep_icc, extra, src, out):
    cmd = [rc, "-f", fmt, "-q", str(q), "--force", "--quiet"]
    if fmt == "webp":
        cmd.append("--lossy")
    if fmt == "jpeg":
        cmd += ["--subsampling", sub]
    if not keep_icc:
        cmd.append("--strip-icc")
    return [*cmd, "-o", out, *extra, src]


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(shlex.quote, cmd))}\n{r.stderr.decode(errors='replace')[:500]}")


def ref_encode(spec, png, q, out):
    if spec["kind"] == "mozjpeg":
        import jpeglib
        jpeglib.version.set("mozjpeg403")
        a = np.array(Image.open(png).convert("RGB"))
        im = jpeglib.from_spatial(a)
        im.samp_factor = ((2, 2), (1, 1), (1, 1)) if spec["sub"] == "420" else ((1, 1), (1, 1), (1, 1))
        im.write_spatial(out, qt=q)
    else:
        words = shlex.split(spec["cmd"], posix=(os.name != "nt"))
        run([w.replace("{in}", png).replace("{out}", out).replace("{q}", str(q)) for w in words])


def at_score(points, target):
    """Interpolate size at `target` score from (score, size) points: linear in score, log in size."""
    pts = sorted(points)
    for (s0, z0), (s1, z1) in zip(pts, pts[1:]):
        if s0 <= target <= s1:
            t = (target - s0) / (s1 - s0) if s1 > s0 else 0
            return float(np.exp(np.log(z0) + t * (np.log(z1) - np.log(z0))))
    return None


def work_one(args):
    png, o = args
    tag = os.path.splitext(os.path.basename(png))[0]
    cache_p = os.path.join(o["cdir"], f"ref_{o['ref_id']}_{tag}.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    ext = OUT_EXT[o["fmt"]]
    with tempfile.TemporaryDirectory(prefix="rcbench_") as scratch:
        rp, mp = [], []
        for q in o["qs"]:
            out = os.path.join(scratch, "rc" + ext)
            run(rc_cmd(o["rc"], o["fmt"], q, o["sub"], o["keep_icc"], o["rc_args"], png, out))
            rp.append((score(png, out, scratch), os.path.getsize(out)))
        for q in o["ref_qs"]:
            if str(q) not in cache:
                out = os.path.join(scratch, "ref" + ext)
                ref_encode(o["ref"], png, q, out)
                cache[str(q)] = (score(png, out, scratch), os.path.getsize(out))
            mp.append(tuple(cache[str(q)]))
    json.dump(cache, open(cache_p, "w"))
    return tag, {t: (at_score(rp, t), at_score(mp, t)) for t in o["targets"]}


def find_rc(arg, here):
    if arg:
        return arg
    exe = "rc.exe" if os.name == "nt" else "rc"
    local = os.path.join(here, "..", "target", "release", exe)
    return local if os.path.exists(local) else (shutil.which("rc") or local)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", help="scratch directory (created if missing)")
    ap.add_argument("--src", action="append", default=[], metavar="PATH")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--format", default="jpeg", choices=sorted(OUT_EXT))
    ap.add_argument("--subsampling", default="420", choices=["420", "444"])
    ap.add_argument("--rc")
    ap.add_argument("--rc-arg", action="append", default=[])
    ap.add_argument("--keep-icc", action="store_true")
    ap.add_argument("--ref-cmd")
    ap.add_argument("--ref-qs")
    ap.add_argument("--ref-name")
    ap.add_argument("--qs", default="25,35,45,55,65,75,85,92")
    ap.add_argument("--targets", default="55,65,75,85")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    srcs = a.src or [p for p in os.environ.get("RC_BENCH_SRC", "").split(os.pathsep) if p]
    if not srcs:
        sys.exit("give at least one --src (image file or directory), or set RC_BENCH_SRC")
    rc = find_rc(a.rc, here)
    if not (os.path.isfile(rc) or shutil.which(rc)):
        sys.exit(f"rc binary not found at {rc}; run `cargo build --release -p rust-compress` or pass --rc")

    if a.ref_cmd:
        for ph in ("{in}", "{out}", "{q}"):
            if ph not in a.ref_cmd:
                sys.exit(f"--ref-cmd must contain {ph}")
        ref = {"kind": "cmd", "cmd": a.ref_cmd}
        name = a.ref_name or os.path.basename(shlex.split(a.ref_cmd)[0])
        ref_id = hashlib.sha1(a.ref_cmd.encode()).hexdigest()[:8]
    elif a.format == "jpeg":
        try:
            import jpeglib  # noqa: F401
        except ImportError:
            sys.exit("the default JPEG reference needs `pip install jpeglib` (mozjpeg); or pass --ref-cmd")
        ref = {"kind": "mozjpeg", "sub": a.subsampling}
        name = a.ref_name or "mozjpeg 4.0.3"
        ref_id = "mozjpeg-" + a.subsampling
    else:
        sys.exit(f"no built-in reference for --format {a.format}; pass --ref-cmd (see --help)")

    qs = [int(x) for x in a.qs.split(",")]
    ref_qs = [int(x) for x in (a.ref_qs or a.qs).split(",")]
    targets = [float(x) for x in a.targets.split(",")]
    os.makedirs(a.work, exist_ok=True)
    cdir, files = corpus(a.work, srcs, a.size, a.limit)
    opts = dict(rc=rc, fmt=a.format, sub=a.subsampling, keep_icc=a.keep_icc, rc_args=a.rc_arg, qs=qs, ref=ref,
                ref_qs=ref_qs, ref_id=ref_id, cdir=cdir, targets=targets)
    res = {t: [] for t in targets}
    with ProcessPoolExecutor(a.jobs) as ex:
        for tag, d in ex.map(work_one, [(f, opts) for f in files]):
            line = tag.ljust(24)
            for t, (r, m) in d.items():
                if r and m:
                    res[t].append(r / m - 1)
                    line += f" s{t:.0f}:{(r / m - 1) * 100:+5.1f}%"
            print(line, flush=True)
    print(f"MEAN rc ({a.format}) vs {name} size at equal SSIMULACRA2 (positive = rc larger):")
    for t in targets:
        if res[t]:
            print(f"  score {t:.0f}: mean {np.mean(res[t]) * 100:+.1f}%  worst {np.max(res[t]) * 100:+.1f}%  (n={len(res[t])})")
    print("(a file counts in a row only if both encoders' quality sweeps reach that score)")


if __name__ == "__main__":
    main()
