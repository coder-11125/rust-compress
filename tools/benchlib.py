"""Shared plumbing for the per-format benchmarks (bench_png.py, bench_jpeg.py, bench_webp.py, bench_avif.py,
bench_gif.py). Not run directly. Works on Linux, macOS and Windows; needs numpy, pillow and ssimulacra2.

The quality benchmarks all work the same way: for each image, rc and a reference encoder are run over a sweep of
quality settings, every output is scored with SSIMULACRA2 against the lossless source, and the size each encoder
needs to reach each target score is interpolated (linear in score, log in size) and compared. Positive percentages
mean rc's file is larger.
"""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

try:
    import numpy as np
    from PIL import Image
    import ssimulacra2
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing Python package ({e.name}); install with: pip install numpy pillow ssimulacra2")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def find_files(paths, exts):
    """Files (or directories searched recursively) from `paths` with one of the extensions, sorted."""
    found = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                found += [os.path.join(root, n) for n in names if os.path.splitext(n)[1].lower() in exts]
        elif os.path.isfile(p):
            found.append(p)
        else:
            sys.exit(f"--src path not found: {p}")
    return sorted(set(found))


def sources(a):
    srcs = a.src or [p for p in os.environ.get("RC_BENCH_SRC", "").split(os.pathsep) if p]
    if not srcs:
        sys.exit("give at least one --src (file or directory), or set RC_BENCH_SRC")
    return srcs


def corpus(work, srcs, size, limit):
    """Decode the sources to lossless RGB PNGs under WORKDIR/corpus-<hash>; returns the PNG paths."""
    files = find_files(srcs, IMAGE_EXTS)
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


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(shlex.quote, cmd))}\n{r.stderr.decode(errors='replace')[:500]}")


def render(template, **values):
    """Split a command template and substitute {name} placeholders (each value as one argument word)."""
    words = shlex.split(template, posix=(os.name != "nt"))
    out = []
    for w in words:
        for k, v in values.items():
            w = w.replace("{" + k + "}", str(v))
        out.append(w)
    return out


def check_template(template, *placeholders):
    for ph in placeholders:
        if "{" + ph + "}" not in template:
            sys.exit(f"--ref-cmd must contain {{{ph}}}")


def find_rc(arg):
    if arg:
        return arg
    exe = "rc.exe" if os.name == "nt" else "rc"
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", exe)
    found = local if os.path.exists(local) else (shutil.which("rc") or local)
    if not (os.path.isfile(found) or shutil.which(found)):
        sys.exit(f"rc binary not found at {found}; run `cargo build --release -p rust-compress` or pass --rc")
    return found


def at_score(points, target):
    """Interpolate size at `target` score from (score, size) points: linear in score, log in size."""
    pts = sorted(points)
    for (s0, z0), (s1, z1) in zip(pts, pts[1:]):
        if s0 <= target <= s1:
            t = (target - s0) / (s1 - s0) if s1 > s0 else 0
            return float(np.exp(np.log(z0) + t * (np.log(z1) - np.log(z0))))
    return None


def score_file(ref_png, enc, scratch):
    """SSIMULACRA2 of an encoded file against the lossless reference, decoded through Pillow."""
    t = os.path.join(scratch, "dec.png")
    try:
        Image.open(enc).convert("RGB").save(t)
    except Exception as e:
        raise SystemExit(f"cannot decode {enc} with Pillow ({e}); is this format supported by your Pillow?")
    return ssimulacra2.compute_ssimulacra2(ref_png, t)


def add_common(ap, *, qs, targets, size=1280, limit=12):
    """Options every quality benchmark shares."""
    ap.add_argument("work", help="scratch directory (created if missing)")
    ap.add_argument("--src", action="append", default=[], metavar="PATH",
                    help="image file or directory (recursive); repeatable, or set RC_BENCH_SRC")
    ap.add_argument("--limit", type=int, default=limit, help=f"at most N images (default {limit}; 0 = all)")
    ap.add_argument("--size", type=int, default=size, help=f"downscale so the longer side is at most PX (default {size}; 0 = off)")
    ap.add_argument("--rc", help="rc binary (default: target/release/rc next to this repository, else on PATH)")
    ap.add_argument("--rc-arg", action="append", default=[], help="extra argument for rc, repeatable")
    ap.add_argument("--qs", default=qs, help=f"rc quality settings to sweep (default {qs})")
    ap.add_argument("--ref-qs", help="quality settings for the reference (default: --qs)")
    ap.add_argument("--targets", default=targets, help=f"SSIMULACRA2 scores to compare at (default {targets})")
    ap.add_argument("--ref-cmd", help="reference command with {in}, {out} and {q} placeholders (see the script's --help)")
    ap.add_argument("--ref-name", help="label for the reference in the report")
    ap.add_argument("--jobs", type=int, default=4, help="images in parallel (default 4)")


def _work_one(args):
    png, o = args
    tag = os.path.splitext(os.path.basename(png))[0]
    cache_p = os.path.join(o["cdir"], f"ref_{o['ref_id']}_{tag}.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    with tempfile.TemporaryDirectory(prefix="rcbench_") as scratch:
        rp, mp = [], []
        for q in o["qs"]:
            out = os.path.join(scratch, "rc" + o["ext"])
            run(o["rc_cmd"](o["rc"], q, o["rc_args"], png, out))
            rp.append((score_file(png, out, scratch), os.path.getsize(out)))
        for q in o["ref_qs"]:
            if str(q) not in cache:
                out = os.path.join(scratch, "ref" + o["ext"])
                o["ref_encode"](png, q, out)
                cache[str(q)] = (score_file(png, out, scratch), os.path.getsize(out))
            mp.append(tuple(cache[str(q)]))
    json.dump(cache, open(cache_p, "w"))
    return tag, {t: (at_score(rp, t), at_score(mp, t)) for t in o["targets"]}


def quality_bench(a, *, fmt, ext, rc_cmd, ref_encode, ref_name, ref_id, rc, size_default_note=""):
    """Run the equal-SSIMULACRA2 comparison. `rc_cmd(rc, q, extra_args, src, out)` builds rc's command line and
    `ref_encode(png, q, out)` runs the reference; both must be picklable (module-level functions)."""
    qs = [int(x) for x in a.qs.split(",")]
    ref_qs = [int(x) for x in (a.ref_qs or a.qs).split(",")]
    targets = [float(x) for x in a.targets.split(",")]
    os.makedirs(a.work, exist_ok=True)
    cdir, files = corpus(a.work, sources(a), a.size, a.limit)
    opts = dict(rc=rc, rc_args=a.rc_arg, qs=qs, ref_qs=ref_qs, targets=targets, cdir=cdir, ext=ext,
                rc_cmd=rc_cmd, ref_encode=ref_encode, ref_id=ref_id)
    res = {t: [] for t in targets}
    with ProcessPoolExecutor(a.jobs) as ex:
        for tag, d in ex.map(_work_one, [(f, opts) for f in files]):
            line = tag.ljust(24)
            for t, (r, m) in d.items():
                if r and m:
                    res[t].append(r / m - 1)
                    line += f" s{t:.0f}:{(r / m - 1) * 100:+5.1f}%"
            print(line, flush=True)
    print(f"MEAN rc ({fmt}) vs {ref_name} size at equal SSIMULACRA2 (positive = rc larger):")
    for t in targets:
        if res[t]:
            print(f"  score {t:.0f}: mean {np.mean(res[t]) * 100:+.1f}%  worst {np.max(res[t]) * 100:+.1f}%  (n={len(res[t])})")
    print("(a file counts in a row only if both encoders' quality sweeps reach that score)")


def ref_id_for(template):
    return hashlib.sha1(template.encode()).hexdigest()[:8]
