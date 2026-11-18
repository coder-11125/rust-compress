#!/usr/bin/env python3
"""Lossy GIF size at equal quality: rc vs gifsicle (or any command), on any animated GIFs.

For every GIF, rc and the reference are run over a range of lossiness settings. Quality is the mean
SSIMULACRA2 over a sample of frames, comparing the optimized GIF's frames with the frames of the
*input* GIF (both fully composited, so disposal and transparency are handled by Pillow). The size each
needs to reach each target score is interpolated (linear in score, log in size) and compared.
Positive percentages mean rc's file is larger. Works on Linux, macOS and Windows.

Setup:  python3 -m venv env && env/bin/pip install numpy pillow ssimulacra2 gifsicle-bin
        cargo build --release -p rust-compress

Usage:  bench_gif.py WORKDIR --src GIFS [--src MORE ...] [options]

    --src PATH         GIF file or directory (searched recursively); repeatable, or RC_BENCH_SRC
    --limit N          at most N files (default 12; 0 = all)
    --frames N         frames sampled per GIF for scoring (default 8; evenly spaced)
    --levels LIST      lossiness levels swept for rc, 0 = lossless (default 0,20,40,60,80,100,140,200)
    --ref-cmd CMD      reference command with {in}, {out} and {l} (its lossiness); default is gifsicle
                       -O3 --lossy={l} from `gifsicle` on PATH or next to this Python
    --ref-levels LIST  levels for the reference (default: --levels)
    --rc PATH          rc binary (default: target/release/rc[.exe] next to this repository, else on PATH)
    --rc-arg ARG       extra rc argument, repeatable
    --targets LIST     SSIMULACRA2 scores to compare at (default 60,70,80,90)
    --jobs N           files in parallel (default 4)
"""
import argparse, hashlib, json, os, shlex, shutil, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

import benchlib as B
from benchlib import np, Image, ssimulacra2, run, at_score
from PIL import ImageSequence


def frames_of(path, picks):
    """Composited RGB frames (as arrays) at the given indices; the index list is clipped to the GIF length."""
    out = {}
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        want = {min(i, n - 1) for i in picks}
        for i, fr in enumerate(ImageSequence.Iterator(im)):
            if i in want:
                out[i] = fr.convert("RGB").copy()
    return out


def score(orig_frames, path, scratch):
    """Mean SSIMULACRA2 of `path`'s sampled frames against the original GIF's same frames."""
    got = frames_of(path, list(orig_frames))
    vals = []
    for i, ref in orig_frames.items():
        a, b = os.path.join(scratch, "a.png"), os.path.join(scratch, "b.png")
        ref.save(a)
        got.get(i, got[max(got)]).save(b)
        vals.append(ssimulacra2.compute_ssimulacra2(a, b))
    return float(np.mean(vals))


def rc_cmd(rc, level, extra, src, out):
    cmd = [rc, "-f", "gif", "--force", "--quiet", "-o", out, *extra]
    if level > 0:
        cmd += ["--lossy", "--gif-lossy", str(level)]
    return [*cmd, src]


def ref_cmd(template, level, src, out):
    words = shlex.split(template, posix=(os.name != "nt"))
    return [w.replace("{in}", src).replace("{out}", out).replace("{l}", str(level)) for w in words]


def work_one(args):
    gif, o = args
    tag = os.path.splitext(os.path.basename(gif))[0]
    cache_p = os.path.join(o["work"], f"ref_{o['ref_id']}_{tag}.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    with Image.open(gif) as im:
        n = getattr(im, "n_frames", 1)
    picks = sorted({round(i * (n - 1) / max(1, o["frames"] - 1)) for i in range(o["frames"])}) if n > 1 else [0]
    orig = frames_of(gif, picks)
    base = os.path.getsize(gif)
    with tempfile.TemporaryDirectory(prefix="rcgif_") as scratch:
        rp, mp = [], []
        for l in o["levels"]:
            out = os.path.join(scratch, "rc.gif")
            run(rc_cmd(o["rc"], l, o["rc_args"], gif, out))
            rp.append((score(orig, out, scratch), os.path.getsize(out)))
        for l in o["ref_levels"]:
            if str(l) not in cache:
                out = os.path.join(scratch, "ref.gif")
                run(ref_cmd(o["ref"], l, gif, out))
                cache[str(l)] = (score(orig, out, scratch), os.path.getsize(out))
            mp.append(tuple(cache[str(l)]))
    json.dump(cache, open(cache_p, "w"))
    return tag, base, rp, mp, {t: (at_score(rp, t), at_score(mp, t)) for t in o["targets"]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--src", action="append", default=[])
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--levels", default="0,20,40,60,80,100,140,200")
    ap.add_argument("--ref-cmd")
    ap.add_argument("--ref-levels")
    ap.add_argument("--rc")
    ap.add_argument("--rc-arg", action="append", default=[])
    ap.add_argument("--targets", default="60,70,80,90")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="print every (score, size) point")
    a = ap.parse_args()

    srcs = B.sources(a)
    rc = B.find_rc(a.rc)
    template = a.ref_cmd
    name = "custom"
    if not template:
        g = shutil.which("gifsicle") or os.path.join(os.path.dirname(sys.executable), "gifsicle")
        if not os.path.exists(g):
            sys.exit("gifsicle not found; `pip install gifsicle-bin` or pass --ref-cmd")
        template, name = f"{shlex.quote(g)} -O3 --lossy={{l}} {{in}} -o {{out}}", "gifsicle -O3 --lossy"
    else:
        name = os.path.basename(shlex.split(template)[0])
    for ph in ("{in}", "{out}", "{l}"):
        if ph not in template:
            sys.exit(f"--ref-cmd must contain {ph}")
    files = B.find_files(srcs, {".gif"})
    files = files[: a.limit] if a.limit > 0 else files
    if not files:
        sys.exit("no GIFs found")
    levels = [int(x) for x in a.levels.split(",")]
    ref_levels = [int(x) for x in (a.ref_levels or a.levels).split(",")]
    targets = [float(x) for x in a.targets.split(",")]
    os.makedirs(a.work, exist_ok=True)
    opts = dict(rc=rc, rc_args=a.rc_arg, levels=levels, ref=template, ref_levels=ref_levels, targets=targets,
                frames=a.frames, work=a.work, ref_id=hashlib.sha1(template.encode()).hexdigest()[:8])
    res = {t: [] for t in targets}
    with ProcessPoolExecutor(a.jobs) as ex:
        for tag, base, rp, mp, d in ex.map(work_one, [(f, opts) for f in files]):
            line = f"{tag.ljust(16)} input {base:>7} B |"
            for t, (r, m) in d.items():
                if r and m:
                    res[t].append(r / m - 1)
                    line += f" s{t:.0f}:{(r / m - 1) * 100:+6.1f}%"
            print(line, flush=True)
            if a.verbose:
                print("   rc :", " ".join(f"{s:.1f}/{z}" for s, z in rp))
                print("   ref:", " ".join(f"{s:.1f}/{z}" for s, z in mp))
    print(f"MEAN rc vs {name}, size at equal mean SSIMULACRA2 (positive = rc larger):")
    for t in targets:
        if res[t]:
            print(f"  score {t:.0f}: mean {np.mean(res[t]) * 100:+.1f}%  worst {np.max(res[t]) * 100:+.1f}%  (n={len(res[t])})")


if __name__ == "__main__":
    main()
