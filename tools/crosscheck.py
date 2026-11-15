#!/usr/bin/env python3
"""Cross-check rust-compress against independent implementations.

The unit tests only prove that the encoders and decoders here agree with each other. This script feeds the same data
through programs written by other people and reports every disagreement:

  lossless  libjpeg-turbo's lossless JPEG encoder/decoder (imagecodecs) and the JPEG committee's reference libjpeg
            (pylibjpeg-libjpeg) for arithmetic-coded lossless files
  arith     arithmetic-coded JPEGs written by IJG libjpeg 7/8/9 and libjpeg-turbo (jpeglib), compared coefficient by
            coefficient after a lossless transcode
  hier      hierarchical JPEG: every mode of this crate's pyramid writer, decoded by the reference libjpeg
  resize    linear-light resizing against numpy + Pillow's float Lanczos
  anim      animated WebP and GIF against libwebp and Pillow on panning footage (size and PSNR)
  browser   headless Chrome (WebCodecs ImageDecoder) against Pillow, frame by frame
  jpeg422   4:2:2 JPEG: this crate's files read by libjpeg-turbo / Pillow / jpeglib (sampling factors, pixels), and
            libjpeg-turbo's 4:2:2, 4:2:0 and 4:4:4 files through this crate's decoder and its lossless transcode
  gzip      gzip, zlib and preset-dictionary streams against Python's zlib/gzip and the system gzip, both directions:
            every level/strategy/window, stored-block and segment boundaries, header fields, concatenated members,
            dictionaries larger than the window, and random corruption (needs `cargo build --release -p rc-deflate
            --example gzip_tool`, done for you)
  apng      APNG written with dirty rectangles and OVER/SOURCE blending, decoded by ffmpeg's own APNG decoder and by
            Pillow (libpng), every frame compared with the canvas it must reproduce (needs ffmpeg; Pillow-only without)

Setup (a virtualenv is advisable):
    pip install numpy pillow imagecodecs pylibjpeg-libjpeg jpeglib
    cargo build --release -p rust-compress

Run:  python tools/crosscheck.py [--rc target/release/rc] [--work DIR] [area ...]      (default: everything)
"""
import argparse
import base64
import collections
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"
PHOTOS = Path("/Library/Desktop Pictures")  # real photographs on macOS; any large JPEG will do elsewhere
results = []  # (area, name, ok, detail)


def report(area, name, ok, detail=""):
    results.append((area, name, ok, detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {area}: {name}{(' - ' + detail) if detail else ''}")


def photo(name="Flower 3.jpg", shrink=4):
    """A real photograph, box-downscaled, as an RGB uint8 array."""
    import imagecodecs

    path = PHOTOS / name
    if not path.exists():
        candidates = sorted(Path(sys.prefix).glob("**/*.jpg")) + sorted(FIX.glob("*.jpg"))
        path = candidates[0]
    a = imagecodecs.jpeg8_decode(path.read_bytes()).astype(np.uint32)
    h, w = (a.shape[0] // shrink) * shrink, (a.shape[1] // shrink) * shrink
    return a[:h, :w].reshape(h // shrink, shrink, w // shrink, shrink, 3).mean(axis=(1, 3)).round().astype(np.uint8)


def example(work, name, *args, pkg="rc-jpeg"):
    subprocess.run(["cargo", "run", "-q", "--release", "-p", pkg, "--example", name, "--", *map(str, args)], cwd=ROOT, check=True)


# ----------------------------------------------------------------------------------------------------------------------
def check_lossless(rc, work):
    import imagecodecs
    import libjpeg

    out = work / "lossless"
    out.mkdir()
    example(work, "make_lossless", out, *sorted(FIX.glob("lossless_*.jpg")))
    bad = collections.Counter()
    total = collections.Counter()
    for f in sorted(out.glob("*.jpg")):
        stem, variant = f.stem.rsplit(".", 1)
        ref = imagecodecs.jpeg8_decode((FIX / f"{stem}.jpg").read_bytes())
        want = ((ref >> 2) << 2) if variant == "pt2" else ref
        kind = "SOF11 (reference libjpeg)" if variant.startswith("arith") else "SOF3 (libjpeg-turbo)"
        total[kind] += 1
        try:
            got = libjpeg.decode(f.read_bytes(), reshape=True) if variant.startswith("arith") else imagecodecs.jpeg8_decode(f.read_bytes())
            if got.shape != want.shape or not np.array_equal(got.astype(np.int64), want.astype(np.int64)):
                bad[kind] += 1
        except Exception:
            bad[kind] += 1
    for kind, n in total.items():
        report("lossless", f"{n} recoded files decode to the original samples in {kind}", bad[kind] == 0, f"{bad[kind]} wrong" if bad[kind] else "")


# ----------------------------------------------------------------------------------------------------------------------
def check_arith(rc, work):
    import jpeglib

    out = work / "arith"
    out.mkdir()
    img = photo()
    crops = {"rgb420": img[:211, :333], "rgb444": img[:211, :333], "gray": np.ascontiguousarray(img[:187, :251, 1:2])}

    def coefs(p):
        d = jpeglib.read_dct(str(p))
        return [d.Y] + ([d.Cb, d.Cr] if d.has_chrominance else [])

    n = bad = 0
    for ver in ("7", "8", "9", "turbo210"):
        jpeglib.version.set(ver)
        for kind, arr in crops.items():
            for q in (30, 75, 95):
                for prog in (False, True):
                    p = out / f"{ver}_{kind}_{q}_{'prog' if prog else 'seq'}.jpg"
                    im = jpeglib.from_spatial(np.ascontiguousarray(arr))
                    if kind != "gray":
                        im.samp_factor = ((2, 2), (1, 1), (1, 1)) if kind == "rgb420" else ((1, 1), (1, 1), (1, 1))
                    im.write_spatial(str(p), qt=q, flags=["+ARITH_CODE"] + (["+PROGRESSIVE_MODE"] if prog else []))
                    t = p.with_suffix(".huff.jpg")
                    # --subsampling 444 keeps 4:4:4 as it is (below q90 the tool otherwise converts it to 4:2:0).
                    sub = ["--subsampling", {"rgb444": "444", "rgb420": "420"}.get(kind, "auto")]
                    subprocess.run([rc, str(p), "--arithmetic", "off", "-q", "100", *sub, "-o", str(t), "--force"], check=True, capture_output=True)
                    a, b = coefs(p), coefs(t)
                    n += 1
                    bad += not (len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)))
    report("arith", f"{n} arithmetic JPEGs from IJG libjpeg 7/8/9 + libjpeg-turbo 2.1 transcode to identical coefficients", bad == 0, f"{bad} differ" if bad else "")


# ----------------------------------------------------------------------------------------------------------------------
def check_hier(rc, work):
    import libjpeg

    out = work / "hier"
    out.mkdir()
    example(work, "hier_matrix", out)
    exact_ok = exact_n = lossy_n = 0
    skipped16 = 0
    worst_ll = 0
    for line in (out / "manifest.txt").read_text().splitlines():
        name, w, h, ch, depth, kind = line.split()
        w, h, ch, depth = int(w), int(h), int(ch), int(depth)
        if depth == 16:
            skipped16 += 1  # the reference rejects precision 16 inside a DHP
            continue
        src = np.frombuffer((out / f"{name}.raw").read_bytes(), np.uint8).reshape((h, w, ch) if ch > 1 else (h, w))
        got = libjpeg.decode((out / f"{name}.jpg").read_bytes(), colour_transform=0, reshape=True)
        if kind == "exact" and name.startswith("ll_") or name.startswith("shape_") and name.endswith("lossless"):
            exact_n += 1
            exact_ok += got.shape == src.shape and np.array_equal(got.astype(int), src.astype(int))
        elif kind == "exact":  # DCT frames then a lossless one: only the IDCT rounding of the DCT frames can differ
            exact_n += 1
            d = np.abs(got.astype(int) - src.astype(int)).max() if got.shape == src.shape else 99
            worst_ll = max(worst_ll, d)
            exact_ok += d <= 2
        else:
            lossy_n += 1
    report("hier", f"{exact_ok}/{exact_n} lossless-ended pyramids reproduce the source in the reference libjpeg (DCT ones within {worst_ll} levels)", exact_ok == exact_n)
    report("hier", f"{lossy_n} lossy DCT pyramids decoded by the reference, {skipped16} 16-bit ones skipped (it rejects them)", True)


# ----------------------------------------------------------------------------------------------------------------------
def check_resize(rc, work):
    import imagecodecs
    from PIL import Image

    def lin(v):
        v = v / 255.0
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    def enc(l):
        l = np.clip(l, 0, 1)
        return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055) * 255.0

    def lanczos(a, size):
        return np.stack([np.asarray(Image.fromarray(a[..., c].astype(np.float32), mode="F").resize(size, Image.LANCZOS)) for c in range(a.shape[-1])], -1)

    img = photo("Mojave Day.jpg", 8)
    Image.fromarray(img).save(work / "r_in.png")
    worst = 0.0
    for spec, size in (("50%", (img.shape[1] // 2, img.shape[0] // 2)), ("200x", (200, int(np.floor(img.shape[0] * 200 / img.shape[1] + 0.5))))):
        subprocess.run([rc, str(work / "r_in.png"), "--resize", spec, "-f", "png", "-o", str(work / "r_out.png"), "--force"], check=True, capture_output=True)
        got = imagecodecs.png_decode((work / "r_out.png").read_bytes()).astype(float)
        ref = np.rint(enc(lanczos(lin(img.astype(float)), size)))
        worst = max(worst, float(np.abs(got - ref).mean()))
    report("resize", f"linear-light resize matches numpy + Pillow float Lanczos (mean abs diff {worst:.3f} levels)", worst < 0.1)


# ----------------------------------------------------------------------------------------------------------------------
def check_anim(rc, work):
    from PIL import Image, ImageSequence

    big = Image.fromarray(photo("Mojave Day.jpg", 4))
    frames = [big.crop((200 + i * 6, 150 + int(8 * np.sin(i / 2)), 520 + i * 6, 350 + int(8 * np.sin(i / 2)))) for i in range(12)]
    frames[0].save(work / "pan.png", save_all=True, append_images=frames[1:], duration=60, loop=0)
    src = np.stack([np.asarray(f) for f in frames]).astype(float)

    def psnr(p):
        out = np.stack([np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(Image.open(p))]).astype(float)
        return 10 * np.log10(255 * 255 / ((out - src) ** 2).mean())

    wins = 0
    for q in (40, 75, 90):
        mine = work / f"o{q}.webp"
        theirs = work / f"r{q}.webp"
        subprocess.run([rc, str(work / "pan.png"), "-f", "webp", "--lossy", "-q", str(q), "-o", str(mine), "--force"], check=True, capture_output=True)
        frames[0].save(theirs, save_all=True, append_images=frames[1:], duration=60, loop=0, quality=q, method=4)
        a, b = (mine.stat().st_size, psnr(mine)), (theirs.stat().st_size, psnr(theirs))
        wins += a[0] <= b[0] and a[1] >= b[1]
        print(f"      q{q}: this tool {a[0]:>7,} B {a[1]:.2f} dB   libwebp {b[0]:>7,} B {b[1]:.2f} dB")
    report("anim", f"lossy animated WebP is no larger and no worse than libwebp at {wins}/3 settings", wins == 3)


# ----------------------------------------------------------------------------------------------------------------------
def check_browser(rc, work):
    from PIL import Image, ImageSequence

    chrome = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not os.path.exists(chrome):
        print("  [skip] browser: set CHROME to a Chrome/Chromium binary")
        return
    w = work / "web"
    w.mkdir()
    fr = []
    from PIL import ImageDraw

    for t in range(8):
        im = Image.new("RGB", (96, 64), (10, 40, 200))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 48, 32], fill=(240, 240, 60))
        d.rectangle([4 + 8 * t, 10, 20 + 8 * t, 26], fill=(250, 20, 20))
        fr.append(im)
    fr[0].save(w / "src.gif", save_all=True, append_images=fr[1:], duration=[60 + 10 * t for t in range(8)], loop=0)
    sp = []
    for t in range(8):
        im = Image.new("RGBA", (96, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse([6 + 9 * t, 8 + t, 30 + 9 * t, 32 + t], fill=(30 * t % 256, 200, 90, 255))
        sp.append(im)
    sp[0].save(w / "sprite.png", save_all=True, append_images=sp[1:], duration=80, loop=0)
    for args, out in (
        (["src.gif"], "re.gif"), (["src.gif", "-f", "webp"], "a.webp"), (["src.gif", "-f", "avif", "-q", "85"], "a.avif"),
        (["sprite.png", "-f", "gif"], "sprite.gif"), (["sprite.png", "-f", "webp"], "sprite.webp"),
    ):
        subprocess.run([rc, *[str(w / a) if a.endswith((".gif", ".png")) else a for a in args], "-o", str(w / out), "--force"], check=True, capture_output=True)
    types = {"gif": "image/gif", "webp": "image/webp", "avif": "image/avif"}
    files = [f for f in sorted(os.listdir(w)) if f.rsplit(".", 1)[-1] in types]
    (w / "test.html").write_text(
        """<!doctype html><script>
const files=%s; const b64=u=>{let s='';for(let i=0;i<u.length;i+=0x8000)s+=String.fromCharCode.apply(null,u.subarray(i,i+0x8000));return btoa(s)};
(async()=>{const res={};for(const [name,type] of files){try{const buf=await (await fetch(name)).arrayBuffer();
const dec=new ImageDecoder({data:buf,type,colorSpaceConversion:'none'});await dec.tracks.ready;await dec.completed;
const n=dec.tracks.selectedTrack.frameCount,frames=[];for(let i=0;i<n;i++){const {image}=await dec.decode({frameIndex:i});
const c=new OffscreenCanvas(image.displayWidth,image.displayHeight),g=c.getContext('2d',{colorSpace:'srgb'});g.drawImage(image,0,0);
const d=g.getImageData(0,0,c.width,c.height);frames.push({w:c.width,h:c.height,dur:image.duration,rgba:b64(new Uint8Array(d.data.buffer))});image.close();}
res[name]={ok:true,frames};}catch(e){res[name]={ok:false,error:String(e)};}}
await fetch('/result',{method:'POST',body:JSON.stringify(res)});})();</script>""" % json.dumps([[f, types[f.rsplit(".", 1)[-1]]] for f in files])
    )
    got = {}

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            got["body"] = self.rfile.read(int(self.headers["Content-Length"])).decode()
            self.send_response(200)
            self.end_headers()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(w)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={work / 'chrome'}", f"http://127.0.0.1:{srv.server_address[1]}/test.html"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while "body" not in got and time.time() - t0 < 180:
        time.sleep(0.5)
    proc.terminate()
    srv.shutdown()
    if "body" not in got:
        report("browser", "Chrome did not report back", False)
        return
    res = json.loads(got["body"])
    for name in files:
        r = res[name]
        if not r["ok"]:
            report("browser", name, False, r["error"][:60])
            continue
        ref = [np.asarray(f.convert("RGBA")).astype(int) for f in ImageSequence.Iterator(Image.open(w / name))]
        worst = 0
        for pa, f in zip(ref, r["frames"]):
            ba = np.frombuffer(base64.b64decode(f["rgba"]), np.uint8).reshape(f["h"], f["w"], 4).astype(int)
            vis = (ba[..., 3] > 0) & (pa[..., 3] > 0)
            worst = max(worst, int(np.abs(ba[..., :3] - pa[..., :3])[vis].max()) if vis.any() else 0)
        report("browser", f"{name}: {len(r['frames'])} frames in Chrome vs Pillow", len(ref) == len(r["frames"]) and worst <= 1, f"worst visible diff {worst}")


def check_jpeg422(rc, work):
    import imagecodecs
    import jpeglib
    from PIL import Image, JpegImagePlugin

    rng = np.random.default_rng(1)
    imgs = {}
    shot = photo(shrink=8) if PHOTOS.exists() else None
    if shot is not None:
        imgs["photo"] = shot
    xs, ys = np.linspace(0, 255, 331)[None, :, None] * np.ones((1, 1, 3)), np.linspace(0, 255, 177)[:, None, None]
    imgs["gradient_331x177"] = np.clip(xs * 0.7 + ys * 0.3 + rng.normal(0, 3, (177, 331, 3)), 0, 255).astype(np.uint8)
    imgs["noise_17x23"] = rng.integers(0, 255, (23, 17, 3), dtype=np.uint8)
    imgs["noise_9x8"] = rng.integers(0, 255, (8, 9, 3), dtype=np.uint8)
    sf = lambda path: [tuple(int(v) for v in r) for r in jpeglib.read_dct(str(path)).samp_factor]
    for name, arr in imgs.items():
        png = work / f"{name}.png"
        Image.fromarray(arr).save(png)
        for q in (60, 90):
            out = work / f"{name}_q{q}.jpg"
            subprocess.run([rc, str(png), "-f", "jpeg", "--subsampling", "422", "-q", str(q), "--force", "-o", str(out)], check=True, capture_output=True)
            data = out.read_bytes()
            lj, pil = imagecodecs.jpeg8_decode(data), np.asarray(Image.open(out).convert("RGB"))
            ok = JpegImagePlugin.get_sampling(Image.open(out)) == 1 and sf(out) == [(1, 2), (1, 1), (1, 1)] and lj.shape == arr.shape and np.array_equal(lj, pil)
            # a 4:2:2 file from libjpeg-turbo's own encoder at the same quality is the yardstick for fidelity
            ref = imagecodecs.jpeg8_decode(imagecodecs.jpeg8_encode(arr, level=q, subsampling="422"))
            mse = lambda a: ((a.astype(float) - arr) ** 2).mean()
            db = lambda a: 99 if mse(a) == 0 else 10 * np.log10(255 * 255 / mse(a))
            report("jpeg422", f"{name} q{q}: sampling and pixels in libjpeg-turbo / Pillow / jpeglib", ok and db(lj) > db(ref) - 3, f"{db(lj):.1f} dB vs libjpeg's own {db(ref):.1f} dB")
    for name, arr in imgs.items():
        for sub in ("422", "420", "444"):
            f = work / f"lj_{name}_{sub}.jpg"
            f.write_bytes(imagecodecs.jpeg8_encode(arr, level=85, subsampling=sub))
            want = imagecodecs.jpeg8_decode(f.read_bytes())
            back = work / f"ours_{name}_{sub}.png"
            subprocess.run([rc, str(f), "-f", "png", "--force", "-o", str(back)], check=True, capture_output=True)
            worst = int(np.abs(np.asarray(Image.open(back).convert("RGB")).astype(int) - want.astype(int)).max())
            report("jpeg422", f"libjpeg {sub} {name}: this crate's decoder vs libjpeg", worst <= 3, f"worst difference {worst}")
            if sub != "444":  # 4:4:4 is deliberately resampled to 4:2:0 below q90
                tr = work / f"tr_{name}_{sub}.jpg"
                subprocess.run([rc, str(f), "-q", "100", "--force", "-o", str(tr)], check=True, capture_output=True)
                kept = tr.exists() and np.array_equal(imagecodecs.jpeg8_decode(tr.read_bytes()), want) and sf(tr) == sf(f)
                report("jpeg422", f"libjpeg {sub} {name}: lossless transcode is pixel-identical", kept)
    # explicit resampling of a 4:2:0 and a 4:4:4 source to 4:2:2
    for sub in ("420", "444"):
        src = work / f"lj_gradient_331x177_{sub}.jpg"
        out = work / f"to422_{sub}.jpg"
        subprocess.run([rc, str(src), "--subsampling", "422", "-q", "85", "--force", "-o", str(out)], check=True, capture_output=True)
        report("jpeg422", f"{sub} source re-sampled to 4:2:2", sf(out) == [(1, 2), (1, 1), (1, 1)] and imagecodecs.jpeg8_decode(out.read_bytes()).shape == imgs["gradient_331x177"].shape)


def check_gzip(rc, work):
    import gzip
    import random
    import struct
    import zlib

    subprocess.run(["cargo", "build", "-q", "--release", "-p", "rc-deflate", "--example", "gzip_tool"], cwd=ROOT, check=True)
    tool = str(ROOT / "target" / "release" / "examples" / "gzip_tool")
    rnd = random.Random(11)
    fails = collections.Counter()
    total = collections.Counter()

    def run(*a):
        return subprocess.run([tool, *map(str, a)], capture_output=True, text=True)

    def ours_c(kind, level, data, dic=None):
        (work / "in").write_bytes(data)
        args = ["compress", kind, level, work / "in", work / "out"]
        if dic is not None:
            (work / "dict").write_bytes(dic)
            args.append(work / "dict")
        assert run(*args).returncode == 0
        return (work / "out").read_bytes()

    def ours_d(kind, blob, dic=None):
        (work / "in").write_bytes(blob)
        args = ["decompress", kind, work / "in", work / "out"]
        if dic is not None:
            (work / "dict").write_bytes(dic)
            args.append(work / "dict")
        r = run(*args)
        return ((work / "out").read_bytes() if r.returncode == 0 else None), r.stderr.strip()

    def check(group, ok, what):
        total[group] += 1
        if not ok:
            fails[group] += 1
            print(f"    FAIL {group}: {what}")

    def mixed(n):
        out = bytearray()
        while len(out) < n:
            c = rnd.random()
            if c < 0.3:
                out += bytes(rnd.randrange(256) for _ in range(rnd.randrange(50, 400)))
            elif c < 0.6 and len(out) > 100:
                s = rnd.randrange(len(out) - 50)
                out += out[s : s + rnd.randrange(20, 300)]
            else:
                out += bytes([rnd.randrange(256)]) * rnd.randrange(5, 200)
        return bytes(out[:n])

    sets = {"empty": b"", "one": b"x", "s65535": mixed(65535), "s65536": mixed(65536), "s65537": mixed(65537),
            "text": b"The quick brown fox jumps over the lazy dog. " * 5000, "mixed700k": mixed(700_000),
            "mixed1.6M": mixed(1_600_000), "random300k": os.urandom(300_000), "zeros2M": bytes(2_000_000)}
    for name, data in sets.items():
        for level in (0, 1, 6, 9):
            g, z, r = ours_c("gzip", level, data), ours_c("zlib", level, data), ours_c("raw", level, data)
            check("ours -> python", gzip.decompress(g) == data and zlib.decompress(z) == data and zlib.decompress(r, -15) == data, f"{name} L{level}")
            if level in (1, 9):
                (work / "o.gz").write_bytes(g)
                p = subprocess.run(["gzip", "-dc", str(work / "o.gz")], capture_output=True)
                check("ours -> system gzip", p.returncode == 0 and p.stdout == data, f"{name} L{level}")
    for name in ("text", "mixed700k", "s65537"):
        data = sets[name]
        dic = mixed(40_000)[-20_000:] + data[:3000]
        for level in (1, 6, 9):
            o = zlib.decompressobj(zdict=dic)
            check("ours -> python dictionary", o.decompress(ours_c("zlibdict", level, data, dic)) == data and o.eof, f"{name} L{level}")
    for name, data in sets.items():
        for level in range(10):
            for strat in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED):
                if (level not in (0, 1, 6, 9) or len(data) > 700_000) and strat != zlib.Z_DEFAULT_STRATEGY:
                    continue
                for wbits, kind in ((31, "gzip"), (15, "zlib"), (-15, "raw")):
                    c = zlib.compressobj(level, zlib.DEFLATED, wbits, 9 if level else 8, strat)
                    check("python -> ours", ours_d(kind, c.compress(data) + c.flush())[0] == data, f"{kind} {name} L{level} strategy {strat}")
        for wb in (9, 12):
            c = zlib.compressobj(6, zlib.DEFLATED, wb, 1)
            check("python -> ours", ours_d("zlib", c.compress(data) + c.flush())[0] == data, f"window bits {wb} {name}")
    for name in ("text", "mixed700k", "s65537", "empty"):
        data = sets[name]
        dic = mixed(30_000) + data[:2000]
        for level in (1, 6, 9):
            c = zlib.compressobj(level, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY, dic)
            blob = c.compress(data) + c.flush()
            check("python -> ours dictionary", ours_d("zlibdict", blob, dic)[0] == data, f"{name} L{level}")
            check("wrong dictionary rejected", ours_d("zlibdict", blob, dic[:-1])[0] is None, name)
    data = sets["mixed700k"]
    big = mixed(200_000)  # larger than the 32 KiB window
    for level in (1, 9):
        c = zlib.compressobj(level, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY, big)
        check("python -> ours dictionary", ours_d("zlibdict", c.compress(data) + c.flush(), big)[0] == data, f"200 KB dictionary L{level}")
        check("ours -> python dictionary", zlib.decompressobj(zdict=big).decompress(ours_c("zlibdict", level, data, big)) == data, f"200 KB dictionary L{level}")
    for name in ("text", "mixed700k", "empty"):
        data = sets[name]
        (work / "f.bin").write_bytes(data)
        for opt in ("-1", "-6", "-9", "-N -9", "-n"):
            blob = subprocess.run(f"gzip {opt} -c {work / 'f.bin'}", shell=True, capture_output=True).stdout
            check("system gzip -> ours", ours_d("gzip", blob)[0] == data, f"gzip {opt} {name}")
        one = subprocess.run(f"gzip -c {work / 'f.bin'}", shell=True, capture_output=True).stdout
        check("concatenated members", ours_d("gzip", one * 3)[0] == data * 3, name)
    data = sets["text"][:50_000]
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    body = co.compress(data) + co.flush()
    trailer = struct.pack("<II", zlib.crc32(data), len(data))
    for flags in range(32):
        hdr = bytes([0x1F, 0x8B, 8, flags, 1, 2, 3, 4, 2, 3])
        if flags & 4:
            hdr += struct.pack("<H", 7) + b"Ap\x04\x00abc"
        if flags & 8:
            hdr += b"name.txt\0"
        if flags & 16:
            hdr += b"a comment\0"
        if flags & 2:
            hdr += struct.pack("<H", zlib.crc32(hdr) & 0xFFFF)
        blob = hdr + body + trailer
        check("header fields", ours_d("gzip", blob)[0] == data and gzip.decompress(blob) == data, f"FLG={flags:05b}")
    g = gzip.compress(data)

    def py(b):
        try:
            return gzip.decompress(b)
        except Exception:
            return None

    damaged = {"trailing junk": g + b"junk!", "truncated": g[:-3], "bad crc": g[:-8] + bytes([g[-8] ^ 1]) + g[-7:],
               "bad size": g[:-1] + bytes([g[-1] ^ 1]), "bad magic": b"\x1f\x8c" + g[2:], "bad method": g[:2] + b"\x09" + g[3:]}
    for k, b in damaged.items():
        check("damaged streams rejected", ours_d("gzip", b)[0] is None, k)
    check("zero padding accepted", ours_d("gzip", g + bytes(7))[0] == data, "")
    # RFC 1952 makes reserved flag bits an error; Python's gzip ignores them, so only ours is held to the RFC.
    check("reserved flag rejected", ours_d("gzip", g[:3] + b"\x20" + g[4:])[0] is None, "")
    for i in range(400):
        b = bytearray(g)
        for _ in range(rnd.randrange(1, 4)):
            b[rnd.randrange(len(b))] = rnd.randrange(256)
        if i % 9 == 0:
            b = b[: rnd.randrange(len(b))]
        out, p = ours_d("gzip", bytes(b))[0], py(bytes(b))
        check("random corruption", out is None or out == data or (p is not None and out == p), f"mutation {i} accepted wrong data")
    for group in total:
        report("gzip", group, fails[group] == 0, f"{total[group] - fails[group]} of {total[group]} cases")


def check_apng(rc, work):
    from PIL import Image, ImageSequence

    example(work, "apng_cases", work, pkg="rc-png")
    have_ffmpeg = shutil.which("ffmpeg") is not None
    if not have_ffmpeg:
        print("  [skip] apng: ffmpeg not found, checking with Pillow only")
    for png in sorted(work.glob("*.png")):
        name = png.stem
        w, h, nf = map(int, (work / f"{name}.info").read_text().split())
        want = np.frombuffer((work / f"{name}.rgba").read_bytes(), np.uint8).reshape(nf, h, w, 4).astype(int)
        exact = name.endswith("_keephidden")  # hidden colours of transparent pixels must survive too

        def same(got):
            if got.shape != want.shape:
                return f"{got.shape[0]} frames of {got.shape[1:]} vs {want.shape[0]} of {want.shape[1:]}"
            diff = (got != want).any(axis=-1)
            if not exact:
                diff &= ~((got[..., 3] == 0) & (want[..., 3] == 0))
            if diff.any():
                k, y, x = np.argwhere(diff)[0]
                return f"frame {k} pixel {x},{y}: want {want[k, y, x].tolist()} got {got[k, y, x].tolist()}"
            return ""

        pil = np.stack([np.asarray(f.convert("RGBA")) for f in ImageSequence.Iterator(Image.open(png))]).astype(int)
        err = same(pil)
        report("apng", f"{name} in Pillow", not err, err)
        if have_ffmpeg:
            raw = work / f"{name}.ff"
            subprocess.run(["ffmpeg", "-v", "fatal", "-y", "-i", str(png), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgba", str(raw)], check=True)
            data = np.frombuffer(raw.read_bytes(), np.uint8)
            err = same(data.reshape(-1, h, w, 4).astype(int)) if data.size % (w * h * 4) == 0 else "output size"
            report("apng", f"{name} in ffmpeg", not err, err)


AREAS = {"lossless": check_lossless, "arith": check_arith, "hier": check_hier, "resize": check_resize, "anim": check_anim, "browser": check_browser, "apng": check_apng, "jpeg422": check_jpeg422, "gzip": check_gzip}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("areas", nargs="*", choices=list(AREAS), help="default: all")
    ap.add_argument("--rc", default=str(ROOT / "target" / "release" / "rc"))
    ap.add_argument("--work", help="scratch directory (default: a temporary one)")
    args = ap.parse_args()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="crosscheck-"))
    work.mkdir(parents=True, exist_ok=True)
    for area in args.areas or list(AREAS):
        print(f"== {area}")
        sub = work / area
        sub.mkdir(exist_ok=True)
        try:
            AREAS[area](args.rc, sub)
        except Exception as e:
            report(area, "could not run", False, f"{type(e).__name__}: {e}")
    failed = [r for r in results if not r[2]]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed" + (f"; failed: {[r[1] for r in failed]}" if failed else ""))
    sys.exit(1 if failed else 0)
