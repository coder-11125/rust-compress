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


def example(work, name, *args):
    subprocess.run(["cargo", "run", "-q", "-p", "rc-jpeg", "--example", name, "--", *map(str, args)], cwd=ROOT, check=True)


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


AREAS = {"lossless": check_lossless, "arith": check_arith, "hier": check_hier, "resize": check_resize, "anim": check_anim, "browser": check_browser}

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
