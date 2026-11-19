# SSIMULACRA2 verification pairs

Synthetic image pairs (gradients, plasma texture, shapes, text-like strokes; no third-party images) with the score the
reference implementation gives each pair. `expected.csv` lines are `name,reference.png,distorted.png,score`; the score
is printed with `repr` (17 significant digits). Several pairs share one reference file.

Regenerate (deterministic, byte-identical on repeat): `python tools/ssim2_vectors.py [--out DIR]`. It needs numpy,
pillow, scipy and `ssimulacra2` (`pip install ssimulacra2`). JPEG and WebP distortions come from the installed Pillow,
so a different libjpeg/libwebp build can change those PNGs and their scores slightly.

Cases:

- `identical_*`, `gray_identical`: score must be exactly 100.
- `size_WxH`: 8x8, 8x9, 9x8, 15x15, 16x16, 17x33, 31x65, 64x64, 100x75, 129x97, 256x144, 320x256, 513x385. These
  exercise edge handling of the blur, the 2x2 downsample with odd sides, and the number of scales (a scale is added only
  while both sides are >= 8, so six scales need a short side >= 256).
- `jpeg_q*`, `webp_q*`, `blur_*`, `noise_*`, `posterize_*`, `colour_shift`, `brightness_*`, `sharpen_halos`,
  `translate_1px`, `different_image`: distortions on a 96x72 base, covering blocking, blur, noise, banding, colour and
  luminance error, ringing, misalignment and unrelated content. `detail_*` repeat some on a 64x64 high-detail base.
- `gray_*`: grayscale-origin images stored as RGB (R=G=B); `gray_vs_colour` compares one with its colour original.
- `dark_*`, `bright_*`, `black_vs_white`, `flat_gray_vs_gray+1`: extremes of the luminance range and flat images.

Expected values come from the pure-Python port py-ssimulacra2 (BSD-3, https://github.com/Pacidus/py-ssimulacra2), not
from the official libjxl tool. They are only as faithful to libjxl as that port is. The port reads PNGs through
Pillow `.convert("RGB")`, so alpha is dropped; no alpha pairs are included. Scores can be negative (the metric is not
clamped below).

`tools/ssim2_parity.py` compares an implementation against the port on real photographs.
