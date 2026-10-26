# Changelog

## Unreleased

- Resizing now filters in linear light (assuming the sRGB transfer curve) instead of on the gamma-encoded values, so
  fine detail keeps its brightness (a black/white checkerboard averages to 188, not 128). `--resize-encoded` brings back
  the old behavior, which still matches Pillow's Lanczos to within 0.13 levels; the new default differs from Pillow by
  about 2 levels on a photo-like image.
- CMYK and YCCK JPEGs are now re-quantized to `-q` in the DCT domain (they were only ever transcoded losslessly, so `-q`
  did nothing for them). Sources at or below the target quality are still transcoded losslessly. All four ink channels of
  CMYK use the luma table; in YCCK, Y and K do and Cb/Cr use the chroma table. The requantizer now gives components that
  end up with the same table one slot and builds the trellis rate model per component, which also fixes files whose Cb
  and Cr start with different tables. Output decodes in libjpeg with the Adobe flag intact (31-79% smaller on the
  fixtures).
- New YCCK test files (`fixtures/ycck*.jpg`, 4:2:0-style and 4:4:4), built by `jpeg/examples/make_ycck.rs` and checked
  against libjpeg's decode.
- Animated WebP output: `-f webp` turns animated GIF, APNG, animated WebP and animated AVIF into an animated WebP.
  Lossless by default, lossy (VP8) frames with `--lossy`; after the first frame only the changed rectangle is stored,
  and each lossless frame is coded both replacing the canvas and blended over it, keeping the smaller. An animated WebP
  input is now re-encoded with `--lossy` (it was always left unchanged before). Verified frame by frame with Pillow
  (libwebp): pixel-identical in the lossless case.
- GIF output: `-f gif` writes stills and animations (any input format). One shared palette (`--colors`, default 256),
  frames stored as the changed rectangle with unchanged pixels transparent, a new LZW encoder, and disposal handling
  for frames that make visible pixels transparent again; animations are not dithered, stills use `--dither`. A GIF kept
  as a GIF is re-coded without changing any pixel (about 30% smaller on the test animation) and is left unchanged when
  its frames need more than one palette, unless `--lossy` or `--resize` is given. GIFs are now picked up by directory
  runs. Verified frame by frame with Pillow.
- The color quantizer moved from `rc-png` to `rc-core` (still available as `rc_png::quant`), so the GIF writer shares it.

## 0.1.1 - 26-10-2026

- `rc update`: checks GitHub for the newest release, shows `current -> newer`, and replaces the running binary.
  The archive's SHA-256 is verified against the published `.sha256` file and the new binary must pass `--version`
  before the atomic swap, so a failed update leaves the installed one untouched. `--check` only reports, `--to <tag>`
  installs a specific release (older ones too), `--force` reinstalls. Needs `curl` and `tar` on `PATH`; a file named
  `update` must now be passed as `./update`. The Windows swap is implemented but untested.

- Repository layout: a Cargo workspace with one crate per codec (`core`, `deflate`, `png`, `jpeg`, `webp`, `gif`, `avif`) and a
  `cli` crate for the `rc` binary; each crate has its own tests and examples, and the fixtures moved to `fixtures/`.

- AVIF: inter prediction. The decoder implements the whole AV1 inter toolset (reference management, compound
  prediction, inter-intra, OBMC, warped motion, temporal motion vectors, scaled references) and matches dav1d bit for
  bit on libaom streams; animated AVIFs from libavif and ffmpeg now decode. Animated AVIF output uses inter frames
  (block matching against the previous frame) and is 6-8x smaller than key-frame-only output on panning footage.

- AVIF: intra block copy. The decoder reads `allow_intrabc` streams bit for bit like dav1d (libaom screen-content
  streams, 4:2:0/4:4:4, 8/10-bit, tiles); the encoder finds exact repeats with a window hash and keeps the block copy
  variant when it wins (a text screenshot shrinks about 4.5x; ordinary photos are unaffected).

- Animated AVIF output (`avis`) from animated GIF, APNG, animated WebP and AVIF sequences; verified with libavif (Pillow) and dav1d (ffmpeg).

- AVIF: 10-bit output (`--bit-depth auto|8|10`); 16-bit sources are kept at 10 bits by default.

- AVIF encoder: about 2x faster (candidate transforms are ranked in the transform domain and only the winner is
  reconstructed; the mode search uses DCT only), at roughly 1% larger output on photos.
- AVIF encoder: loop restoration (Wiener and self-guided, per-unit switchable), written by a second pass over the
  tiles. Streams decode identically in dav1d.
- AVIF encoder: blocks with no residual are now signalled as skipped (the flag was always written as false), and the
  quantizer dead zone is tighter (about 2% smaller at equal PSNR).
- Measured against libaom (`cpu-used 4`) at equal PSNR with both outputs decoded the same way: −6.5% on average over
  eight images (previous build −5.6%).

## 0.1.0 - 19-10-2026

- Initial release: fast PNG and JPEG compression with a from-scratch engine (DEFLATE, PNG, JPEG).
- PNG: binary-tree match finder and dual-seed cost refinement for levels 7-9; bigram and brute-force
  filter strategies. Level 7 matches or beats `oxipng -o2` and level 9 matches or beats `oxipng -o4`
  on most images.
- Packaging: MIT license, crate metadata, CI and release workflows, a `curl | sh` installer and
  SHA-256 checksums for the release binaries.
- Lossless WebP (VP8L) reading and writing; GIF (still) input; CMYK/YCCK JPEG transcoding; animated PNG
  and lossy/animated WebP are passed through untouched instead of failing.
- `--keep-metadata` (EXIF, XMP, comments, PNG text/time/density, converted between formats),
  `--resize` (Lanczos-3 downscale), and stdin/stdout support (`-` as input, `-o -`).
- Lossy WebP (VP8), written from scratch: a decoder that matches libwebp bit for bit and a
  rate-distortion encoder (`-f webp --lossy -q N`) with a lossless `ALPH` alpha channel.
- AVIF, written from scratch: an AV1 still-picture decoder (bit-exact with dav1d), a HEIF container reader and
  writer (grids, alpha, transforms, ICC/EXIF/XMP) and a rate-distortion intra encoder (`-f avif -q N`). See
  `THIRD_PARTY_NOTICES.md` for the generated tables.
- Hostile-input mutation tests for every decoder.
