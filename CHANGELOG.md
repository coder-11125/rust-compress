# Changelog

## Unreleased

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
