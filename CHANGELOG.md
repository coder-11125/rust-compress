# Changelog

## 0.2.0 - 02-11-2026

- Hierarchical JPEG decoding is now cross-checked against the JPEG committee's reference libjpeg (Richter's, via
  `pylibjpeg-libjpeg`) on a matrix of more than a hundred streams, which found and fixed three defects: the
  two-dimensional upsampling is a single truncating average of four neighbours (it was two truncating steps, up to one
  level off); reconstructions are no longer clamped between frames (a lossless frame can now correct a DCT frame's
  overshoot); and each frame is dequantized with the tables in force when it started (a `DQT` before the next frame
  used to change how the previous one was decoded, which broke every hierarchical file whose frames use different
  tables). The pyramid writer gained same-size refinement and one-direction `EXP` modes. Fixtures were regenerated.
- Quantization tables are latched per component at its first scan (as libjpeg does), for all JPEG decoding.
- Arithmetic-coded lossless JPEG (SOF11/SOF15) is now confirmed by the reference libjpeg: 24 plain files (2-16 bits,
  gray/RGB/CMYK, with and without restart intervals) and the hierarchical forms decode identically.
- Arithmetic-coded JPEG written by IJG libjpeg 7, 8, 9 and libjpeg-turbo 2.1 (120 files, sequential and progressive)
  decodes to coefficients identical to libjpeg's own; eight of them are fixtures with Huffman twins.
- `rc update` on Windows: uses the system `tar.exe` and `curl.exe` (a `tar` earlier on `PATH`, such as Git's, cannot
  read `.zip` archives) and removes the `rc.exe.old` a previous update left behind. SHA-256 moved to `rc-core`. A
  cross-platform test swaps in a real binary on every OS.
- `tools/crosscheck.py` re-runs these comparisons (libjpeg-turbo, IJG libjpeg, the reference libjpeg, Pillow, libwebp,
  Chrome). Verified results: Chrome 150 decodes our GIF, animated WebP and animated AVIF frame for frame like Pillow;
  linear-light resizing matches numpy + Pillow; lossy animated WebP is 5-11% smaller and 0.2-1.6 dB better than libwebp.
- Hierarchical JPEG (T.81 Annex J) decoding: `DHP`/`EXP` frame sequences with non-differential DCT or lossless first
  frames and differential DCT (Huffman and arithmetic, sequential and progressive) and differential lossless frames,
  reconstructed with the standard's truncating bilinear upsampling. Kept as a JPEG it is left unchanged unless `--lossy`;
  other targets (`-f png`, ...) flatten it. Includes `rc_jpeg::hierarchical::write_hierarchical`, a pyramid writer used
  to test the decoder. Verified only against that writer (no other program reads hierarchical JPEG here): lossless
  pyramids and DCT-then-lossless pyramids reproduce the source exactly; seven fixtures are kept.
- Arithmetic-coded lossless JPEG (SOF11, and SOF15 inside hierarchical images) per Annex H: read, and written with
  `--arithmetic on`. libjpeg-turbo is built without it, so this is checked against our own writer only.
- Lossless JPEG (SOF3, Huffman): decoded exactly at 2 to 16 bits (gray, colour, CMYK; predictors 1-7; point transform;
  restart intervals; subsampled components) and recoded without changing a sample, trying all seven predictors and
  optimizing the Huffman tables: 4.2% smaller on nine photo-based files (0% to 15.4%). A lossless JPEG stays lossless
  unless `--lossy` or `--resize` is given; `-f png` keeps the precision. Verified against libjpeg-turbo's lossless
  encoder and decoder (12 reference files; 119 recoded files decode to the original samples).
- 12-bit CMYK and YCCK JPEG decoding (plain, Adobe, Adobe YCC-coded); matches libjpeg-turbo within 1-2 levels of 4095.
- JPEG arithmetic coding: SOF9 (sequential) and SOF10 (progressive) files are decoded and written, including restart
  intervals and non-default `DAC` conditioning. `--arithmetic auto|on|off` picks the output coding: `auto` (default)
  keeps an arithmetic-coded source arithmetic-coded and writes Huffman otherwise, `on` converts any JPEG losslessly,
  `off` converts back to Huffman for compatibility (honoured even when the result is larger). On five 5120x2880 photos
  the arithmetic files are 15.1% smaller than the originals and 6.0% smaller than our optimized Huffman output, at
  about the same encoding CPU time; decoding is about 30% slower. Many programs, including every browser, cannot open
  arithmetic-coded JPEG, so it is never enabled for a Huffman source by default.
- 12-bit JPEG decoding (SOF1/SOF2, Huffman or arithmetic): read into a 16-bit image, so `-f png` keeps the precision
  and 8-bit outputs reduce it with a note. Matches libjpeg-turbo's 12-bit decoder to within 1-3 levels of 4095.
- Verification: the arithmetic files our writer produces decode pixel-identically in libjpeg-turbo 3.1.4 and 3.2.0
  (50 small files in every mode, and five full-size photos); six of them are kept as fixtures, as are three 12-bit files
  made by libjpeg-turbo. A QM-coder state-table, carry and termination stress test and a wide-coefficient test cover the
  coder itself.
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
