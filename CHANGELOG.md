# Changelog

## Unreleased

- Metadata no longer leaks through the "keep the original" fallback. When re-encoding does not help, `rc` keeps the
  original bytes, which used to carry their EXIF, XMP, text chunks and comments along although only colour
  information is meant to survive. The descriptive metadata is now cut out of the PNG chunks, JPEG segments or WebP
  chunks without touching the image data (a saving, reported as such), unless `--keep-metadata` is given. A file
  whose EXIF rotates the image is left alone, and GIF and AVIF are still copied as they are.
- `--in-place` follows a symlink and updates the file it points to (the link used to be replaced by a regular file),
  refuses a file with several hard links (replacing it would split them), and no longer overwrites a file that was
  changed while it was being processed.
- New `--max-pixels` lowers the image size limit, and `--max-memory` (default 4096 MiB) makes images that would need a
  lot of memory wait for each other instead of being decoded all at once; one 20-megapixel 16-bit PNG batch peaked at
  0.47 GB instead of 1.18 GB. A JPEG with more than 1000 scans and an animation with more than 50,000 frames are
  refused (they only cost time).
- Embedded ICC profiles that are not plausible (no `acsp` header, size field not matching, over 8 MiB) are dropped
  instead of being carried along as an arbitrary payload.
- A conversion or resize that makes a file bigger is reported as `larger` (JSON status `larger`, summary count
  `larger`) instead of `compressed`; the README no longer promises that every output is smaller.
- File names and error text are printed with control characters escaped, so a name cannot inject terminal escape
  sequences. `--dither` must be 0.0 to 1.0, `--jobs` 1 to 1024, and a pipe or device given as an input is reported
  as "not a regular file" instead of "no such file". The "no files found" message names all five formats.
- Fixed an endless loop in `deflate::inflate::stream_stats` on truncated input and on the reserved block type 3 (found
  by the `inflate` fuzz target); only the fuzzer and an example call it, not the CLI.
- Removed the stale mention of `oss-fuzz/` from the fuzz README.

- Fixed a crash on any 12 to 15 byte input that starts with an `ftyp` box: the AVIF format check panicked, which
  aborted the whole batch (release builds use `panic = "abort"`) before any report was written.
- `rc update` and `install.sh` now require the published SHA-256 checksum and refuse to install without one (before,
  a missing `.sha256` only printed a warning and the unverified binary was installed and run). The checksum file must
  hold a real digest, `RC_UPDATE_BASE_URL` must be `https://` (plain `http://` is accepted only for localhost), and
  downloads cannot be redirected from `https` back to `http`.

## 0.3.0 - 15-11-2026

- AVIF encoder: palette mode now also runs in inter frames (its trial is skipped only for lossless, which the AV1
  encoder never supported), and the AV1 encoder is checked in every format it can be given: 4:4:4, 4:2:2 and 4:2:0
  at 8, 10 and 12 bits, palette on and off, all bit-exact in dav1d and libaom. Two bugs that this exposed are fixed:
  the sequence header wrote profile 1 for 12-bit 4:4:4 (the 12-bit flag was never sent, so decoders read 10-bit),
  and the partition search tried partitions with no valid chroma block size in 4:2:2 and panicked. `base_q_idx = 0`
  (lossless) now returns an error instead of panicking. None of these formats is reachable from the CLI.
- AVIF encoder palette mode: pictures with enough small few-colour blocks (screenshots, UI, text, line art) get a
  second pass with `allow_screen_content_tools`, and the cheaper pass in rate-distortion terms is kept (so photos
  are untouched and pay nothing). Per block the search tries the block's exact colours when it has 2-8 of them, or
  8/4 clustered colours (weighted k-means) when it has up to 64, for luma and for the U/V pair, and prices the
  palette cache reuse, colour literals/deltas, V delta coding and the index map from the live CDFs. Block-copy
  frames can use palette blocks too. Output is bit-exact in dav1d and libaom (8/10-bit, 4:2:0/4:4:4, monochrome,
  odd sizes, many tiles). Palette is used for key frames only (not inter frames of animations).
- APNG dirty rectangles and blend modes: after the first frame, each frame is stored as only the rectangle that
  changed from the previous canvas (`fcTL` offsets), using `APNG_BLEND_OP_OVER` when that is exact and smaller
  (unchanged pixels inside the rectangle become transparent; only possible when every changed pixel is opaque) and
  `SOURCE` otherwise. A frame identical to its predecessor becomes a 1x1 no-op, so frame counts and delays are
  unchanged. Frames are compressed in parallel. Dispose operations are not used yet.
- gzip and zlib preset dictionaries (`rc-deflate`): `gzip_compress`/`gzip_decompress` (RFC 1952; the decoder reads
  concatenated members and the optional extra/name/comment/header-CRC fields, and checks CRC-32 and size), and
  `zlib_compress_dict`/`zlib_decompress_dict` plus `deflate_compress_dict`/`inflate_with_end_dict` for streams that
  start from a preset dictionary (FDICT, DICTID checked against the dictionary's Adler-32; only the last 32 KiB can
  be referenced). `zlib_decompress` still rejects dictionary streams, now with a clearer message. Verified against
  Python's zlib/gzip and the `gzip` tool in both directions.
- JPEG 4:2:2: `--subsampling 422` (and `Subsampling::S422` in `rc-jpeg`) halves chroma horizontally only. JPEG
  sources already in 4:2:2 are recognized by the DCT-domain route; AVIF output treats 422 like 420.
- GIF multi-palette lossless recode: an animated GIF whose frames each fit a palette but whose union does not is
  now re-encoded exactly instead of being left for `--lossy`. Frames that share one palette still use a single
  global color table; when they differ, every image gets a local color table (index 0 reserved for
  transparency/unchanged pixels) while keeping the changed-rectangle coding, disposal handling, delays and loop
  count. `palette_is_exact` now accepts either one shared palette or one local palette per frame.
- Batch output templates and JSON reports: `--name-template` replaces the default output path pattern when `-o` and
  `--in-place` are not used (`{root}`, `{dir}`, `{rel}`, `{rel_stem}`, `{stem}`, `{name}`, `{ext}`, `{suffix}`), and
  `--json` prints per-file results, the summary and input-level problems as one JSON object instead of text lines.
- CLI batch ergonomics: directory walks can be filtered with repeatable `--include`/`--exclude` globs (`*` and `?`
  stay within a path component, `**` crosses directories; patterns match the file name or the path relative to the
  input directory), default outputs use a configurable `--min-suffix` (still `<name>.<suffix>.<ext>`, and files
  already ending in `.<suffix>.` are skipped), and `--in-place` now preserves the original permissions and
  modification time after the atomic replace.
- APNG output: animated GIF, APNG, animated WebP and animated AVIF inputs now convert to an animated PNG with
  `-f png`. Frames are stored as complete RGB/RGBA canvases (lossless; `--lossy` quantizes them) with per-frame
  durations and looping; the first frame uses `IDAT` and later frames `fdAT`, with APNG sequence numbers on every
  `fcTL`/`fdAT` chunk. Verified with Pillow and this crate's own APNG decoder.
- PNG binary transparency: gray/RGBA images whose alpha is only fully transparent or opaque can drop the alpha
  channel and store the transparent pixels as a `tRNS` colour key instead (when hidden colours may be rewritten),
  instead of always keeping GA/RGBA.
- Quality targets beyond JPEG SSIM: `--target-ssim` now also applies to lossy WebP, AVIF and `--lossy` PNG, and the
  new `--target-psnr`, `--target-ms-ssim` and `--target-size` options do the same (PSNR and multi-scale SSIM metrics
  were added to `rc-core`). The search bisects quality, or palette size for PNG, and the report note shows the choice.
- Running `rc` with no arguments now prints the full usage (`rc [OPTIONS] <INPUT>...` and `rc update [OPTIONS]`), the
  same as `--help`. The error used to show only `rc <INPUT>...`, which hid both the options and the `update` command.

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
