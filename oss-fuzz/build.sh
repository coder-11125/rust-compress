#!/bin/bash -eu
# Builds every target in fuzz/ and packages seed corpora from the checked-in fixtures.
cd "$SRC/rust-compress"
cargo fuzz build -O --fuzz-dir fuzz

TARGET_DIR="fuzz/target/x86_64-unknown-linux-gnu/release"
for target in inflate png jpeg_decode jpeg_coefs webp gif avif av1_obu; do
  cp "$TARGET_DIR/$target" "$OUT/$target"
done

seed() { # seed <target> <glob...>
  local name=$1; shift
  zip -qj "$OUT/${name}_seed_corpus.zip" "$@"
}
seed inflate     fixtures/*.png
seed png         fixtures/*.png
seed jpeg_decode fixtures/*.jpg
seed jpeg_coefs  fixtures/*.jpg
seed webp        fixtures/*.webp
seed gif         fixtures/*.gif
seed avif        fixtures/*.avif
seed av1_obu     fixtures/*.obu
