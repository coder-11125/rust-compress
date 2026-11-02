#![no_main]
use libfuzzer_sys::fuzz_target;

const MAX_OUT: usize = 16 << 20;

fuzz_target!(|data: &[u8]| {
    let _ = rc_deflate::inflate::inflate(data, 0, MAX_OUT);
    let _ = rc_deflate::inflate::inflate_with_end(data, 1 << 10, MAX_OUT);
    let _ = rc_deflate::inflate::zlib_decompress(data, 0, MAX_OUT);
    let _ = rc_deflate::inflate::stream_stats(data);
});
