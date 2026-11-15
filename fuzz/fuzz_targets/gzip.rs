#![no_main]
use libfuzzer_sys::fuzz_target;
use rc_deflate::inflate::{gzip_decompress, inflate_with_end_dict, zlib_decompress_dict};
use rc_deflate::{gzip_compress, zlib_compress_dict};

const MAX_OUT: usize = 16 << 20;

// gzip members and preset-dictionary streams from untrusted bytes, then round trips of the bytes themselves.
fuzz_target!(|data: &[u8]| {
    let _ = gzip_decompress(data, 0, MAX_OUT);

    // The first byte says how much of the rest is the dictionary; the remainder is the stream (or the payload).
    let Some((&n, rest)) = data.split_first() else { return };
    let (dict, body) = rest.split_at((n as usize * 4).min(rest.len()));
    let _ = zlib_decompress_dict(body, dict, 0, MAX_OUT);
    let _ = inflate_with_end_dict(body, dict, 0, MAX_OUT);

    if data.len() <= 1 << 16 {
        let g = gzip_compress(data, 1);
        assert_eq!(gzip_decompress(&g, 0, MAX_OUT).unwrap(), data, "gzip round trip");
        let z = zlib_compress_dict(body, 1 + n as u32 % 9, dict);
        assert_eq!(zlib_decompress_dict(&z, dict, 0, MAX_OUT).unwrap(), body, "zlib dictionary round trip");
    }
});
