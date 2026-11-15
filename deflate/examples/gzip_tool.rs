//! Developer tool for cross-checking this crate's gzip / zlib / DEFLATE against other implementations.
//!
//!   gzip_tool compress   <gzip|zlib|raw> <level> <in> <out>
//!   gzip_tool compress   zlibdict        <level> <in> <out> <dict>
//!   gzip_tool decompress <gzip|zlib|raw> <in> <out>           (exit status 1 and a message on stderr if rejected)
//!   gzip_tool decompress zlibdict        <in> <out> <dict>
use rc_deflate::inflate::{gzip_decompress, inflate, zlib_decompress, zlib_decompress_dict};
use rc_deflate::{deflate_compress, gzip_compress, zlib_compress, zlib_compress_dict};

const MAX_OUT: usize = 1 << 30;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let read = |i: usize| std::fs::read(&a[i]).unwrap_or_else(|e| panic!("{}: {e}", a[i]));
    let (mode, kind) = (a[1].as_str(), a[2].as_str());
    let out = match mode {
        "compress" => {
            let level: u32 = a[3].parse().unwrap();
            let data = read(4);
            let r = match kind {
                "gzip" => gzip_compress(&data, level),
                "zlib" => zlib_compress(&data, level),
                "raw" => deflate_compress(&data, level),
                "zlibdict" => zlib_compress_dict(&data, level, &read(6)),
                _ => panic!("unknown format {kind}"),
            };
            std::fs::write(&a[5], r).unwrap();
            return;
        }
        "decompress" => {
            let data = read(3);
            match kind {
                "gzip" => gzip_decompress(&data, 0, MAX_OUT),
                "zlib" => zlib_decompress(&data, 0, MAX_OUT),
                "raw" => inflate(&data, 0, MAX_OUT),
                "zlibdict" => zlib_decompress_dict(&data, &read(5), 0, MAX_OUT),
                _ => panic!("unknown format {kind}"),
            }
        }
        _ => panic!("compress or decompress"),
    };
    match out {
        Ok(v) => std::fs::write(&a[4], v).unwrap(),
        Err(e) => {
            eprintln!("rejected: {e}");
            std::process::exit(1);
        }
    }
}
