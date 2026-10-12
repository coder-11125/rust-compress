//! Developer probe: compare the structure of DEFLATE streams (e.g. ours vs. oxipng/zopfli).
use rust_compress::deflate::{deflate_compress, inflate::stream_stats};

fn main() {
    let mut args = std::env::args().skip(1);
    let raw = std::fs::read(args.next().expect("usage: stream_probe <raw filtered data> <zlib streams...>")).unwrap();
    for level in [7u32, 9] {
        let ours = deflate_compress(&raw, level);
        println!("ours level {level}: {} B {:?}", ours.len(), stream_stats(&ours).unwrap());
    }
    for path in args {
        let z = std::fs::read(&path).unwrap();
        let body = &z[2..z.len() - 4];
        println!("{path}: {} B {:?}", z.len(), stream_stats(body).unwrap());
    }
}
