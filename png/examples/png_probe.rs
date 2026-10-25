//! Developer probe: sizes/timings of the PNG encoder for one file at several levels.
use rc_png::decode::decode;
use rc_png::encode::{EncodeOptions, encode};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: png_probe <file.png>");
    let data = std::fs::read(&path).unwrap();
    let t = Instant::now();
    let img = decode(&data).unwrap();
    println!("original {} bytes; decode {:.2?}", data.len(), t.elapsed());
    for level in [1u32, 6, 7, 8, 9] {
        let t = Instant::now();
        let out = encode(&img, &EncodeOptions { level, ..Default::default() });
        println!(
            "level {level}: {} bytes ({:+.1}%) in {:.2?}",
            out.len(),
            (out.len() as f64 / data.len() as f64 - 1.0) * 100.0,
            t.elapsed()
        );
    }
}
