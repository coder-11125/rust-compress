//! Developer probe: raw DEFLATE sizes/timings for one file at several levels.
use rc_deflate::deflate_compress;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: deflate_probe <file>");
    let data = std::fs::read(&path).unwrap();
    println!("input {} bytes", data.len());
    let levels: Vec<u32> = std::env::var("LEVELS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 3, 6, 7, 8, 9]);
    for level in levels {
        let t = Instant::now();
        let out = deflate_compress(&data, level);
        println!("ours level {level}: {} bytes in {} ms", out.len(), t.elapsed().as_millis());
    }
}
