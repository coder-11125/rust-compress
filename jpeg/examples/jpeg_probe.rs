//! Developer probe: time the phases of the JPEG -> JPEG path for one file.
use rc_jpeg::decode::parse;
use rc_jpeg::encode::{write_best, write_jpeg};
use rc_jpeg::progressive::write_progressive;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: jpeg_probe <file.jpg> [quality]");
    let quality: u8 = std::env::args().nth(2).and_then(|q| q.parse().ok()).unwrap_or(80);
    let data = std::fs::read(path).unwrap();
    let t = Instant::now();
    let parsed = parse(&data).unwrap();
    println!("parse (entropy decode)      {:>8.1?}", t.elapsed());
    let t = Instant::now();
    let mut ci = parsed.into_coefs().unwrap();
    println!("into_coefs                  {:>8.1?}", t.elapsed());
    let t = Instant::now();
    ci.requantize(quality, true);
    println!("requantize q{quality} (trellis)   {:>8.1?}", t.elapsed());
    let t = Instant::now();
    let b = write_jpeg(&ci);
    println!("write baseline              {:>8.1?}  {} B", t.elapsed(), b.len());
    let t = Instant::now();
    let p = write_progressive(&ci);
    println!("write progressive (search)  {:>8.1?}  {} B", t.elapsed(), p.len());
    let t = Instant::now();
    let _ = write_best(&ci);
    println!("write_best                  {:>8.1?}", t.elapsed());
}
