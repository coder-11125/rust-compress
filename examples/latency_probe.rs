//! Developer probe: in-process latency (median of N runs) of decode and encode for one image.
use rust_compress::jpeg::{decode as jd, encode as je};
use rust_compress::png::{decode as pd, encode as pe};

unsafe extern "C" {
    fn clock() -> i64;
}

/// Process CPU time in milliseconds (immune to other processes competing for the machine).
fn cpu_ms() -> f64 {
    // SAFETY: clock() has no preconditions; CLOCKS_PER_SEC is 1_000_000 on the platforms we run on.
    unsafe { clock() as f64 / 1000.0 }
}

/// Minimum over several batches of CPU time per call, which filters out cache-cold outliers.
fn median_ms(mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = cpu_ms();
        for _ in 0..5 {
            f();
        }
        best = best.min((cpu_ms() - t) / 5.0);
    }
    best
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: latency_probe <file.png|file.jpg>");
    let data = std::fs::read(&path).unwrap();
    if data.starts_with(&[0xFF, 0xD8]) {
        let img = jd::decode(&data).unwrap();
        println!("{}x{} JPEG, {} B", img.width, img.height, data.len());
        println!("  decode                      {:6.2} ms", median_ms(|| drop(jd::decode(&data).unwrap())));
        let fast = je::EncodeOptions { trellis: false, ..Default::default() };
        println!("  encode pixels q80 no-trellis{:6.2} ms", median_ms(|| drop(je::encode(&img, &fast).unwrap())));
        let full = je::EncodeOptions::default();
        println!("  encode pixels q80 trellis   {:6.2} ms", median_ms(|| drop(je::encode(&img, &full).unwrap())));
        let ci = jd::parse(&data).unwrap().into_coefs().unwrap();
        println!(
            "  coef-domain requantize q80  {:6.2} ms",
            median_ms(|| {
                let mut c = ci.clone();
                c.requantize(80, false);
                drop(je::write_jpeg(&c));
            })
        );
    } else {
        let img = pd::decode(&data).unwrap();
        println!("{}x{} PNG, {} B", img.width, img.height, data.len());
        println!("  decode                      {:6.2} ms", median_ms(|| drop(pd::decode(&data).unwrap())));
        for level in [1u32, 3, 6, 7] {
            let o = pe::EncodeOptions { level, ..Default::default() };
            println!("  encode level {level}                {:6.2} ms", median_ms(|| drop(pe::encode(&img, &o))));
        }
    }
}
