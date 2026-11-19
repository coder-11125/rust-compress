//! Prints the SSIMULACRA 2 score of two PNG files: `ssim2 REFERENCE.png DISTORTED.png`.
//! Used by tools/ssim2_parity.py to compare this implementation with the Python reference.
use rc_png::decode::decode;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: ssim2 REFERENCE.png DISTORTED.png");
        std::process::exit(2);
    }
    let load = |p: &str| {
        let data = std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("{p}: {e}");
            std::process::exit(1)
        });
        decode(&data).unwrap_or_else(|e| {
            eprintln!("{p}: {e}");
            std::process::exit(1)
        })
    };
    let (a, b) = (load(&args[0]), load(&args[1]));
    if std::env::var_os("SSIM2_DECODE_ONLY").is_some() {
        return;
    }
    let start = std::time::Instant::now();
    let result = rc_core::ssimulacra2::ssimulacra2(&a, &b);
    if std::env::var_os("SSIM2_TIME").is_some() {
        eprintln!("score took {:.3} s for {}x{}", start.elapsed().as_secs_f64(), a.width, a.height);
    }
    match result {
        Ok(score) => println!("{score:.10}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    }
}
