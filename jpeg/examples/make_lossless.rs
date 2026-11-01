//! Developer tool: rewrite lossless JPEG files with every predictor, restart markers and a point transform, for
//! cross-checking with other decoders.
//!
//! usage: make_lossless <out_dir> <in.jpg>...
//! For `a.jpg` it writes `a.p1.jpg` .. `a.p7.jpg`, `a.auto.jpg`, `a.rst.jpg`, `a.arith.jpg`, `a.arith_rst.jpg` and (when the precision allows) `a.pt2.jpg`.
use rc_jpeg::decode::parse;
use rc_jpeg::lossless::LosslessOptions;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args.next().expect("out_dir");
    for input in args {
        let data = std::fs::read(&input).unwrap();
        let stem = Path::new(&input).file_stem().unwrap().to_string_lossy().into_owned();
        let save = |name: &str, bytes: &[u8]| {
            std::fs::write(Path::new(&out_dir).join(format!("{stem}.{name}.jpg")), bytes).unwrap()
        };
        let frame = || parse(&data).unwrap().lossless_frame().unwrap();
        for p in 1..=7u8 {
            let r =
                frame().write(&LosslessOptions { predictor: Some(p), restart_rows: 0, ..Default::default() }).unwrap();
            save(&format!("p{p}"), &r.bytes);
        }
        save("auto", &frame().write(&LosslessOptions::default()).unwrap().bytes);
        save(
            "rst",
            &frame().write(&LosslessOptions { predictor: None, restart_rows: 3, ..Default::default() }).unwrap().bytes,
        );
        // Arithmetic-coded (SOF11): no other writer exists, so these are for trying other decoders.
        save("arith", &frame().write(&LosslessOptions { arithmetic: true, ..Default::default() }).unwrap().bytes);
        save(
            "arith_rst",
            &frame().write(&LosslessOptions { arithmetic: true, restart_rows: 3, ..Default::default() }).unwrap().bytes,
        );
        let mut f = frame();
        if f.precision > 4 {
            f.point_transform = 2;
            for c in &mut f.comps {
                c.samples.iter_mut().for_each(|v| *v >>= 2);
            }
            save("pt2", &f.write(&LosslessOptions::default()).unwrap().bytes);
        }
    }
}
