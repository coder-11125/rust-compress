//! Developer tool: rewrite JPEG files with arithmetic coding in every mode, for cross-checking with other decoders.
//!
//! usage: make_arith <out_dir> <in.jpg>...
//! For `a.jpg` it writes `a.seq.jpg`, `a.prog.jpg`, `a.seq_rst.jpg`, `a.prog_rst.jpg`, `a.seq_cond.jpg`.
use rc_jpeg::arith::{ArithOptions, Conditioning, write_arithmetic_with};
use rc_jpeg::decode::parse;
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args.next().expect("out_dir");
    let custom = Conditioning { dc_l: [2, 1, 0, 0], dc_u: [6, 9, 1, 1], ac_k: [12, 30, 5, 5] };
    for input in args {
        let data = std::fs::read(&input).unwrap();
        let ci = parse(&data).unwrap().into_coefs().unwrap();
        let stem = Path::new(&input).file_stem().unwrap().to_string_lossy().into_owned();
        let modes = [
            ("seq", ArithOptions::default()),
            ("prog", ArithOptions { progressive: true, ..ArithOptions::default() }),
            ("seq_rst", ArithOptions { restart_interval: 3, ..ArithOptions::default() }),
            ("prog_rst", ArithOptions { progressive: true, restart_interval: 5, ..ArithOptions::default() }),
            ("seq_cond", ArithOptions { conditioning: custom.clone(), ..ArithOptions::default() }),
        ];
        for (name, opts) in modes {
            std::fs::write(Path::new(&out_dir).join(format!("{stem}.{name}.jpg")), write_arithmetic_with(&ci, &opts))
                .unwrap();
        }
    }
}
