//! Developer tool: decode a raw AV1 OBU stream (or the first item of an AVIF) to planar YUV.
use std::io::Write;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    match rc_avif::av1::decoder::decode(&data) {
        Ok(d) => {
            eprintln!(
                "{}x{} bd{} {}x{} sub{}{} tiles {}x{} q{} lossless {} cdef {} lf {:?} lr {:?}",
                d.frame.width,
                d.frame.height,
                d.frame.bit_depth,
                d.hdr.mi_cols,
                d.hdr.mi_rows,
                d.frame.sub_x,
                d.frame.sub_y,
                d.hdr.tile_info.cols,
                d.hdr.tile_info.rows,
                d.hdr.quant.base_q_idx,
                d.hdr.coded_lossless,
                d.seq.enable_cdef,
                d.hdr.loop_filter.level,
                d.hdr.lr.kind
            );
            let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
            for p in &d.frame.planes {
                for y in 0..p.height {
                    for &v in p.row(y) {
                        if d.frame.bit_depth > 8 {
                            out.write_all(&v.to_le_bytes()).unwrap();
                        } else {
                            out.write_all(&[v as u8]).unwrap();
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
