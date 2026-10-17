//! Developer tool: encode a PNG as an AV1 OBU stream with the from-scratch encoder.
//! usage: av1_enc in.png out.obu recon.yuv [qidx] [bsize]
use rust_compress::avif::av1::encode::{EncodeParams, encode_picture};
use rust_compress::avif::av1::frame::{Frame, Plane};
use rust_compress::avif::av1::obu::ColorConfig;
use rust_compress::avif::convert::rgb_to_yuv;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let img = rust_compress::png::decode::decode(&data).unwrap();
    let color = ColorConfig {
        bit_depth: 8,
        mono_chrome: false,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
        color_range: true,
        sub_x: 1,
        sub_y: 1,
        chroma_sample_position: 0,
        separate_uv_delta_q: false,
    };
    let yuv = rgb_to_yuv(&img, 8, 1, 1, 0.299, 0.114);
    let (w, h) = (img.width as usize, img.height as usize);
    let mk = |v: &Vec<u16>, pw: usize, ph: usize| {
        let mut p = Plane::new(pw, ph);
        p.data.copy_from_slice(v);
        p
    };
    let frame = Frame {
        planes: vec![mk(&yuv.y, w, h), mk(&yuv.u, yuv.cw, yuv.ch), mk(&yuv.v, yuv.cw, yuv.ch)],
        bit_depth: 8,
        sub_x: 1,
        sub_y: 1,
        width: w,
        height: h,
    };
    if let Some(path) = a.get(6) {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for pl in &frame.planes {
            for y in 0..pl.height {
                out.write_all(&pl.row(y).iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
            }
        }
    }
    let mut p = EncodeParams::default();
    if let Some(q) = a.get(4) {
        p.base_q_idx = q.parse().unwrap();
    }
    if let Some(b) = a.get(5) {
        p.fixed_bsize = b.parse().unwrap();
    }
    if let Ok(v) = std::env::var("AV1_LAMBDA") {
        p.lambda_scale = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("AV1_EFFORT") {
        p.effort = v.parse().unwrap();
    }
    if let Ok(v) = std::env::var("AV1_ROUND") {
        p.quant_round = v.parse().unwrap();
    }
    let e = encode_picture(&frame, &color, &p).unwrap();
    let mut obu = vec![0x12, 0x00];
    obu.extend_from_slice(&e.data);
    std::fs::write(&a[2], &obu).unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[3]).unwrap());
    for pl in &e.recon.planes {
        for y in 0..pl.height {
            out.write_all(&pl.row(y).iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
        }
    }
    eprintln!("{} bytes", e.data.len());
}
