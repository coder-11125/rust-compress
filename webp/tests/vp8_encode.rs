//! The VP8 encoder's stream must decode to exactly the reconstruction the encoder predicted.
use rc_webp::vp8::decode::decode_frame;
use rc_webp::vp8::encode::{Options, Planes, encode_frame};
use std::path::PathBuf;

fn dir() -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "..", "fixtures", "vp8"].iter().collect()
}

fn vp8_payload(file: &[u8]) -> &[u8] {
    let mut pos = 12;
    while pos + 8 <= file.len() {
        let len = u32::from_le_bytes(file[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if &file[pos..pos + 4] == b"VP8 " {
            return &file[pos + 8..pos + 8 + len];
        }
        pos += 8 + len + (len & 1);
    }
    panic!("no VP8 chunk");
}

fn crop(plane: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
    (0..h).flat_map(|r| plane[r * stride..r * stride + w].to_vec()).collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let sse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum();
    if sse == 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 * a.len() as f64 / sse).log10() }
}

#[test]
fn encoded_streams_decode_to_the_predicted_reconstruction() {
    let index = std::fs::read_to_string(dir().join("index.txt")).unwrap();
    for line in index.lines().take(8) {
        let f: Vec<&str> = line.split(' ').collect();
        let (name, w, h): (&str, usize, usize) = (f[0], f[1].parse().unwrap(), f[2].parse().unwrap());
        let file = std::fs::read(dir().join(format!("{name}.webp"))).unwrap();
        let src = decode_frame(vp8_payload(&file)).unwrap();
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y = crop(&src.y, src.y_stride, w, h);
        let u = crop(&src.u, src.uv_stride, cw, ch);
        let v = crop(&src.v, src.uv_stride, cw, ch);
        for quality in [20.0, 75.0, 95.0] {
            let planes = Planes { width: w, height: h, y: &y, u: &u, v: &v };
            let enc = encode_frame(&planes, Options { quality, level: 4 }).unwrap();
            let got = decode_frame(&enc.data).unwrap();
            assert_eq!(
                crop(&got.y, got.y_stride, w, h),
                crop(&enc.recon.y, enc.recon.y_stride, w, h),
                "{name} q{quality} y"
            );
            assert_eq!(
                crop(&got.u, got.uv_stride, cw, ch),
                crop(&enc.recon.u, enc.recon.uv_stride, cw, ch),
                "{name} q{quality} u"
            );
            assert_eq!(
                crop(&got.v, got.uv_stride, cw, ch),
                crop(&enc.recon.v, enc.recon.uv_stride, cw, ch),
                "{name} q{quality} v"
            );
            let p = psnr(&y, &crop(&got.y, got.y_stride, w, h));
            eprintln!("{name} {w}x{h} q{quality}: {} bytes, Y psnr {p:.2}", enc.data.len());
            assert!(p > 20.0, "{name} q{quality} psnr {p}");
            if quality == 75.0 {
                std::fs::write(std::env::temp_dir().join(format!("enc_{name}.vp8")), &enc.data).ok();
                let mut raw = crop(&enc.recon.y, enc.recon.y_stride, w, h);
                raw.extend(crop(&enc.recon.u, enc.recon.uv_stride, cw, ch));
                raw.extend(crop(&enc.recon.v, enc.recon.uv_stride, cw, ch));
                std::fs::write(std::env::temp_dir().join(format!("enc_{name}.recon")), raw).ok();
            }
        }
    }
}
