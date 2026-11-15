#![no_main]
use libfuzzer_sys::fuzz_target;
use rc_avif::av1::decoder::decode;
use rc_avif::av1::encode::{EncodeParams, encode_picture};
use rc_avif::av1::frame::{Frame, Plane};
use rc_avif::av1::obu::ColorConfig;

// Differential check of the palette encoder against the decoder: a small picture built from a few colours (so palette
// blocks are likely) is coded with palette mode forced on, and the decoder must reproduce the encoder's reconstruction.
fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let (head, body) = data.split_at(6);
    let (w, h) = (8 + head[0] as usize % 72, 8 + head[1] as usize % 72);
    let (mono, sx, sy) = match head[2] % 4 {
        0 => (false, 0u8, 0u8),
        1 => (false, 1, 1),
        2 => (false, 1, 0),
        _ => (true, 1, 1),
    };
    let bit_depth = [8u8, 10, 12][(head[2] >> 6) as usize % 3];
    let sh = bit_depth as u32 - 8;
    let qidx = 1 + head[3] as u32 % 255;
    let palette: Vec<[u8; 3]> =
        body.chunks(3).take(7).map(|c| [c[0], *c.get(1).unwrap_or(&128), *c.get(2).unwrap_or(&128)]).collect();
    let pick = |x: usize, y: usize| -> [u8; 3] {
        let i = (y / 8) * 16 + x / 8;
        let b = body.get(body.len() / 2 + i % (body.len() / 2).max(1)).copied().unwrap_or(0);
        let mut c = palette[b as usize % palette.len()];
        if b & 0x80 != 0 {
            // a little texture inside the block, so some blocks need more than their flat colour
            c[0] = c[0].wrapping_add(((x * 3 + y) % 5) as u8 * (b & 3));
        }
        c
    };
    let (cw, ch) = (w.div_ceil(1 << sx), h.div_ceil(1 << sy));
    let mut planes = vec![Plane::new(w, h)];
    if !mono {
        planes.push(Plane::new(cw, ch));
        planes.push(Plane::new(cw, ch));
    }
    for y in 0..h {
        for x in 0..w {
            let c = pick(x, y);
            planes[0].data[y * w + x] = (c[0] as u16) << sh;
            if !mono && x % (1 << sx) == 0 && y % (1 << sy) == 0 {
                planes[1].data[(y >> sy) * cw + (x >> sx)] = (c[1] as u16) << sh;
                planes[2].data[(y >> sy) * cw + (x >> sx)] = (c[2] as u16) << sh;
            }
        }
    }
    let frame = Frame { planes, bit_depth, sub_x: sx, sub_y: sy, width: w, height: h };
    let color = ColorConfig {
        bit_depth,
        mono_chrome: mono,
        color_primaries: 1,
        transfer_characteristics: 13,
        matrix_coefficients: 6,
        color_range: true,
        sub_x: sx,
        sub_y: sy,
        chroma_sample_position: 0,
        separate_uv_delta_q: false,
    };
    let params = EncodeParams {
        base_q_idx: qidx as u8,
        effort: 1 + head[4] % 8,
        intrabc: false,
        palette_frame: true,
        ..EncodeParams::default()
    };
    let Ok(enc) = encode_picture(&frame, &color, &params) else { return };
    let mut stream = vec![0x12, 0x00];
    stream.extend_from_slice(&enc.data);
    let dec = decode(&stream).expect("the encoder's own stream must decode");
    for (p, q) in dec.frame.planes.iter().zip(&enc.recon.planes) {
        for y in 0..p.height.min(q.height) {
            assert_eq!(p.row(y), q.row(y), "decoder and encoder disagree on row {y}");
        }
    }
});
