//! Developer tool: build YCCK JPEGs (Adobe transform 2) from an RGB PNG, for the CMYK test fixtures.
//!
//! usage: make_ycck <input.png> <out_420.jpg> <out_444.jpg>
//!
//! The picture is split the way Photoshop's CMYK export does it: K carries the darkness (`255 - max(r, g, b)` of ink),
//! C, M and Y the remaining color, all stored inverted (255 = no ink). The first three planes are then YCC-coded from
//! `255 - light`, exactly what libjpeg's YCCK decoder undoes. The 4:2:0-style file samples Y and K at full resolution
//! and Cb/Cr at half, which is the layout those exports use. Input sizes must be multiples of 16.
use rc_core::image::{ColorType, Image};
use rc_jpeg::encode::{CoefImage, EncodeOptions, Subsampling, build_coefs, write_jpeg};
use rc_png::decode::decode;

fn build(rgb: &Image, sub: Subsampling, quality: u8) -> CoefImage {
    let (w, h) = (rgb.width as usize, rgb.height as usize);
    let (mut ycc, mut kplane) = (Vec::with_capacity(w * h * 3), Vec::with_capacity(w * h));
    for p in rgb.data.chunks_exact(3) {
        let max = p[0].max(p[1]).max(p[2]) as u32;
        // Light each ink lets through, so that light_c * light_k / 255 reproduces the source channel.
        for &v in p {
            let light = (v as u32 * 255 + max / 2).checked_div(max).unwrap_or(255);
            ycc.push(255 - light.min(255) as u8);
        }
        kplane.push(max as u8);
    }
    let opts = EncodeOptions { quality, subsampling: sub, ..EncodeOptions::default() };
    let mut ci = build_coefs(&Image::new(rgb.width, rgb.height, ColorType::Rgb, 8, ycc).unwrap(), &opts);
    let mut k =
        build_coefs(&Image::new(rgb.width, rgb.height, ColorType::Gray, 8, kplane).unwrap(), &opts).comps.remove(0);
    // K is sampled like Y, and uses the luma table.
    k.id = 4;
    k.h = ci.comps[0].h;
    k.v = ci.comps[0].v;
    k.tq = ci.comps[0].tq;
    assert_eq!(k.blocks_w, ci.comps[0].blocks_w, "input size must be a multiple of the MCU size");
    ci.comps.push(k);
    ci.adobe_transform = Some(2);
    ci
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (input, out420, out444) =
        (args.next().expect("input.png"), args.next().expect("out_420"), args.next().expect("out_444"));
    let img = decode(&std::fs::read(input).unwrap()).unwrap();
    assert_eq!(img.color, ColorType::Rgb, "expected an RGB PNG");
    assert!(img.width.is_multiple_of(16) && img.height.is_multiple_of(16), "size must be a multiple of 16");
    std::fs::write(out420, write_jpeg(&build(&img, Subsampling::S420, 85))).unwrap();
    std::fs::write(out444, write_jpeg(&build(&img, Subsampling::S444, 92))).unwrap();
}
