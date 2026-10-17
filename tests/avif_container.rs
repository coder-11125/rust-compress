//! AVIF container: parsing real-world files (Pillow/libavif, ffmpeg/libaom, a hand-built grid), writing files
//! other decoders accept, and surviving damaged input.
//!
//! The expected values for the real-world fixtures were produced offline and independently of the parser: a
//! small Python box walker extracted each item's bytes (CRC-32 and length below), `ffmpeg -f obu` decoded the
//! extracted payload to exactly the pixels `ffmpeg` decodes from the `.avif` itself, and Pillow 12.3 (libavif
//! 1.4.2) opened every fixture, including the grid. No decoder is needed to run this test.
mod common;
use common::fixture;
use rust_compress::avif::container::{
    self, Av1Config, Av1Stream, AvifOutput, Clap, ColourInfo, ImageData, Nclx, Orientation, Transform,
};
use rust_compress::error::Error;
use rust_compress::util::checksum::crc32;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

fn coded(item: &container::ImageItem) -> &container::CodedItem {
    match &item.data {
        ImageData::Coded(c) => c,
        ImageData::Grid(_) => panic!("expected a coded item"),
    }
}

// ----------------------------------------------------------------------------------------------------------
// Real-world files
// ----------------------------------------------------------------------------------------------------------

#[test]
fn pillow_rgb() {
    let f = container::parse(&fixture("avif_rgb.avif")).unwrap();
    assert_eq!(&f.major_brand, b"avif");
    assert!(f.compatible_brands.contains(b"mif1") && f.compatible_brands.contains(b"miaf"));
    assert!(!f.is_sequence && f.alpha.is_none() && f.exif.is_none() && f.xmp.is_none());
    assert_eq!((f.primary.width, f.primary.height), (64, 48));
    let c = coded(&f.primary);
    assert_eq!((c.data.len(), crc32(&c.data)), (131, 0xa328b953));
    assert_eq!(c.av1c.to_bytes(), [0x81, 0x00, 0x0c, 0x00]);
    assert_eq!((c.av1c.seq_profile, c.av1c.chroma_subsampling_x, c.av1c.chroma_subsampling_y), (0, 1, 1));
    assert!(!c.av1c.monochrome && c.av1c.bit_depth() == 8);
    assert_eq!(c.pixi.as_deref(), Some(&[8u8, 8, 8][..]));
    assert_eq!(c.ispe, Some((64, 48)));
    // libavif's nclx for plain sRGB input: BT.709 primaries, sRGB transfer, BT.601 matrix, full range.
    assert_eq!(
        f.primary.colour.nclx,
        Some(Nclx { colour_primaries: 1, transfer_characteristics: 13, matrix_coefficients: 6, full_range: true })
    );
    assert!(f.primary.transforms.is_empty());
}

#[test]
fn pillow_alpha() {
    let f = container::parse(&fixture("avif_rgba.avif")).unwrap();
    assert_eq!((f.primary.width, f.primary.height), (64, 48));
    assert_eq!(crc32(&coded(&f.primary).data), 0xa328b953);
    let a = f.alpha.expect("alpha item");
    assert!(!a.premultiplied);
    assert_eq!(a.aux_type, "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha");
    assert_eq!((a.item.width, a.item.height), (64, 48));
    let ac = coded(&a.item);
    assert_eq!((ac.data.len(), crc32(&ac.data)), (43, 0x525e01f6));
    assert!(ac.av1c.monochrome);
    assert_eq!(ac.av1c.to_bytes(), [0x81, 0x00, 0x1c, 0x00]);
    assert_eq!(ac.pixi.as_deref(), Some(&[8u8][..]));
}

#[test]
fn pillow_metadata_icc_exif_xmp_irot() {
    let f = container::parse(&fixture("avif_meta.avif")).unwrap();
    assert_eq!(crc32(&coded(&f.primary).data), 0x736a256a);
    let icc = fixture("avif_meta_icc.bin");
    assert_eq!(f.icc(), Some(&icc[..]));
    // Pillow's nclx was written alongside the ICC profile.
    assert!(f.primary.colour.nclx.is_some());
    assert_eq!(f.primary.transforms, vec![Transform::Rotate(3)]);
    assert_eq!(f.primary.net_orientation(), Orientation { rotate: 3, mirror: false });
    let exif = f.exif.as_ref().expect("exif");
    // The bare TIFF block: byte order mark first, the orientation tag Pillow wrote readable.
    assert!(exif.starts_with(b"II*\0") || exif.starts_with(b"MM\0*"));
    // Pillow turns the orientation into the irot above and resets the tag, but keeps the other tags.
    assert!(exif.windows(7).any(|w| w == b"TestCam"));
    assert_ne!(rust_compress::exif::orientation(exif), Some(6));
    assert_eq!(f.xmp.as_deref(), Some(&b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">hello</x:xmpmeta>"[..]));
    let m = f.metadata();
    assert_eq!((m.icc.as_deref(), m.exif.is_some(), m.xmp.is_some()), (Some(&icc[..]), true, true));
}

#[test]
fn ffmpeg_file() {
    let f = container::parse(&fixture("avif_ffmpeg.avif")).unwrap();
    assert_eq!((f.primary.width, f.primary.height), (96, 64));
    let c = coded(&f.primary);
    assert_eq!((c.data.len(), crc32(&c.data)), (601, 0x2d303eee));
    // libaom through ffmpeg codes 4:4:4 and signals it in av1C.
    assert_eq!((c.av1c.chroma_subsampling_x, c.av1c.chroma_subsampling_y, c.av1c.monochrome), (0, 0, false));
    let red = container::parse(&fixture("avif_red_ffmpeg.avif")).unwrap();
    assert_eq!((red.primary.width, red.primary.height), (64, 64));
    assert_eq!((coded(&red.primary).data.len(), crc32(&coded(&red.primary).data)), (34, 0x0f6a5f3f));
}

#[test]
fn grid_of_four_tiles_cropped() {
    let f = container::parse(&fixture("avif_grid.avif")).unwrap();
    assert_eq!((f.primary.width, f.primary.height), (120, 100));
    let ImageData::Grid(g) = &f.primary.data else { panic!("not a grid") };
    assert_eq!((g.version, g.flags, g.rows, g.columns, g.output_width, g.output_height), (0, 0, 2, 2, 120, 100));
    // Tile ids 1..=4 in row-major order, each with its own ispe/av1C and payload.
    let expect = [(178, 0xb62e6e37), (376, 0x48cdf2db), (404, 0x603344fb), (284, 0xf2f3d2ff)];
    assert_eq!(g.tiles.len(), 4);
    for (i, (t, (len, crc))) in g.tiles.iter().zip(expect).enumerate() {
        assert_eq!(t.item_id, i as u32 + 1);
        assert_eq!(t.ispe, Some((64, 64)));
        assert_eq!((t.data.len(), crc32(&t.data)), (len, crc));
        assert_eq!(t.av1c.to_bytes(), [0x81, 0x00, 0x0c, 0x00]);
    }
    assert_eq!(f.primary.av1_config(), &g.tiles[0].av1c);
}

#[test]
fn animated_file_yields_its_still_primary_item() {
    let bytes = fixture("avif_anim.avif");
    let f = container::parse(&bytes).unwrap();
    assert!(f.is_sequence);
    assert_eq!((f.primary.width, f.primary.height), (32, 32));
    assert!(matches!(f.primary.data, ImageData::Coded(_)));
}

// ----------------------------------------------------------------------------------------------------------
// Writer
// ----------------------------------------------------------------------------------------------------------

/// The 64x64 red still made by `ffmpeg -f lavfi -i color=c=red:s=64x64 -frames:v 1 -c:v libaom-av1 -f obu`:
/// a temporal delimiter (`12 00`) followed by sequence header and frame.
fn red_stream() -> Av1Stream {
    red_stream_and_colour().0
}

fn red_stream_and_colour() -> (Av1Stream, ColourInfo) {
    let obu = fixture("av1_red_64.obu");
    assert_eq!(&obu[..2], &[0x12, 0x00], "temporal delimiter");
    // The same stream inside the AVIF that ffmpeg's own muxer made: identical bytes without the delimiter.
    let reference = container::parse(&fixture("avif_red_ffmpeg.avif")).unwrap();
    let rc = coded(&reference.primary);
    assert_eq!(rc.data, &obu[2..]);
    (Av1Stream { data: obu[2..].to_vec(), config: rc.av1c.clone() }, reference.primary.colour.clone())
}

/// The file Pillow (libavif 1.4.2) and ffmpeg were checked against; see `written_red_is_stable`.
fn written_red() -> Vec<u8> {
    let (stream, colour) = red_stream_and_colour();
    let mut o = AvifOutput::new(64, 64, stream);
    o.colour = colour;
    container::write(&o)
}

#[test]
fn written_red_is_stable() {
    // Verified offline: Pillow 12.3 / libavif 1.4.2 opens this as a 64x64 RGB image, every pixel identical to
    // what it decodes from ffmpeg's own avif_red_ffmpeg.avif, and ffmpeg decodes it to the same frame too.
    let out = written_red();
    if let Ok(dir) = std::env::var("AVIF_DUMP_DIR") {
        std::fs::write(format!("{dir}/written_red.avif"), &out).unwrap();
    }
    assert_eq!(out, fixture("avif_written_red.avif"), "writer output changed; re-verify with Pillow and ffmpeg");
    let f = container::parse(&out).unwrap();
    assert_eq!((f.primary.width, f.primary.height), (64, 64));
    assert_eq!(coded(&f.primary).data, red_stream().data);
}

#[test]
fn written_full_is_stable() {
    // Colour + alpha payloads of avif_rgba.avif with every kind of metadata, rotation and a clean aperture-free
    // layout. Pillow decodes it to the same RGBA pixels as avif_rgba.avif (verified offline), with the ICC
    // profile, EXIF and XMP exposed in `Image.info`.
    let src = container::parse(&fixture("avif_rgba.avif")).unwrap();
    let meta = container::parse(&fixture("avif_meta.avif")).unwrap();
    let alpha = src.alpha.as_ref().unwrap();
    let mut o = AvifOutput::new(
        64,
        48,
        Av1Stream { data: coded(&src.primary).data.clone(), config: coded(&src.primary).av1c.clone() },
    );
    o.alpha = Some(Av1Stream { data: coded(&alpha.item).data.clone(), config: coded(&alpha.item).av1c.clone() });
    o.colour = ColourInfo { nclx: src.primary.colour.nclx, icc: meta.primary.colour.icc.clone() };
    o.exif = meta.exif.clone();
    o.xmp = meta.xmp.clone();
    let out = container::write(&o);
    if let Ok(dir) = std::env::var("AVIF_DUMP_DIR") {
        std::fs::write(format!("{dir}/written_full.avif"), &out).unwrap();
    }
    assert_eq!(out, fixture("avif_written_full.avif"), "writer output changed; re-verify with Pillow and ffmpeg");
    let f = container::parse(&out).unwrap();
    assert_eq!(f.metadata(), meta.metadata());
    assert_eq!(f.alpha.as_ref().unwrap().item.pixi.as_deref(), Some(&[8u8][..]));
}

#[test]
fn write_parse_roundtrip_all_options() {
    let red = red_stream();
    let mut mono = red.clone();
    mono.config.monochrome = true;
    mono.config.config_obus = vec![1, 2, 3];
    let mut o = AvifOutput::new(64, 64, red.clone());
    o.alpha = Some(mono.clone());
    o.alpha_premultiplied = true;
    o.colour = ColourInfo { nclx: Some(Nclx::SRGB), icc: Some((0..=255).collect()) };
    let clap = Clap {
        width_n: 60,
        width_d: 1,
        height_n: 50,
        height_d: 1,
        horiz_off_n: -1,
        horiz_off_d: 2,
        vert_off_n: 3,
        vert_off_d: 4,
    };
    o.clap = Some(clap);
    o.rotation = 3;
    o.mirror = Some(0);
    o.exif = Some(b"Exif\0\0MM\0*\0\0\0\x08\0\0".to_vec());
    o.xmp = Some(b"<?xpacket?>".to_vec());
    let f = container::parse(&container::write(&o)).unwrap();
    assert_eq!(f.primary.transforms, vec![Transform::Clap(clap), Transform::Rotate(3), Transform::Mirror(0)]);
    assert_eq!(f.primary.clap(), Some(&clap));
    assert_eq!(f.primary.colour, o.colour);
    assert_eq!(coded(&f.primary).data, red.data);
    assert_eq!(coded(&f.primary).av1c, red.config);
    let a = f.alpha.unwrap();
    assert!(a.premultiplied);
    assert_eq!(coded(&a.item).av1c, mono.config);
    assert_eq!(coded(&a.item).data, mono.data);
    // The Exif\0\0 prefix is dropped in favour of the bare TIFF block.
    assert_eq!(f.exif.as_deref(), Some(&b"MM\0*\0\0\0\x08\0\0"[..]));
    assert_eq!(f.xmp.as_deref(), Some(&b"<?xpacket?>"[..]));
}

#[test]
fn writer_shares_identical_properties_and_places_exif_prefix() {
    // A monochrome colour image and its alpha plane have identical ispe, pixi and av1C: one property each.
    let mut s = red_stream();
    s.config.monochrome = true;
    let mut o = AvifOutput::new(64, 64, s.clone());
    o.alpha = Some(s);
    o.exif = Some(b"II*\0\x08\0\0\0".to_vec());
    let out = container::write(&o);
    let count = |needle: &[u8]| out.windows(needle.len()).filter(|w| *w == needle).count();
    assert_eq!(count(b"ispe"), 1);
    assert_eq!(count(b"pixi"), 1);
    assert_eq!(count(b"av1C"), 1);
    // exif_tiff_header_offset = 0, then the TIFF block, in the mdat.
    assert_eq!(count(b"\0\0\0\0II*\0\x08\0\0\0"), 1);
    assert!(container::parse(&out).unwrap().alpha.is_some());
}

// ----------------------------------------------------------------------------------------------------------
// Box-level variants the real-world files do not exercise
// ----------------------------------------------------------------------------------------------------------

fn be(v: u64, n: usize) -> Vec<u8> {
    v.to_be_bytes()[8 - n..].to_vec()
}

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut o = (body.len() as u32 + 8).to_be_bytes().to_vec();
    o.extend_from_slice(kind);
    o.extend_from_slice(body);
    o
}

fn fbx(kind: &[u8; 4], version: u8, body: &[u8]) -> Vec<u8> {
    let mut b = vec![version, 0, 0, 0];
    b.extend_from_slice(body);
    bx(kind, &b)
}

#[derive(Clone, Copy)]
struct Synth {
    iloc_version: u8,
    /// Width of the offset and length fields (4 or 8).
    field: usize,
    base_offset: bool,
    extent_index: bool,
    idat: bool,
    split: bool,
    /// 0 = plain, 1 = 64-bit size, 2 = size 0 (to end of file).
    mdat_kind: u8,
    infe_version: u8,
    pitm_version: u8,
    ipma_wide: bool,
    junk_boxes: bool,
    /// An extra property box (type, essential) appended to `ipco` and associated with the item.
    extra_prop: Option<(&'static [u8; 4], bool)>,
}

const BASE: Synth = Synth {
    iloc_version: 0,
    field: 4,
    base_offset: false,
    extent_index: false,
    idat: false,
    split: false,
    mdat_kind: 0,
    infe_version: 2,
    pitm_version: 0,
    ipma_wide: false,
    junk_boxes: false,
    extra_prop: None,
};

/// A one-item AVIF whose item bytes are `payload`, laid out as `s` says.
fn synth(s: Synth, payload: &[u8]) -> Vec<u8> {
    let build = |data_start: u64| -> Vec<u8> {
        let mut m = fbx(b"hdlr", 0, &[&[0u8; 4][..], b"pict", &[0u8; 13]].concat());
        m.extend(fbx(b"pitm", s.pitm_version, &if s.pitm_version == 0 { be(1, 2) } else { be(1, 4) }));
        let mut infe = if s.infe_version >= 3 { be(1, 4) } else { be(1, 2) };
        infe.extend([0, 0]);
        infe.extend(b"av01\0");
        m.extend(fbx(b"iinf", 0, &[be(1, 2), fbx(b"infe", s.infe_version, &infe)].concat()));

        let v = s.iloc_version;
        let mut iloc = vec![(s.field as u8) << 4 | s.field as u8];
        let bsz = if s.base_offset { s.field as u8 } else { 0 };
        let isz = if v >= 1 && s.extent_index { 4u8 } else { 0 };
        iloc.push(bsz << 4 | isz);
        iloc.extend(if v < 2 { be(1, 2) } else { be(1, 4) });
        iloc.extend(if v < 2 { be(1, 2) } else { be(1, 4) }); // item id
        if v >= 1 {
            iloc.extend(be(s.idat as u64, 2)); // construction method
        }
        iloc.extend(be(0, 2)); // data reference index
        let n = payload.len() as u64;
        let pieces: Vec<(u64, u64)> = if s.split { vec![(0, 7), (7, n - 7)] } else { vec![(0, n)] };
        let abs = if s.idat { 0 } else { data_start };
        if s.base_offset {
            iloc.extend(be(abs, s.field));
        }
        iloc.extend(be(pieces.len() as u64, 2));
        for (off, len) in pieces {
            if isz > 0 {
                iloc.extend(be(0, 4));
            }
            iloc.extend(be(if s.base_offset { off } else { abs + off }, s.field));
            iloc.extend(be(len, s.field));
        }
        m.extend(fbx(b"iloc", v, &iloc));

        let mut props =
            vec![fbx(b"ispe", 0, &[be(16, 4), be(16, 4)].concat()), bx(b"av1C", &Av1Config::default().to_bytes())];
        let mut assoc = vec![(1u16, false), (2, true)];
        if let Some((kind, essential)) = s.extra_prop {
            props.push(bx(kind, &[0, 0, 0, 0]));
            assoc.push((3, essential));
        }
        let wide = s.ipma_wide;
        let mut ipma = vec![wide as u8, 0, 0, wide as u8];
        ipma.extend(be(1, 4));
        ipma.extend(be(1, if wide { 4 } else { 2 }));
        ipma.push(assoc.len() as u8);
        for (idx, essential) in assoc {
            if wide {
                ipma.extend(be((idx | (essential as u16) << 15) as u64, 2));
            } else {
                ipma.push(idx as u8 | (essential as u8) << 7);
            }
        }
        m.extend(bx(b"iprp", &[bx(b"ipco", &props.concat()), bx(b"ipma", &ipma)].concat()));
        if s.idat {
            m.extend(bx(b"idat", payload));
        }
        if s.junk_boxes {
            m.extend(bx(b"free", &[0; 9]));
            m.extend(bx(b"unkn", &[1, 2, 3]));
        }
        fbx(b"meta", 0, &m)
    };
    let mut head = bx(b"ftyp", b"avif\0\0\0\0mif1avif");
    if s.junk_boxes {
        head.extend(bx(b"free", &[7; 20]));
        head.extend(bx(b"skip", &[]));
        let mut u = 36u32.to_be_bytes().to_vec();
        u.extend(b"uuid");
        u.extend([0xAA; 16]);
        u.extend([1; 12]);
        head.extend(u);
    }
    let mdat_hdr = match s.mdat_kind {
        1 => 16,
        _ => 8,
    };
    let meta_len = build(0).len();
    // For idat construction nothing points into mdat, but we still write one (with the payload, unused).
    let start = (head.len() + meta_len + mdat_hdr) as u64;
    let mut out = head;
    out.extend(build(start));
    match s.mdat_kind {
        0 => out.extend((payload.len() as u32 + 8).to_be_bytes()),
        1 => {
            out.extend(1u32.to_be_bytes());
            out.extend(b"mdat");
            out.extend((payload.len() as u64 + 16).to_be_bytes());
        }
        _ => out.extend(0u32.to_be_bytes()),
    }
    if s.mdat_kind != 1 {
        out.extend(b"mdat");
    }
    out.extend(payload);
    if s.junk_boxes && s.mdat_kind == 0 {
        out.extend(bx(b"free", &[0; 3]));
    }
    out
}

#[test]
fn iloc_and_box_variants() {
    let payload: Vec<u8> = (0..40u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
    let mut n = 0;
    for iloc_version in 0..=2u8 {
        for field in [4usize, 8] {
            for base_offset in [false, true] {
                for idat in [false, true] {
                    if idat && iloc_version == 0 {
                        continue;
                    }
                    for split in [false, true] {
                        let s = Synth {
                            iloc_version,
                            field,
                            base_offset,
                            idat,
                            split,
                            extent_index: iloc_version > 0 && split,
                            mdat_kind: (n % 3) as u8,
                            infe_version: if iloc_version == 2 { 3 } else { 2 },
                            pitm_version: (n % 2) as u8,
                            ipma_wide: n % 2 == 1,
                            junk_boxes: n % 4 < 2,
                            extra_prop: (n % 5 == 0).then_some((b"zzzz", false)),
                        };
                        n += 1;
                        let f = container::parse(&synth(s, &payload)).unwrap_or_else(|e| panic!("case {n}: {e}"));
                        assert_eq!(coded(&f.primary).data, payload, "case {n}");
                        assert_eq!((f.primary.width, f.primary.height), (16, 16));
                    }
                }
            }
        }
    }
    assert!(n >= 20);
}

#[test]
fn unknown_essential_property_is_unsupported_but_optional_ones_are_skipped() {
    let payload = [1u8; 12];
    let with = |kind: &'static [u8; 4], essential: bool| {
        synth(Synth { extra_prop: Some((kind, essential)), ..BASE }, &payload)
    };
    assert!(container::parse(&with(b"zzzz", false)).is_ok());
    assert!(matches!(container::parse(&with(b"zzzz", true)), Err(Error::Unsupported(_))));
    // Known-ignorable properties stay harmless even when essential.
    for k in [b"a1op", b"lsel", b"a1lx", b"clli"] {
        assert!(container::parse(&with(k, true)).is_ok(), "{}", String::from_utf8_lossy(k));
    }
}

#[test]
fn brands_and_sequences() {
    let ok = synth(BASE, &[1; 8]);
    assert!(container::parse(&ok).is_ok());
    // No AVIF-compatible brand.
    let mut heic = ok.clone();
    heic[8..12].copy_from_slice(b"heic");
    heic[16..20].copy_from_slice(b"heic");
    heic[20..24].copy_from_slice(b"heic");
    assert!(matches!(container::parse(&heic), Err(Error::Unsupported(_))));
    // An avis file with a moov but no meta is unsupported; without ftyp it is invalid.
    let mut seq = bx(b"ftyp", b"avis\0\0\0\0avismsf1");
    seq.extend(bx(b"moov", &[0; 8]));
    assert!(matches!(container::parse(&seq), Err(Error::Unsupported(_))));
    assert!(matches!(container::parse(&seq[20..]), Err(Error::Invalid(_))));
}

#[test]
fn too_many_grid_tiles_and_items_are_refused() {
    let mut d = fixture("avif_grid.avif");
    // Grid descriptor is the last (8-byte) extent of the mdat: version, flags, rows-1, columns-1, w, h.
    let n = d.len();
    d[n - 6] = 255;
    d[n - 5] = 255;
    assert!(container::parse(&d).is_err(), "256x256 tiles but 4 references");
    // 4096 items declared in iinf with the real ones missing: refused, never allocated.
    let mut d = fixture("avif_rgb.avif");
    let at = d.windows(4).position(|w| w == b"iinf").unwrap() + 8;
    d[at..at + 2].copy_from_slice(&5000u16.to_be_bytes());
    assert!(container::parse(&d).is_err());
}

// ----------------------------------------------------------------------------------------------------------
// Hostile input
// ----------------------------------------------------------------------------------------------------------

/// Small deterministic generator so failures are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn exercise(bytes: &[u8]) {
    if let Ok(f) = container::parse(bytes) {
        // Whatever parses must re-write into something that parses back to the same payload.
        if let ImageData::Coded(c) = &f.primary.data {
            let mut o = AvifOutput::new(
                f.primary.width,
                f.primary.height,
                Av1Stream { data: c.data.clone(), config: c.av1c.clone() },
            );
            o.colour = f.primary.colour.clone();
            o.exif = f.exif.clone();
            o.xmp = f.xmp.clone();
            let back = container::parse(&container::write(&o)).expect("own output parses");
            assert_eq!(coded(&back.primary).data, c.data);
            assert_eq!(back.exif, f.exif);
            assert_eq!(back.xmp, f.xmp);
        }
    }
}

fn hammer(name: &str, original: &[u8], failures: &mut Vec<String>) {
    let mut cases: Vec<(String, Vec<u8>)> = Vec::new();
    for cut in (0..original.len().min(120)).chain((0..64).map(|i| original.len() * i / 64)) {
        cases.push((format!("truncate@{cut}"), original[..cut].to_vec()));
    }
    let mut rng = Rng(0xa71f ^ original.len() as u64);
    // HOSTILE_ROUNDS=20000 cargo test --release --test avif_container  for a deeper (slower) sweep.
    let rounds = std::env::var("HOSTILE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1500);
    for i in 0..rounds {
        let mut m = original.to_vec();
        for _ in 0..1 + i % 4 {
            // Most damage lands in the box headers and tables at the front, where the structure lives.
            let at = if i % 3 != 0 { rng.below(m.len().min(original.len().min(700))) } else { rng.below(m.len()) };
            m[at] = match rng.below(4) {
                0 => 0x00,
                1 => 0xFF,
                2 => m[at].wrapping_add(1),
                _ => rng.next() as u8,
            };
        }
        cases.push((format!("flip#{i}"), m));
    }
    let mut tail = original.to_vec();
    tail.extend((0..4096).map(|_| rng.next() as u8));
    cases.push(("trailing-garbage".into(), tail));
    for (label, bytes) in cases {
        let start = Instant::now();
        let r = catch_unwind(AssertUnwindSafe(|| exercise(&bytes)));
        if r.is_err() {
            failures.push(format!("{name} {label}: panicked"));
        } else if start.elapsed() > Duration::from_secs(10) {
            failures.push(format!("{name} {label}: took {:?}", start.elapsed()));
        }
    }
}

#[test]
fn damaged_avifs_never_panic() {
    let mut failures = Vec::new();
    for name in [
        "avif_rgb.avif",
        "avif_rgba.avif",
        "avif_meta.avif",
        "avif_ffmpeg.avif",
        "avif_grid.avif",
        "avif_anim.avif",
        "avif_written_red.avif",
        "avif_written_full.avif",
    ] {
        hammer(name, &fixture(name), &mut failures);
    }
    let payload = [5u8; 33];
    for (i, s) in [
        BASE,
        Synth { iloc_version: 2, field: 8, base_offset: true, infe_version: 3, pitm_version: 1, ..BASE },
        Synth { iloc_version: 1, idat: true, extent_index: true, split: true, junk_boxes: true, ..BASE },
        Synth { mdat_kind: 1, ipma_wide: true, ..BASE },
    ]
    .into_iter()
    .enumerate()
    {
        hammer(&format!("synth{i}"), &synth(s, &payload), &mut failures);
    }
    for f in failures.iter().take(20) {
        eprintln!("{f}");
    }
    assert!(failures.is_empty(), "{} hostile-input failures (first shown above)", failures.len());
}

#[test]
fn absurd_sizes_and_counts_are_rejected_cheaply() {
    let start = Instant::now();
    let base = fixture("avif_rgba.avif");
    let find = |d: &[u8], k: &[u8; 4]| d.windows(4).position(|w| w == k).unwrap();
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();

    // Top-level and nested box sizes beyond the data, including a 64-bit size of 2^63.
    let mut d = base.clone();
    let m = find(&d, b"meta") - 4;
    d[m..m + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    cases.push(("meta size u32::MAX", d));
    let mut d = base.clone();
    let m = find(&d, b"meta") - 4;
    d[m..m + 4].copy_from_slice(&1u32.to_be_bytes());
    d[m + 8..m + 16].copy_from_slice(&(1u64 << 63).to_be_bytes());
    cases.push(("meta size 2^63", d));
    let mut d = base.clone();
    let m = find(&d, b"iloc") - 4;
    d[m..m + 4].copy_from_slice(&7u32.to_be_bytes());
    cases.push(("iloc size 7", d));

    // iloc: huge counts, offsets and lengths.
    let iloc = find(&base, b"iloc");
    // After type: version/flags(4), sizes(2), item_count(2), then item 1: id, dri, extent_count, extent...
    let mut d = base.clone();
    d[iloc + 10..iloc + 12].copy_from_slice(&0xFFFFu16.to_be_bytes());
    cases.push(("iloc item_count 65535", d));
    let mut d = base.clone();
    d[iloc + 16..iloc + 18].copy_from_slice(&0xFFFFu16.to_be_bytes());
    cases.push(("iloc extent_count 65535", d));
    let mut d = base.clone();
    d[iloc + 18..iloc + 22].copy_from_slice(&u32::MAX.to_be_bytes());
    cases.push(("iloc offset u32::MAX", d));
    let mut d = base.clone();
    d[iloc + 22..iloc + 26].copy_from_slice(&u32::MAX.to_be_bytes());
    cases.push(("iloc length u32::MAX", d));
    let mut d = base.clone();
    d[iloc + 8] = 0x88;
    cases.push(("iloc 8-byte fields misread", d));

    // iinf / ipma / iref counts.
    let mut d = base.clone();
    let p = find(&d, b"iinf") + 8;
    d[p..p + 2].copy_from_slice(&0xFFFFu16.to_be_bytes());
    cases.push(("iinf count 65535", d));
    let mut d = base.clone();
    let p = find(&d, b"ipma") + 8;
    d[p..p + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    cases.push(("ipma entry_count u32::MAX", d));
    let mut d = base.clone();
    let p = find(&d, b"iref") + 12;
    d[p + 4..p + 6].copy_from_slice(&0xFFFFu16.to_be_bytes());
    cases.push(("iref reference count 65535", d));

    // Absurd image size in ispe.
    for (w, h) in [(0u32, 0u32), (u32::MAX, u32::MAX), (1, u32::MAX), (100_000, 100_000)] {
        let mut d = base.clone();
        let p = find(&d, b"ispe") + 8;
        d[p..p + 4].copy_from_slice(&w.to_be_bytes());
        d[p + 4..p + 8].copy_from_slice(&h.to_be_bytes());
        cases.push(("ispe size", d));
    }

    for (label, bytes) in cases {
        let r = catch_unwind(AssertUnwindSafe(|| container::parse(&bytes)));
        assert!(r.is_ok(), "{label}: panicked");
        // Every one of these is damage; the parser must not accept the file as it was.
        let accepted = r.unwrap().ok();
        if let Some(f) = accepted {
            assert!(f.primary.width < 100_000, "{label}: accepted with a hostile size");
        }
    }
    assert!(start.elapsed() < Duration::from_secs(5), "rejections took {:?}", start.elapsed());
}

#[test]
fn repeated_grid_tile_reference_is_budgeted() {
    // A 255x16 grid (4080 tiles) in which every tile is the same item: the data is shared in the file, but
    // the copied total is limited to a small multiple of the file length.
    let tile_payload = vec![9u8; 4000];
    let tiles = 255 * 16;
    let mut infe = Vec::new();
    let mut dimg = Vec::new();
    let ids: Vec<u16> = (1..=tiles as u16 + 1).collect();
    // Item 1 is the only real tile; the grid (id 2) lists item 1 over and over.
    for &id in &ids[..2] {
        let kind: &[u8; 4] = if id == 1 { b"av01" } else { b"grid" };
        let mut b = be(id as u64, 2);
        b.extend([0, 0]);
        b.extend(kind);
        b.push(0);
        infe.extend(fbx(b"infe", 2, &b));
    }
    for _ in 0..tiles {
        dimg.extend(be(1, 2));
    }
    let mut dimg_box = be(2, 2);
    dimg_box.extend(be(tiles as u64, 2));
    dimg_box.extend(&dimg);
    let iref = fbx(b"iref", 0, &bx(b"dimg", &dimg_box));
    let ispe = fbx(b"ispe", 0, &[be(16, 4), be(16, 4)].concat());
    let ispe_g = fbx(b"ispe", 0, &[be(255 * 16, 4), be(16 * 16, 4)].concat());
    let av1c = bx(b"av1C", &Av1Config::default().to_bytes());
    let ipma = fbx(b"ipma", 0, &[be(2, 4), be(1, 2), vec![2, 1, 0x82], be(2, 2), vec![1, 3]].concat());
    let iprp = bx(b"iprp", &[bx(b"ipco", &[ispe, av1c, ispe_g].concat()), ipma].concat());
    let mut gdata = vec![0u8, 1, 15, 254];
    gdata.extend(be(255 * 16, 4));
    gdata.extend(be(16 * 16, 4));
    let meta = |t0: u32, g0: u32| {
        let mut iloc = vec![0x44, 0];
        iloc.extend(be(2, 2));
        iloc.extend([be(1, 2), be(0, 2), be(1, 2), be(t0 as u64, 4), be(tile_payload.len() as u64, 4)].concat());
        iloc.extend([be(2, 2), be(0, 2), be(1, 2), be(g0 as u64, 4), be(gdata.len() as u64, 4)].concat());
        let mut m = fbx(b"hdlr", 0, &[&[0u8; 4][..], b"pict", &[0u8; 13]].concat());
        m.extend(fbx(b"pitm", 0, &be(2, 2)));
        m.extend(fbx(b"iinf", 0, &[be(2, 2), infe.clone()].concat()));
        m.extend(fbx(b"iloc", 0, &iloc));
        m.extend(iref.clone());
        m.extend(iprp.clone());
        fbx(b"meta", 0, &m)
    };
    let ftyp = bx(b"ftyp", b"avif\0\0\0\0avif");
    let ml = meta(0, 0).len();
    let t0 = (ftyp.len() + ml + 8) as u32;
    let mut file = ftyp;
    file.extend(meta(t0, t0 + tile_payload.len() as u32));
    file.extend(bx(b"mdat", &[tile_payload.clone(), gdata.clone()].concat()));
    let start = Instant::now();
    let r = container::parse(&file);
    // 4080 copies of 4000 bytes = 16 MB against a 20 KB file: over the budget either way, never an OOM.
    assert!(r.is_err(), "budget not enforced");
    assert!(start.elapsed() < Duration::from_secs(5));
}
