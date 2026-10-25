//! The ISOBMFF / HEIF container as used by AVIF (ISO/IEC 14496-12, 23008-12 and the AVIF specification):
//! a defensive parser and a minimal, conforming writer. Only boxes and item bookkeeping live here; the AV1
//! payload (a sequence header plus one frame as an OBU stream) is opaque bytes.
//!
//! # Parser scope
//!
//! * `ftyp` must list an `avif`, `avis`, `mif1` or `miaf` brand (major or compatible).
//! * Animated `avis` sequences: only the still primary item of the `meta` box is returned (libavif writes one
//!   next to the `moov`); [`AvifFile::is_sequence`] says the file also carries a sequence. A sequence without a
//!   primary item is [`Error::Unsupported`]. The `moov`/`trak` boxes are never interpreted.
//! * `meta`: `hdlr` (must be `pict` when present), `pitm` (v0/v1), `iinf`/`infe` (v0-v3), `iref`, `iprp`
//!   (`ipco` + `ipma` v0/v1, 7 or 15 bit indices), `iloc` (v0-v2, field sizes 0/4/8, construction method 0 =
//!   file offset and 1 = `idat`, any number of extents) and `idat`.
//! * Item types: `av01` (coded image), `grid` (derived), `iden` (identity: re-uses another image item and
//!   adds its own transformations), `Exif`, `mime` with `application/rdf+xml` (XMP). Anything else is ignored
//!   unless it is the primary item (then [`Error::Unsupported`]).
//! * Properties understood: `ispe`, `pixi`, `av1C`, `colr` (`nclx`, `prof`, `rICC`), `irot`, `imir`, `clap`,
//!   `pasp`, `auxC`. `a1op`, `lsel`, `a1lx`, `clli`, `mdcv`, `cclv`, `amve`, `altt`, `cmin`, `cmex`, `rloc`
//!   and `udes` are skipped even when marked essential. Any other *essential* property makes the item (and so,
//!   if needed, the file) unsupported; non-essential unknown properties are skipped.
//! * Not supported: items in external files (`data_reference_index != 0`), `iloc` construction method 2, items
//!   with a content encoding, encrypted/protected items, `iden`/`grid` nesting deeper than a few levels.
//!
//! # Limits
//!
//! At most [`MAX_ITEMS`] items, [`MAX_GRID_TILES`] grid tiles, 65536 properties (256 associations per item); 65536 `iloc`
//! extents in total; images larger than [`MAX_PIXELS`] are rejected. Nothing is allocated from a count read
//! out of the file without checking it against the bytes that remain, every box size and offset is checked
//! against the input, and the total bytes copied out as item data are capped at four times the file length
//! plus 1 MiB (an item can legitimately be referenced twice, e.g. a repeated grid tile, but not a million times).
//! There is no recursion on box nesting: the nesting depth the parser walks is fixed.
//!
//! # Conventions
//!
//! * EXIF is returned as the bare TIFF block (the `exif_tiff_header_offset` field and the bytes it skips,
//!   usually an `Exif\0\0` prefix, are removed), like the rest of the crate. XMP is the raw packet.
//! * Transformations are returned in property order. [`net_orientation`] folds `irot`/`imir` into one flip +
//!   rotation. `imir` axis 0 is a mirror about the vertical axis (left and right swap), axis 1 about the
//!   horizontal axis (top and bottom swap), as in ISO/IEC 23008-12:2022; HEIF's first edition was ambiguous
//!   about this. A `clap` crop is reported in the list but not folded into the orientation.
//! * The `prem` reference is accepted in either direction between the colour and alpha items (libavif writes
//!   it from the alpha item to the colour item, which is also what [`write`] does).

use super::sequence::TrackOut;
use rc_core::error::{Error, Result};
use rc_core::image::{MAX_PIXELS, Metadata};
use std::cell::Cell;
use std::collections::HashMap;

pub const MAX_ITEMS: usize = 4096;
pub const MAX_GRID_TILES: usize = 4096;
const MAX_PROPERTIES: usize = 1 << 16;
const MAX_EXTENTS: usize = 1 << 16;
const MAX_DERIVATION_DEPTH: u32 = 4;

pub type FourCc = [u8; 4];

const ALPHA_URNS: [&str; 2] = ["urn:mpeg:mpegB:cicp:systems:auxiliary:alpha", "urn:mpeg:avif:alpha"];
/// Properties that may be marked essential without making an item unusable for us.
const IGNORABLE: [&FourCc; 12] =
    [b"a1op", b"lsel", b"a1lx", b"clli", b"mdcv", b"cclv", b"amve", b"altt", b"cmin", b"cmex", b"rloc", b"udes"];

// ---------------------------------------------------------------------------------------------------------
// Public data model
// ---------------------------------------------------------------------------------------------------------

/// The `av1C` box (AV1 codec configuration record).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1Config {
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: u8,
    pub chroma_subsampling_y: u8,
    pub chroma_sample_position: u8,
    /// `initial_presentation_delay_minus_one` when `initial_presentation_delay_present` is set.
    pub initial_presentation_delay_minus_one: Option<u8>,
    /// Optional configOBUs (normally the sequence header OBU).
    pub config_obus: Vec<u8>,
}

impl Default for Av1Config {
    /// 8-bit 4:2:0, profile 0, level 2.0 (index 0), no configOBUs.
    fn default() -> Self {
        Av1Config {
            seq_profile: 0,
            seq_level_idx_0: 0,
            seq_tier_0: 0,
            high_bitdepth: false,
            twelve_bit: false,
            monochrome: false,
            chroma_subsampling_x: 1,
            chroma_subsampling_y: 1,
            chroma_sample_position: 0,
            initial_presentation_delay_minus_one: None,
            config_obus: Vec::new(),
        }
    }
}

impl Av1Config {
    pub fn parse(b: &[u8]) -> Result<Self> {
        if b.len() < 4 {
            return Err(Error::invalid("av1C: too short"));
        }
        if b[0] != 0x81 {
            return Err(Error::invalid("av1C: bad marker/version"));
        }
        Ok(Av1Config {
            seq_profile: b[1] >> 5,
            seq_level_idx_0: b[1] & 31,
            seq_tier_0: b[2] >> 7,
            high_bitdepth: b[2] & 0x40 != 0,
            twelve_bit: b[2] & 0x20 != 0,
            monochrome: b[2] & 0x10 != 0,
            chroma_subsampling_x: (b[2] >> 3) & 1,
            chroma_subsampling_y: (b[2] >> 2) & 1,
            chroma_sample_position: b[2] & 3,
            initial_presentation_delay_minus_one: (b[3] & 0x10 != 0).then_some(b[3] & 15),
            config_obus: b[4..].to_vec(),
        })
    }

    /// The box body (without the box header).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = vec![0x81, (self.seq_profile & 7) << 5 | (self.seq_level_idx_0 & 31)];
        o.push(
            (self.seq_tier_0 & 1) << 7
                | (self.high_bitdepth as u8) << 6
                | (self.twelve_bit as u8) << 5
                | (self.monochrome as u8) << 4
                | (self.chroma_subsampling_x & 1) << 3
                | (self.chroma_subsampling_y & 1) << 2
                | (self.chroma_sample_position & 3),
        );
        o.push(match self.initial_presentation_delay_minus_one {
            Some(d) => 0x10 | (d & 15),
            None => 0,
        });
        o.extend_from_slice(&self.config_obus);
        o
    }

    pub fn bit_depth(&self) -> u8 {
        match (self.high_bitdepth, self.twelve_bit) {
            (_, true) => 12,
            (true, false) => 10,
            _ => 8,
        }
    }

    /// Colour channels in the coded picture (1 for monochrome, else 3).
    pub fn channels(&self) -> u8 {
        if self.monochrome { 1 } else { 3 }
    }
}

/// `colr` with colour type `nclx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nclx {
    pub colour_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    pub full_range: bool,
}

impl Nclx {
    /// sRGB primaries and transfer, BT.601 matrix, full range (what libavif writes for plain sRGB).
    pub const SRGB: Nclx =
        Nclx { colour_primaries: 1, transfer_characteristics: 13, matrix_coefficients: 6, full_range: true };
}

/// Colour description of an image item.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColourInfo {
    pub nclx: Option<Nclx>,
    /// Raw ICC profile (`colr` `prof` or `rICC`).
    pub icc: Option<Vec<u8>>,
}

/// `clap` clean aperture: a fraction for each of width, height, horizontal and vertical offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clap {
    pub width_n: u32,
    pub width_d: u32,
    pub height_n: u32,
    pub height_d: u32,
    pub horiz_off_n: i32,
    pub horiz_off_d: u32,
    pub vert_off_n: i32,
    pub vert_off_d: u32,
}

/// A transformative property, in the order it appears on the item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    Clap(Clap),
    /// `irot`: anticlockwise quarter turns (0..=3).
    Rotate(u8),
    /// `imir` axis: 0 = about the vertical axis (left/right swap), 1 = about the horizontal axis (top/bottom swap).
    Mirror(u8),
}

/// The net effect of `irot`/`imir` properties: first mirror left/right if `mirror`, then rotate anticlockwise
/// by `rotate` quarter turns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Orientation {
    pub rotate: u8,
    pub mirror: bool,
}

impl Orientation {
    pub fn swaps_dimensions(self) -> bool {
        self.rotate & 1 == 1
    }
}

/// Fold the rotations and mirrors of `transforms` (applied in list order) into one [`Orientation`].
/// `Clap` entries are ignored.
pub fn net_orientation(transforms: &[Transform]) -> Orientation {
    // State: output = Rot^rotate(Flip^mirror(source)); a mirror conjugates the rotation (Rot -> Rot^-1).
    let (mut k, mut f) = (0u8, false);
    for t in transforms {
        match *t {
            Transform::Clap(_) => {}
            Transform::Rotate(r) => k = (k + (r & 3)) & 3,
            Transform::Mirror(axis) => {
                k = (4 - k) & 3;
                f = !f;
                if axis & 1 == 1 {
                    // A top/bottom flip is a left/right flip followed by a half turn.
                    k = (k + 2) & 3;
                }
            }
        }
    }
    Orientation { rotate: k, mirror: f }
}

/// One `av01` item: its properties and AV1 payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodedItem {
    pub item_id: u32,
    pub ispe: Option<(u32, u32)>,
    /// `pixi` bits per channel.
    pub pixi: Option<Vec<u8>>,
    pub av1c: Av1Config,
    /// The AV1 OBU stream (all extents concatenated).
    pub data: Vec<u8>,
}

/// A `grid` derived image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    pub version: u8,
    pub flags: u8,
    pub rows: u32,
    pub columns: u32,
    pub output_width: u32,
    pub output_height: u32,
    /// Row-major, `rows * columns` entries.
    pub tiles: Vec<CodedItem>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageData {
    Coded(CodedItem),
    Grid(Grid),
}

/// An image item (the primary or the alpha plane) with the properties that apply to the whole item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageItem {
    pub item_id: u32,
    /// `ispe` size (the grid output size when a grid has no `ispe`), before any transformation.
    pub width: u32,
    pub height: u32,
    pub pixi: Option<Vec<u8>>,
    pub colour: ColourInfo,
    /// `pasp` pixel aspect ratio (horizontal, vertical spacing).
    pub pasp: Option<(u32, u32)>,
    /// `clap`, `irot` and `imir` in property order.
    pub transforms: Vec<Transform>,
    pub data: ImageData,
}

impl ImageItem {
    /// The `av1C` of the item (of the first tile for a grid).
    pub fn av1_config(&self) -> &Av1Config {
        match &self.data {
            ImageData::Coded(c) => &c.av1c,
            ImageData::Grid(g) => &g.tiles[0].av1c,
        }
    }

    pub fn net_orientation(&self) -> Orientation {
        net_orientation(&self.transforms)
    }

    pub fn clap(&self) -> Option<&Clap> {
        self.transforms.iter().find_map(|t| if let Transform::Clap(c) = t { Some(c) } else { None })
    }
}

/// The auxiliary alpha image of the primary item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlphaItem {
    pub item: ImageItem,
    /// The colour image is premultiplied by this alpha (`prem` reference).
    pub premultiplied: bool,
    /// The `auxC` URN that identified it as alpha.
    pub aux_type: String,
}

/// A parsed AVIF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvifFile {
    pub major_brand: FourCc,
    pub compatible_brands: Vec<FourCc>,
    /// The file also contains an image sequence (`avis` / `moov`); only the still primary item is decoded.
    pub is_sequence: bool,
    pub primary: ImageItem,
    pub alpha: Option<AlphaItem>,
    /// Bare TIFF block.
    pub exif: Option<Vec<u8>>,
    /// XMP packet.
    pub xmp: Option<Vec<u8>>,
}

impl AvifFile {
    /// ICC profile of the primary item.
    pub fn icc(&self) -> Option<&[u8]> {
        self.primary.colour.icc.as_deref()
    }

    /// The crate's metadata record: ICC, EXIF and XMP.
    pub fn metadata(&self) -> Metadata {
        Metadata {
            icc: self.icc().map(<[u8]>::to_vec),
            exif: self.exif.clone(),
            xmp: self.xmp.clone(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Low-level reading
// ---------------------------------------------------------------------------------------------------------

pub(super) struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    pub(super) fn new(d: &'a [u8]) -> Self {
        Cur { d, p: 0 }
    }

    pub(super) fn left(&self) -> usize {
        self.d.len() - self.p
    }

    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.left() {
            return Err(Error::invalid("box truncated"));
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }

    pub(super) fn rest(&mut self) -> &'a [u8] {
        let s = &self.d[self.p..];
        self.p = self.d.len();
        s
    }

    pub(super) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub(super) fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(super) fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    /// A 0/4/8 byte unsigned field (iloc sizes).
    pub(super) fn uint(&mut self, size: u8) -> Result<u64> {
        match size {
            0 => Ok(0),
            4 => Ok(self.u32()? as u64),
            8 => self.u64(),
            _ => Err(Error::invalid("iloc: field size must be 0, 4 or 8")),
        }
    }

    pub(super) fn fourcc(&mut self) -> Result<FourCc> {
        let b = self.take(4)?;
        Ok([b[0], b[1], b[2], b[3]])
    }

    /// A NUL-terminated string; an unterminated one runs to the end of the box.
    pub(super) fn cstr(&mut self) -> &'a [u8] {
        let rest = &self.d[self.p..];
        match rest.iter().position(|&b| b == 0) {
            Some(n) => {
                self.p += n + 1;
                &rest[..n]
            }
            None => {
                self.p = self.d.len();
                rest
            }
        }
    }

    /// FullBox header: version and 24-bit flags.
    pub(super) fn full(&mut self) -> Result<(u8, u32)> {
        let v = self.u8()?;
        let b = self.take(3)?;
        Ok((v, (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32))
    }
}

/// Split `data` into boxes: (type, body). 64-bit sizes, size 0 (to the end of `data`) and `uuid` boxes are handled.
/// A trailing fragment shorter than a box header is ignored.
pub(super) fn split_boxes(data: &[u8]) -> Result<Vec<(FourCc, &[u8])>> {
    let mut out = Vec::new();
    let mut p = 0usize;
    while data.len() - p >= 8 {
        let rem = data.len() - p;
        let size32 = u32::from_be_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
        let kind = [data[p + 4], data[p + 5], data[p + 6], data[p + 7]];
        let mut hdr = 8usize;
        let size = match size32 {
            0 => rem,
            1 => {
                if rem < 16 {
                    return Err(Error::invalid("box: truncated 64-bit size"));
                }
                hdr = 16;
                let mut a = [0u8; 8];
                a.copy_from_slice(&data[p + 8..p + 16]);
                usize::try_from(u64::from_be_bytes(a)).map_err(|_| Error::invalid("box: size too large"))?
            }
            n => n as usize,
        };
        if &kind == b"uuid" {
            hdr += 16;
        }
        if size < hdr || size > rem {
            return Err(Error::invalid(format!("box '{}': size {size} out of range", show(&kind))));
        }
        out.push((kind, &data[p + hdr..p + size]));
        p += size;
    }
    Ok(out)
}

pub(super) fn show(k: &FourCc) -> String {
    k.iter().map(|&b| if (0x20..0x7f).contains(&b) { b as char } else { '?' }).collect()
}

// ---------------------------------------------------------------------------------------------------------
// meta box
// ---------------------------------------------------------------------------------------------------------

struct ItemInfo {
    id: u32,
    kind: FourCc,
    content_type: Vec<u8>,
    content_encoding: Vec<u8>,
    protected: bool,
}

#[derive(Clone, Copy)]
struct Extent {
    offset: u64,
    length: u64,
}

struct ItemLoc {
    construction_method: u8,
    data_reference_index: u16,
    base_offset: u64,
    extents: Vec<Extent>,
}

struct ItemRef {
    kind: FourCc,
    from: u32,
    to: Vec<u32>,
}

#[derive(Default)]
struct Meta<'a> {
    primary: Option<u32>,
    items: Vec<ItemInfo>,
    locs: HashMap<u32, ItemLoc>,
    refs: Vec<ItemRef>,
    props: Vec<(FourCc, &'a [u8])>,
    assoc: HashMap<u32, Vec<(usize, bool)>>,
    idat: Option<&'a [u8]>,
}

fn parse_meta(body: &[u8]) -> Result<Meta<'_>> {
    let mut c = Cur::new(body);
    c.full()?;
    let mut m = Meta::default();
    for (kind, b) in split_boxes(c.rest())? {
        match &kind {
            b"hdlr" => {
                let mut h = Cur::new(b);
                h.full()?;
                h.u32()?;
                if &h.fourcc()? != b"pict" {
                    return Err(Error::unsupported("meta handler is not 'pict'"));
                }
            }
            b"pitm" => {
                let mut h = Cur::new(b);
                let (v, _) = h.full()?;
                m.primary = Some(if v == 0 { h.u16()? as u32 } else { h.u32()? });
            }
            b"iinf" => m.items = parse_iinf(b)?,
            b"iloc" => m.locs = parse_iloc(b)?,
            b"iref" => m.refs = parse_iref(b)?,
            b"iprp" => parse_iprp(b, &mut m)?,
            b"idat" => m.idat = Some(b),
            _ => {}
        }
    }
    Ok(m)
}

fn parse_iinf(b: &[u8]) -> Result<Vec<ItemInfo>> {
    let mut c = Cur::new(b);
    let (v, _) = c.full()?;
    let count = if v == 0 { c.u16()? as u64 } else { c.u32()? as u64 };
    if count > MAX_ITEMS as u64 {
        return Err(Error::unsupported(format!("too many items ({count})")));
    }
    let mut items: Vec<ItemInfo> = Vec::new();
    for (kind, body) in split_boxes(c.rest())? {
        if &kind != b"infe" {
            continue;
        }
        if items.len() >= MAX_ITEMS {
            return Err(Error::unsupported("too many items"));
        }
        let it = parse_infe(body)?;
        if items.iter().any(|o| o.id == it.id) {
            return Err(Error::invalid(format!("duplicate item id {}", it.id)));
        }
        items.push(it);
    }
    Ok(items)
}

fn parse_infe(b: &[u8]) -> Result<ItemInfo> {
    let mut c = Cur::new(b);
    let (v, _) = c.full()?;
    let id = if v >= 3 { c.u32()? } else { c.u16()? as u32 };
    let protected = c.u16()? != 0;
    let mut it = ItemInfo { id, kind: [0; 4], content_type: Vec::new(), content_encoding: Vec::new(), protected };
    if v >= 2 {
        it.kind = c.fourcc()?;
        c.cstr(); // item_name
        if &it.kind == b"mime" {
            it.content_type = c.cstr().to_vec();
            it.content_encoding = c.cstr().to_vec();
        }
    } else {
        c.cstr();
        it.content_type = c.cstr().to_vec();
        it.content_encoding = c.cstr().to_vec();
    }
    Ok(it)
}

fn parse_iloc(b: &[u8]) -> Result<HashMap<u32, ItemLoc>> {
    let mut c = Cur::new(b);
    let (v, _) = c.full()?;
    if v > 2 {
        return Err(Error::unsupported(format!("iloc version {v}")));
    }
    let a = c.u8()?;
    let (offset_size, length_size) = (a >> 4, a & 15);
    let b2 = c.u8()?;
    let base_offset_size = b2 >> 4;
    let index_size = if v >= 1 { b2 & 15 } else { 0 };
    let count = if v < 2 { c.u16()? as u64 } else { c.u32()? as u64 };
    if count > MAX_ITEMS as u64 {
        return Err(Error::unsupported(format!("too many items in iloc ({count})")));
    }
    let mut locs = HashMap::new();
    let mut total_extents = 0usize;
    for _ in 0..count {
        let id = if v < 2 { c.u16()? as u32 } else { c.u32()? };
        let construction_method = if v >= 1 { (c.u16()? & 15) as u8 } else { 0 };
        let data_reference_index = c.u16()?;
        let base_offset = c.uint(base_offset_size)?;
        let n = c.u16()? as usize;
        total_extents += n;
        if total_extents > MAX_EXTENTS {
            return Err(Error::unsupported("too many iloc extents"));
        }
        let mut extents = Vec::with_capacity(n.min(c.left() / 2 + 1));
        for _ in 0..n {
            if v >= 1 && index_size > 0 {
                c.uint(index_size)?;
            }
            let offset = c.uint(offset_size)?;
            let length = c.uint(length_size)?;
            extents.push(Extent { offset, length });
        }
        if locs.insert(id, ItemLoc { construction_method, data_reference_index, base_offset, extents }).is_some() {
            return Err(Error::invalid(format!("iloc: duplicate item {id}")));
        }
    }
    Ok(locs)
}

fn parse_iref(b: &[u8]) -> Result<Vec<ItemRef>> {
    let mut c = Cur::new(b);
    let (v, _) = c.full()?;
    let mut refs = Vec::new();
    for (kind, body) in split_boxes(c.rest())? {
        let mut r = Cur::new(body);
        let from = if v == 0 { r.u16()? as u32 } else { r.u32()? };
        let n = r.u16()? as usize;
        let mut to = Vec::with_capacity(n.min(r.left() / 2));
        for _ in 0..n {
            to.push(if v == 0 { r.u16()? as u32 } else { r.u32()? });
        }
        refs.push(ItemRef { kind, from, to });
    }
    Ok(refs)
}

fn parse_iprp<'a>(b: &'a [u8], m: &mut Meta<'a>) -> Result<()> {
    for (kind, body) in split_boxes(b)? {
        match &kind {
            b"ipco" => {
                let list = split_boxes(body)?;
                if list.len() > MAX_PROPERTIES {
                    return Err(Error::unsupported("too many properties"));
                }
                m.props = list;
            }
            b"ipma" => {
                let mut c = Cur::new(body);
                let (v, flags) = c.full()?;
                let count = c.u32()?;
                for _ in 0..count {
                    let id = if v == 0 { c.u16()? as u32 } else { c.u32()? };
                    let n = c.u8()?;
                    let list = m.assoc.entry(id).or_default();
                    for _ in 0..n {
                        let (essential, index) = if flags & 1 == 1 {
                            let x = c.u16()?;
                            (x & 0x8000 != 0, (x & 0x7fff) as usize)
                        } else {
                            let x = c.u8()?;
                            (x & 0x80 != 0, (x & 0x7f) as usize)
                        };
                        list.push((index, essential));
                    }
                    if list.len() > 256 {
                        return Err(Error::unsupported("too many properties on one item"));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------
// Items
// ---------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Props {
    ispe: Option<(u32, u32)>,
    pixi: Option<Vec<u8>>,
    av1c: Option<Av1Config>,
    colour: ColourInfo,
    pasp: Option<(u32, u32)>,
    aux_type: Option<String>,
    transforms: Vec<Transform>,
}

struct Loader<'a> {
    file: &'a [u8],
    meta: &'a Meta<'a>,
    budget: Cell<usize>,
}

impl<'a> Loader<'a> {
    fn info(&self, id: u32) -> Result<&'a ItemInfo> {
        self.meta.items.iter().find(|i| i.id == id).ok_or_else(|| Error::invalid(format!("item {id} not found")))
    }

    fn refs_from(&self, from: u32, kind: &FourCc) -> Vec<u32> {
        self.meta.refs.iter().filter(|r| r.from == from && &r.kind == kind).flat_map(|r| r.to.iter().copied()).collect()
    }

    /// Concatenated extents of an item.
    fn data(&self, id: u32) -> Result<Vec<u8>> {
        let loc = self.meta.locs.get(&id).ok_or_else(|| Error::invalid(format!("item {id} has no iloc entry")))?;
        if loc.data_reference_index != 0 {
            return Err(Error::unsupported("item data in an external file"));
        }
        let (src, idat_base) = match loc.construction_method {
            0 => (self.file, 0u64),
            1 => (self.meta.idat.ok_or_else(|| Error::invalid("idat construction without idat box"))?, 0),
            m => return Err(Error::unsupported(format!("iloc construction method {m}"))),
        };
        let mut out = Vec::new();
        for e in &loc.extents {
            let start = loc
                .base_offset
                .checked_add(e.offset)
                .and_then(|s| s.checked_add(idat_base))
                .ok_or_else(|| Error::invalid("iloc: offset overflow"))?;
            let start = usize::try_from(start).map_err(|_| Error::invalid("iloc: offset out of range"))?;
            if start > src.len() {
                return Err(Error::invalid("iloc: extent outside the file"));
            }
            let len = if e.length == 0 {
                src.len() - start
            } else {
                usize::try_from(e.length).map_err(|_| Error::invalid("iloc: length out of range"))?
            };
            let end = start
                .checked_add(len)
                .filter(|&e| e <= src.len())
                .ok_or_else(|| Error::invalid("iloc: extent outside the file"))?;
            let left = self.budget.get();
            if len > left {
                return Err(Error::invalid("item data exceeds the size budget"));
            }
            self.budget.set(left - len);
            out.extend_from_slice(&src[start..end]);
        }
        Ok(out)
    }

    fn props(&self, id: u32) -> Result<Props> {
        let mut p = Props::default();
        let Some(list) = self.meta.assoc.get(&id) else { return Ok(p) };
        for &(index, essential) in list {
            if index == 0 {
                continue;
            }
            let &(kind, body) = self
                .meta
                .props
                .get(index - 1)
                .ok_or_else(|| Error::invalid(format!("item {id}: property index {index} out of range")))?;
            let mut c = Cur::new(body);
            match &kind {
                b"ispe" => {
                    c.full()?;
                    let (w, h) = (c.u32()?, c.u32()?);
                    p.ispe.get_or_insert((w, h));
                }
                b"pixi" => {
                    c.full()?;
                    let n = c.u8()? as usize;
                    p.pixi.get_or_insert(c.take(n)?.to_vec());
                }
                b"av1C" => {
                    if p.av1c.is_none() {
                        p.av1c = Some(Av1Config::parse(body)?);
                    }
                }
                b"colr" => match &c.fourcc()? {
                    b"nclx" => {
                        let (pr, tr, mc, r) = (c.u16()?, c.u16()?, c.u16()?, c.u8()?);
                        p.colour.nclx.get_or_insert(Nclx {
                            colour_primaries: pr,
                            transfer_characteristics: tr,
                            matrix_coefficients: mc,
                            full_range: r & 0x80 != 0,
                        });
                    }
                    b"prof" | b"rICC" => {
                        p.colour.icc.get_or_insert(c.rest().to_vec());
                    }
                    _ => {}
                },
                b"pasp" => {
                    let (h, v) = (c.u32()?, c.u32()?);
                    p.pasp.get_or_insert((h, v));
                }
                b"auxC" => {
                    c.full()?;
                    let urn = String::from_utf8_lossy(c.cstr()).into_owned();
                    p.aux_type.get_or_insert(urn);
                }
                b"irot" => p.transforms.push(Transform::Rotate(c.u8()? & 3)),
                b"imir" => p.transforms.push(Transform::Mirror(c.u8()? & 1)),
                b"clap" => {
                    let (width_n, width_d, height_n, height_d) = (c.u32()?, c.u32()?, c.u32()?, c.u32()?);
                    let (horiz_off_n, horiz_off_d) = (c.u32()? as i32, c.u32()?);
                    let (vert_off_n, vert_off_d) = (c.u32()? as i32, c.u32()?);
                    if width_d == 0 || height_d == 0 || horiz_off_d == 0 || vert_off_d == 0 {
                        return Err(Error::invalid("clap: zero denominator"));
                    }
                    p.transforms.push(Transform::Clap(Clap {
                        width_n,
                        width_d,
                        height_n,
                        height_d,
                        horiz_off_n,
                        horiz_off_d,
                        vert_off_n,
                        vert_off_d,
                    }));
                }
                k if essential && !IGNORABLE.contains(&k) => {
                    return Err(Error::unsupported(format!("item {id}: essential property '{}'", show(k))));
                }
                _ => {}
            }
        }
        Ok(p)
    }

    fn check_size(w: u32, h: u32) -> Result<()> {
        if w == 0 || h == 0 || w as u64 * h as u64 > MAX_PIXELS {
            return Err(Error::invalid(format!("image size {w}x{h} out of range")));
        }
        Ok(())
    }

    fn coded(&self, id: u32, p: &Props) -> Result<CodedItem> {
        let av1c = p.av1c.clone().ok_or_else(|| Error::invalid(format!("item {id}: missing av1C")))?;
        if let Some((w, h)) = p.ispe {
            Self::check_size(w, h)?;
        }
        let data = self.data(id)?;
        if data.is_empty() {
            return Err(Error::invalid(format!("item {id}: empty image data")));
        }
        Ok(CodedItem { item_id: id, ispe: p.ispe, pixi: p.pixi.clone(), av1c, data })
    }

    fn load_image(&self, id: u32, depth: u32) -> Result<ImageItem> {
        if depth > MAX_DERIVATION_DEPTH {
            return Err(Error::unsupported("derived images nested too deeply"));
        }
        let info = self.info(id)?;
        if info.protected {
            return Err(Error::unsupported("protected item"));
        }
        let mut p = self.props(id)?;
        let (data, ispe) = match &info.kind {
            b"av01" => {
                let (w, h) = p.ispe.ok_or_else(|| Error::invalid(format!("item {id}: missing ispe")))?;
                let coded = self.coded(id, &p)?;
                return Ok(ImageItem {
                    item_id: id,
                    width: w,
                    height: h,
                    pixi: p.pixi,
                    colour: p.colour,
                    pasp: p.pasp,
                    transforms: p.transforms,
                    data: ImageData::Coded(coded),
                });
            }
            b"grid" => {
                let g = self.grid(id)?;
                let ispe = p.ispe.unwrap_or((g.output_width, g.output_height));
                (ImageData::Grid(g), ispe)
            }
            b"iden" => {
                let src = self.refs_from(id, b"dimg");
                if src.len() != 1 {
                    return Err(Error::invalid("iden item must reference exactly one image"));
                }
                let inner = self.load_image(src[0], depth + 1)?;
                let mut transforms = inner.transforms.clone();
                transforms.append(&mut p.transforms);
                p.transforms = transforms;
                let ispe = p.ispe.unwrap_or((inner.width, inner.height));
                if p.colour == ColourInfo::default() {
                    p.colour = inner.colour.clone();
                }
                (inner.data, ispe)
            }
            k => return Err(Error::unsupported(format!("image item type '{}'", show(k)))),
        };
        Self::check_size(ispe.0, ispe.1)?;
        Ok(ImageItem {
            item_id: id,
            width: ispe.0,
            height: ispe.1,
            pixi: p.pixi,
            colour: p.colour,
            pasp: p.pasp,
            transforms: p.transforms,
            data,
        })
    }

    fn grid(&self, id: u32) -> Result<Grid> {
        let d = self.data(id)?;
        let mut c = Cur::new(&d);
        let version = c.u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!("grid version {version}")));
        }
        let flags = c.u8()?;
        let rows = c.u8()? as u32 + 1;
        let columns = c.u8()? as u32 + 1;
        let (output_width, output_height) =
            if flags & 1 == 1 { (c.u32()?, c.u32()?) } else { (c.u16()? as u32, c.u16()? as u32) };
        Self::check_size(output_width, output_height)?;
        let n = (rows * columns) as usize;
        if n > MAX_GRID_TILES {
            return Err(Error::unsupported(format!("grid with {n} tiles")));
        }
        let ids = self.refs_from(id, b"dimg");
        if ids.len() != n {
            return Err(Error::invalid(format!("grid {rows}x{columns} references {} tiles", ids.len())));
        }
        let mut tiles = Vec::with_capacity(n);
        for tid in ids {
            let info = self.info(tid)?;
            if &info.kind != b"av01" || info.protected {
                return Err(Error::unsupported(format!("grid tile type '{}'", show(&info.kind))));
            }
            // Transformative properties on a tile have no meaning; they are not reported.
            tiles.push(self.coded(tid, &self.props(tid)?)?);
        }
        let tile_size = tiles[0].ispe;
        if tiles.iter().any(|t| t.ispe != tile_size) {
            return Err(Error::invalid("grid tiles differ in size"));
        }
        if let Some((tw, th)) = tile_size
            && (output_width as u64 > tw as u64 * columns as u64 || output_height as u64 > th as u64 * rows as u64)
        {
            return Err(Error::invalid("grid output is larger than its tiles"));
        }
        Ok(Grid { version, flags, rows, columns, output_width, output_height, tiles })
    }

    fn find_alpha(&self, primary: u32) -> Result<Option<AlphaItem>> {
        for r in &self.meta.refs {
            if &r.kind != b"auxl" || !r.to.contains(&primary) {
                continue;
            }
            let Ok(info) = self.info(r.from) else { continue };
            if info.protected || !matches!(&info.kind, b"av01" | b"grid") {
                continue;
            }
            let aux = self.props(r.from)?.aux_type;
            let Some(urn) = aux.filter(|u| ALPHA_URNS.contains(&u.as_str())) else { continue };
            let item = self.load_image(r.from, 0)?;
            let premultiplied = self.meta.refs.iter().any(|p| {
                &p.kind == b"prem"
                    && ((p.from == r.from && p.to.contains(&primary)) || (p.from == primary && p.to.contains(&r.from)))
            });
            return Ok(Some(AlphaItem { item, premultiplied, aux_type: urn }));
        }
        Ok(None)
    }

    fn exif(&self, primary: u32) -> Option<Vec<u8>> {
        let cands: Vec<&ItemInfo> = self.meta.items.iter().filter(|i| &i.kind == b"Exif" && !i.protected).collect();
        let linked = cands.iter().find(|i| self.refs_from(i.id, b"cdsc").contains(&primary));
        let it = linked.or(cands.first())?;
        let d = self.data(it.id).ok()?;
        let off = u32::from_be_bytes(d.get(..4)?.try_into().ok()?) as usize;
        let tiff = d.get(4usize.checked_add(off)?..)?;
        let tiff = tiff.strip_prefix(b"Exif\0\0").unwrap_or(tiff);
        (!tiff.is_empty()).then(|| tiff.to_vec())
    }

    fn xmp(&self, primary: u32) -> Option<Vec<u8>> {
        let cands: Vec<&ItemInfo> = self
            .meta
            .items
            .iter()
            .filter(|i| {
                &i.kind == b"mime"
                    && !i.protected
                    && i.content_encoding.is_empty()
                    && i.content_type.starts_with(b"application/rdf+xml")
            })
            .collect();
        let linked = cands.iter().find(|i| self.refs_from(i.id, b"cdsc").contains(&primary));
        let it = linked.or(cands.first())?;
        let d = self.data(it.id).ok()?;
        (!d.is_empty()).then_some(d)
    }
}

/// Parse an AVIF file. Never panics; malformed input is [`Error::Invalid`], valid but unhandled features are
/// [`Error::Unsupported`]. See the module documentation for scope and limits.
pub fn parse(data: &[u8]) -> Result<AvifFile> {
    let top = split_boxes(data)?;
    let mut brands: Option<(FourCc, Vec<FourCc>)> = None;
    let mut meta_body = None;
    let mut has_moov = false;
    for &(kind, body) in &top {
        match &kind {
            b"ftyp" if brands.is_none() => {
                let mut c = Cur::new(body);
                let major = c.fourcc()?;
                c.u32()?;
                let mut compat = Vec::new();
                while c.left() >= 4 {
                    compat.push(c.fourcc()?);
                }
                brands = Some((major, compat));
            }
            b"meta" if meta_body.is_none() => meta_body = Some(body),
            b"moov" => has_moov = true,
            _ => {}
        }
    }
    let (major_brand, compatible_brands) = brands.ok_or_else(|| Error::invalid("missing ftyp box"))?;
    const OK: [&FourCc; 4] = [b"avif", b"avis", b"mif1", b"miaf"];
    if !std::iter::once(&major_brand).chain(compatible_brands.iter()).any(|b| OK.contains(&b)) {
        return Err(Error::unsupported(format!("no AVIF-compatible brand (major '{}')", show(&major_brand))));
    }
    let is_sequence = &major_brand == b"avis" || compatible_brands.contains(b"avis") || has_moov;
    let Some(meta_body) = meta_body else {
        return Err(if is_sequence {
            Error::unsupported("animated AVIF without a still primary item")
        } else {
            Error::invalid("missing meta box")
        });
    };
    let meta = parse_meta(meta_body)?;
    let primary_id = meta.primary.ok_or_else(|| Error::invalid("missing pitm box"))?;
    let loader =
        Loader { file: data, meta: &meta, budget: Cell::new(data.len().saturating_mul(4).saturating_add(1 << 20)) };
    let primary = loader.load_image(primary_id, 0)?;
    let alpha = loader.find_alpha(primary_id)?;
    Ok(AvifFile {
        major_brand,
        compatible_brands,
        is_sequence,
        primary,
        alpha,
        exif: loader.exif(primary_id),
        xmp: loader.xmp(primary_id),
    })
}

// ---------------------------------------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------------------------------------

/// One AV1 stream (colour or alpha): the OBU payload and its `av1C`.
#[derive(Clone, Debug)]
pub struct Av1Stream {
    /// OBU stream: sequence header + frame, without a temporal delimiter (AVIF requires it be omitted).
    pub data: Vec<u8>,
    pub config: Av1Config,
}

/// Everything [`write`] needs for a single-image AVIF.
#[derive(Clone, Debug)]
pub struct AvifOutput {
    pub width: u32,
    pub height: u32,
    pub color: Av1Stream,
    /// Alpha plane, coded as a monochrome AV1 image of the same size.
    pub alpha: Option<Av1Stream>,
    /// Write a `prem` reference: the colour image is premultiplied by the alpha.
    pub alpha_premultiplied: bool,
    pub colour: ColourInfo,
    /// `clap`, `irot` and `imir` are written in the order the AVIF specification requires.
    pub clap: Option<Clap>,
    /// `irot` quarter turns anticlockwise; 0 writes no box.
    pub rotation: u8,
    /// `imir` axis (see [`Transform::Mirror`]).
    pub mirror: Option<u8>,
    /// EXIF as a bare TIFF block (an `Exif\0\0` prefix is tolerated and removed).
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    /// Further frames: the file becomes an `avis` image sequence whose first sample is `color` (and `alpha`).
    pub sequence: Option<SequenceOut>,
}

/// The frames after the first of an animation, as AV1 samples (sequence header only in the first one).
#[derive(Clone, Debug)]
pub struct SequenceOut {
    /// Ticks per second of the durations.
    pub timescale: u32,
    /// Duration of every frame including the first.
    pub durations: Vec<u32>,
    /// Which frames are sync samples (key frames), including the first.
    pub sync: Vec<bool>,
    /// Colour samples of frames 1.. .
    pub color: Vec<Vec<u8>>,
    /// Alpha samples of frames 1.. (empty when there is no alpha).
    pub alpha: Vec<Vec<u8>>,
    pub loop_forever: bool,
}

impl AvifOutput {
    /// Opaque image with sRGB-style signalling and nothing else.
    pub fn new(width: u32, height: u32, color: Av1Stream) -> Self {
        AvifOutput {
            width,
            height,
            color,
            alpha: None,
            alpha_premultiplied: false,
            colour: ColourInfo::default(),
            clap: None,
            rotation: 0,
            mirror: None,
            exif: None,
            xmp: None,
            sequence: None,
        }
    }
}

pub(super) fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(body.len() + 8);
    o.extend_from_slice(&(body.len() as u32 + 8).to_be_bytes());
    o.extend_from_slice(kind);
    o.extend_from_slice(body);
    o
}

pub(super) fn fbx(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = vec![version, (flags >> 16) as u8, (flags >> 8) as u8, flags as u8];
    b.extend_from_slice(body);
    bx(kind, &b)
}

/// Deduplicating property table (`ipco`).
#[derive(Default)]
struct PropTable {
    boxes: Vec<Vec<u8>>,
}

impl PropTable {
    /// 1-based property index of an identical box, adding it when new.
    fn add(&mut self, b: Vec<u8>) -> u16 {
        let i = self.boxes.iter().position(|x| *x == b).unwrap_or_else(|| {
            self.boxes.push(b);
            self.boxes.len() - 1
        });
        i as u16 + 1
    }
}

struct WItem {
    id: u16,
    kind: FourCc,
    content_type: Option<&'static str>,
    data: Vec<u8>,
    props: Vec<(u16, bool)>,
}

fn pixi(channels: u8, depth: u8) -> Vec<u8> {
    let mut b = vec![channels];
    b.extend(std::iter::repeat_n(depth, channels as usize));
    fbx(b"pixi", 0, 0, &b)
}

/// Build a minimal AVIF: `ftyp` (avif; avif, mif1, miaf), `meta`, one `mdat`. The `iloc` box is version 0 with
/// construction method 0 and 4-byte offsets and lengths (8 bytes only when the payload needs them), so `meta`
/// has a size that does not depend on the offsets and the layout is computed in two passes.
pub fn write(img: &AvifOutput) -> Vec<u8> {
    let mut table = PropTable::default();
    let mut items: Vec<WItem> = Vec::new();

    let mut ispe = vec![];
    ispe.extend_from_slice(&img.width.to_be_bytes());
    ispe.extend_from_slice(&img.height.to_be_bytes());
    let ispe = table.add(fbx(b"ispe", 0, 0, &ispe));

    // Colour item: ispe, pixi, av1C, colr, then the transformative properties in clap, irot, imir order.
    let c = &img.color.config;
    let mut props = vec![(ispe, false), (table.add(pixi(c.channels(), c.bit_depth())), false)];
    props.push((table.add(bx(b"av1C", &c.to_bytes())), true));
    if let Some(n) = &img.colour.nclx {
        let mut b = b"nclx".to_vec();
        b.extend_from_slice(&n.colour_primaries.to_be_bytes());
        b.extend_from_slice(&n.transfer_characteristics.to_be_bytes());
        b.extend_from_slice(&n.matrix_coefficients.to_be_bytes());
        b.push((n.full_range as u8) << 7);
        props.push((table.add(bx(b"colr", &b)), false));
    }
    if let Some(icc) = &img.colour.icc {
        let mut b = b"prof".to_vec();
        b.extend_from_slice(icc);
        props.push((table.add(bx(b"colr", &b)), false));
    }
    if let Some(k) = &img.clap {
        let mut b = Vec::new();
        for v in [
            k.width_n,
            k.width_d,
            k.height_n,
            k.height_d,
            k.horiz_off_n as u32,
            k.horiz_off_d,
            k.vert_off_n as u32,
            k.vert_off_d,
        ] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        props.push((table.add(bx(b"clap", &b)), true));
    }
    if img.rotation & 3 != 0 {
        props.push((table.add(bx(b"irot", &[img.rotation & 3])), true));
    }
    if let Some(axis) = img.mirror {
        props.push((table.add(bx(b"imir", &[axis & 1])), true));
    }
    items.push(WItem { id: 1, kind: *b"av01", content_type: None, data: img.color.data.clone(), props });

    // Alpha item shares ispe (and anything else byte-identical) with the colour item.
    if let Some(a) = &img.alpha {
        let mut urn = ALPHA_URNS[0].as_bytes().to_vec();
        urn.push(0);
        let props = vec![
            (ispe, false),
            (table.add(pixi(1, a.config.bit_depth())), false),
            (table.add(bx(b"av1C", &a.config.to_bytes())), true),
            (table.add(fbx(b"auxC", 0, 0, &urn)), false),
        ];
        let id = items.len() as u16 + 1;
        items.push(WItem { id, kind: *b"av01", content_type: None, data: a.data.clone(), props });
    }
    if let Some(e) = &img.exif {
        let tiff = e.strip_prefix(b"Exif\0\0").unwrap_or(e);
        let mut d = vec![0, 0, 0, 0];
        d.extend_from_slice(tiff);
        let id = items.len() as u16 + 1;
        items.push(WItem { id, kind: *b"Exif", content_type: None, data: d, props: vec![] });
    }
    if let Some(x) = &img.xmp {
        let id = items.len() as u16 + 1;
        items.push(WItem {
            id,
            kind: *b"mime",
            content_type: Some("application/rdf+xml"),
            data: x.clone(),
            props: vec![],
        });
    }

    let items_total: u64 = items.iter().map(|i| i.data.len() as u64).sum();
    let extra_total: u64 =
        img.sequence.as_ref().map_or(0, |s| s.color.iter().chain(&s.alpha).map(|d| d.len() as u64).sum());
    let total = items_total + extra_total;
    let wide = total > 0x7fff_ffff;
    let ftyp = if img.sequence.is_some() {
        bx(b"ftyp", b"avis\0\0\0\0avisavifmif1miafmsf1iso8")
    } else {
        bx(b"ftyp", b"avif\0\0\0\0avifmif1miaf")
    };

    let build_meta = |offsets: &[u64]| -> Vec<u8> {
        let mut m = fbx(b"hdlr", 0, 0, &[&[0u8; 4][..], b"pict", &[0u8; 12], &[0u8][..]].concat());
        m.extend(fbx(b"pitm", 0, 0, &1u16.to_be_bytes()));
        let mut iinf = (items.len() as u16).to_be_bytes().to_vec();
        for it in &items {
            let mut b = it.id.to_be_bytes().to_vec();
            b.extend_from_slice(&[0, 0]);
            b.extend_from_slice(&it.kind);
            b.push(0);
            if let Some(ct) = it.content_type {
                b.extend_from_slice(ct.as_bytes());
                b.push(0);
            }
            iinf.extend(fbx(b"infe", 2, 0, &b));
        }
        m.extend(fbx(b"iinf", 0, 0, &iinf));

        let osz: u8 = if wide { 8 } else { 4 };
        let mut iloc = vec![osz << 4 | osz, 0];
        iloc.extend_from_slice(&(items.len() as u16).to_be_bytes());
        for (it, &off) in items.iter().zip(offsets) {
            iloc.extend_from_slice(&it.id.to_be_bytes());
            iloc.extend_from_slice(&[0, 0, 0, 1]); // data_reference_index = 0, extent_count = 1
            if wide {
                iloc.extend_from_slice(&off.to_be_bytes());
                iloc.extend_from_slice(&(it.data.len() as u64).to_be_bytes());
            } else {
                iloc.extend_from_slice(&(off as u32).to_be_bytes());
                iloc.extend_from_slice(&(it.data.len() as u32).to_be_bytes());
            }
        }
        m.extend(fbx(b"iloc", 0, 0, &iloc));

        // iref: alpha -> colour (auxl, prem), metadata -> colour (cdsc).
        let mut iref = Vec::new();
        let r = |kind: &[u8; 4], from: u16| {
            bx(kind, &[&from.to_be_bytes()[..], &1u16.to_be_bytes(), &1u16.to_be_bytes()].concat())
        };
        for it in items.iter().skip(1) {
            match &it.kind {
                b"av01" => {
                    iref.extend(r(b"auxl", it.id));
                    if img.alpha_premultiplied {
                        iref.extend(r(b"prem", it.id));
                    }
                }
                _ => iref.extend(r(b"cdsc", it.id)),
            }
        }
        if !iref.is_empty() {
            m.extend(fbx(b"iref", 0, 0, &iref));
        }

        let ipco: Vec<u8> = table.boxes.concat();
        let wide_idx = table.boxes.len() > 127;
        let with_props: Vec<&WItem> = items.iter().filter(|i| !i.props.is_empty()).collect();
        let mut ipma = (with_props.len() as u32).to_be_bytes().to_vec();
        for it in with_props {
            ipma.extend_from_slice(&it.id.to_be_bytes());
            ipma.push(it.props.len() as u8);
            for &(idx, essential) in &it.props {
                if wide_idx {
                    ipma.extend_from_slice(&(idx | (essential as u16) << 15).to_be_bytes());
                } else {
                    ipma.push(idx as u8 | (essential as u8) << 7);
                }
            }
        }
        let mut iprp = bx(b"ipco", &ipco);
        iprp.extend(fbx(b"ipma", 0, wide_idx as u32, &ipma));
        m.extend(bx(b"iprp", &iprp));
        fbx(b"meta", 0, 0, &m)
    };

    let zeros = vec![0u64; items.len()];
    let meta_len = build_meta(&zeros).len() as u64;
    let big_mdat = total + 8 > u32::MAX as u64;
    let mdat_hdr: u64 = if big_mdat { 16 } else { 8 };
    let mut pos = ftyp.len() as u64 + meta_len + mdat_hdr;
    let mut offsets = Vec::with_capacity(items.len());
    for it in &items {
        offsets.push(pos);
        pos += it.data.len() as u64;
    }
    let meta = build_meta(&offsets);
    debug_assert_eq!(meta.len() as u64, meta_len);

    let mut out = Vec::with_capacity(pos as usize);
    out.extend(ftyp);
    out.extend(meta);
    if big_mdat {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&(total + 16).to_be_bytes());
    } else {
        out.extend_from_slice(&(total as u32 + 8).to_be_bytes());
        out.extend_from_slice(b"mdat");
    }
    for it in &items {
        out.extend_from_slice(&it.data);
    }
    if let Some(seq) = &img.sequence {
        for d in seq.color.iter().chain(&seq.alpha) {
            out.extend_from_slice(d);
        }
        out.extend(sequence_moov(img, seq, &offsets, pos_after_items(&offsets, &items)));
    }
    out
}

fn pos_after_items(offsets: &[u64], items: &[WItem]) -> u64 {
    offsets.last().zip(items.last()).map_or(0, |(&o, i)| o + i.data.len() as u64)
}

/// The `moov` of an animation: sample 0 of each track is the primary item (and the alpha item), the rest follow
/// the items in `mdat`, colour samples first.
fn sequence_moov(img: &AvifOutput, seq: &SequenceOut, item_offsets: &[u64], extras_start: u64) -> Vec<u8> {
    let frames = seq.durations.len();
    let mut pos = extras_start;
    let mut track = |first_off: u64, first_len: usize, rest: &[Vec<u8>]| {
        let mut offsets = vec![first_off];
        let mut sizes = vec![first_len as u32];
        for d in rest {
            offsets.push(pos);
            sizes.push(d.len() as u32);
            pos += d.len() as u64;
        }
        (offsets, sizes)
    };
    let (c_off, c_sz) = track(item_offsets[0], img.color.data.len(), &seq.color);
    let alpha_tr = img.alpha.as_ref().map(|a| track(item_offsets[1], a.data.len(), &seq.alpha));
    let mut colr = Vec::new();
    if let Some(n) = &img.colour.nclx {
        let mut b = b"nclx".to_vec();
        b.extend_from_slice(&n.colour_primaries.to_be_bytes());
        b.extend_from_slice(&n.transfer_characteristics.to_be_bytes());
        b.extend_from_slice(&n.matrix_coefficients.to_be_bytes());
        b.push((n.full_range as u8) << 7);
        colr.extend(bx(b"colr", &b));
    }
    if let Some(icc) = &img.colour.icc {
        let mut b = b"prof".to_vec();
        b.extend_from_slice(icc);
        colr.extend(bx(b"colr", &b));
    }
    let mut tracks = vec![TrackOut {
        id: 1,
        alpha: false,
        config: &img.color.config,
        width: img.width,
        height: img.height,
        extra_entry_boxes: colr,
        offsets: c_off,
        sizes: c_sz,
        durations: seq.durations.clone(),
        sync: seq.sync.clone(),
        aux_for: None,
    }];
    if let (Some(a), Some((offsets, sizes))) = (&img.alpha, alpha_tr) {
        tracks.push(TrackOut {
            id: 2,
            alpha: true,
            config: &a.config,
            width: img.width,
            height: img.height,
            extra_entry_boxes: Vec::new(),
            offsets,
            sizes,
            durations: seq.durations.clone(),
            sync: seq.sync.clone(),
            aux_for: Some(1),
        });
    }
    debug_assert!(tracks.iter().all(|t| t.offsets.len() == frames));
    super::sequence::build_moov(&tracks, seq.timescale, seq.loop_forever)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(byte: u8, mono: bool) -> Av1Stream {
        Av1Stream {
            data: vec![byte; 20],
            config: Av1Config { monochrome: mono, chroma_subsampling_y: 1, ..Av1Config::default() },
        }
    }

    #[test]
    fn av1c_roundtrip() {
        let c = Av1Config {
            seq_profile: 2,
            seq_level_idx_0: 13,
            seq_tier_0: 1,
            high_bitdepth: true,
            twelve_bit: true,
            monochrome: false,
            chroma_subsampling_x: 1,
            chroma_subsampling_y: 0,
            chroma_sample_position: 2,
            initial_presentation_delay_minus_one: Some(3),
            config_obus: vec![1, 2, 3],
        };
        assert_eq!(Av1Config::parse(&c.to_bytes()).unwrap(), c);
        assert_eq!(c.bit_depth(), 12);
        assert!(Av1Config::parse(&[0x80, 0, 0, 0]).is_err());
        assert!(Av1Config::parse(&[0x81]).is_err());
    }

    #[test]
    fn orientation_composition() {
        use Transform::*;
        assert_eq!(net_orientation(&[]), Orientation::default());
        assert_eq!(net_orientation(&[Rotate(1), Rotate(3)]), Orientation::default());
        assert_eq!(net_orientation(&[Mirror(0), Mirror(0)]), Orientation::default());
        assert_eq!(net_orientation(&[Mirror(1)]), Orientation { rotate: 2, mirror: true });
        assert_eq!(net_orientation(&[Mirror(0), Mirror(1)]), Orientation { rotate: 2, mirror: false });
        // Rotate then mirror vs mirror then rotate differ.
        assert_eq!(net_orientation(&[Rotate(1), Mirror(0)]), Orientation { rotate: 3, mirror: true });
        assert_eq!(net_orientation(&[Mirror(0), Rotate(1)]), Orientation { rotate: 1, mirror: true });
        assert!(Orientation { rotate: 3, mirror: false }.swaps_dimensions());
    }

    #[test]
    fn box_sizes() {
        // size 0 = to end, 64-bit size, uuid.
        let mut d = bx(b"free", &[1, 2, 3]);
        d.extend_from_slice(&[0, 0, 0, 0]);
        d.extend_from_slice(b"mdat");
        d.extend_from_slice(&[9; 5]);
        let b = split_boxes(&d).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[1].1, &[9; 5]);
        let mut d = 1u32.to_be_bytes().to_vec();
        d.extend_from_slice(b"free");
        d.extend_from_slice(&20u64.to_be_bytes());
        d.extend_from_slice(&[7; 4]);
        assert_eq!(split_boxes(&d).unwrap()[0].1, &[7; 4]);
        let mut d = 28u32.to_be_bytes().to_vec();
        d.extend_from_slice(b"uuid");
        d.extend_from_slice(&[0; 16]);
        d.extend_from_slice(&[5; 4]);
        assert_eq!(split_boxes(&d).unwrap()[0].1, &[5; 4]);
        // Oversized and undersized boxes.
        assert!(split_boxes(&[0, 0, 1, 0, b'f', b'r', b'e', b'e']).is_err());
        assert!(split_boxes(&[0, 0, 0, 4, b'f', b'r', b'e', b'e']).is_err());
    }

    #[test]
    fn write_parse_roundtrip() {
        let mut o = AvifOutput::new(64, 48, stream(1, false));
        o.alpha = Some(stream(2, true));
        o.alpha_premultiplied = true;
        o.colour = ColourInfo { nclx: Some(Nclx::SRGB), icc: Some(vec![9; 30]) };
        o.rotation = 1;
        o.mirror = Some(1);
        o.exif = Some(b"MM\0*\0\0\0\x08".to_vec());
        o.xmp = Some(b"<x/>".to_vec());
        let f = parse(&write(&o)).unwrap();
        assert_eq!(&f.major_brand, b"avif");
        assert_eq!((f.primary.width, f.primary.height), (64, 48));
        assert_eq!(f.primary.colour.nclx, Some(Nclx::SRGB));
        assert_eq!(f.icc(), Some(&[9u8; 30][..]));
        assert_eq!(f.primary.transforms, vec![Transform::Rotate(1), Transform::Mirror(1)]);
        let ImageData::Coded(c) = &f.primary.data else { panic!() };
        assert_eq!(c.data, o.color.data);
        assert_eq!(c.pixi, Some(vec![8, 8, 8]));
        let a = f.alpha.unwrap();
        assert!(a.premultiplied);
        assert_eq!(a.item.pixi, Some(vec![8]));
        assert!(a.item.av1_config().monochrome);
        assert_eq!(f.exif.as_deref(), Some(&b"MM\0*\0\0\0\x08"[..]));
        assert_eq!(f.xmp.as_deref(), Some(&b"<x/>"[..]));
    }

    #[test]
    fn rejects_foreign_files() {
        assert!(parse(b"").is_err());
        assert!(parse(b"RIFF....WEBP").is_err());
        let heic = bx(b"ftyp", b"heic\0\0\0\0heic");
        assert!(matches!(parse(&heic), Err(Error::Unsupported(_))));
    }
}
