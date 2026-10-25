//! JPEG decoder: baseline, extended-sequential and progressive Huffman, 8-bit, 1, 3 or 4 components
//! (4 = CMYK/YCCK, which the DCT-domain path carries through untouched).

use super::ZIGZAG;
use super::dct::{AAN, idct, round_f32};
use super::encode::{CoefComponent, CoefImage};
use rayon::prelude::*;
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image, MAX_PIXELS, Metadata};

// ---------------------------------------------------------------- bit reader

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
    marker: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos, acc: 0, n: 0, marker: false }
    }

    #[inline(always)]
    fn fill(&mut self) {
        while self.n <= 56 {
            let byte = if self.marker {
                0
            } else {
                match self.data.get(self.pos) {
                    None => {
                        self.marker = true;
                        0
                    }
                    Some(&0xFF) => match self.data.get(self.pos + 1) {
                        Some(0) => {
                            self.pos += 2;
                            0xFF
                        }
                        _ => {
                            self.marker = true;
                            0
                        }
                    },
                    Some(&b) => {
                        self.pos += 1;
                        b
                    }
                }
            };
            self.acc = (self.acc << 8) | byte as u64;
            self.n += 8;
        }
    }

    #[inline(always)]
    fn peek(&mut self, k: u32) -> u32 {
        if self.n < k {
            self.fill();
        }
        ((self.acc >> (self.n - k)) & ((1u64 << k) - 1)) as u32
    }

    #[inline(always)]
    fn skip(&mut self, k: u32) {
        self.n -= k;
    }

    #[inline(always)]
    fn bits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        let v = self.peek(k);
        self.skip(k);
        v
    }

    #[inline(always)]
    fn bit(&mut self) -> bool {
        self.bits(1) == 1
    }

    /// Drop buffered bits, skip any padding up to the next marker and consume it if it is RSTn.
    fn restart(&mut self) {
        self.acc = 0;
        self.n = 0;
        self.marker = false;
        let d = self.data;
        while self.pos + 1 < d.len() {
            if d[self.pos] != 0xFF {
                self.pos += 1;
                continue;
            }
            match d[self.pos + 1] {
                0 => self.pos += 2,
                0xD0..=0xD7 => {
                    self.pos += 2;
                    return;
                }
                _ => return,
            }
        }
    }
}

// ---------------------------------------------------------------- huffman

const FAST: u32 = 9;

#[derive(Clone)]
struct HuffDec {
    fast: [u16; 1 << FAST],
    mincode: [i32; 17],
    maxcode: [i32; 18],
    valptr: [i32; 17],
    vals: Vec<u8>,
}

impl HuffDec {
    fn new(bits: &[u8; 16], vals: &[u8]) -> Result<Self> {
        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if total > 256 || total > vals.len() {
            return Err(Error::invalid("bad DHT symbol count"));
        }
        let mut h = HuffDec {
            fast: [0; 1 << FAST],
            mincode: [0; 17],
            maxcode: [-1; 18],
            valptr: [0; 17],
            vals: vals[..total].to_vec(),
        };
        let mut code = 0i32;
        let mut k = 0usize;
        for len in 1..=16usize {
            let n = bits[len - 1] as i32;
            h.valptr[len] = k as i32;
            h.mincode[len] = code;
            if n > 0 {
                if code + n > (1 << len) {
                    return Err(Error::invalid("oversubscribed JPEG huffman table"));
                }
                if len as u32 <= FAST {
                    for i in 0..n {
                        let c = (code + i) as usize;
                        let sym = h.vals[k + i as usize];
                        let shift = FAST as usize - len;
                        for fill in 0..1usize << shift {
                            h.fast[(c << shift) | fill] = ((len as u16) << 8) | sym as u16;
                        }
                    }
                }
                code += n;
                k += n as usize;
                h.maxcode[len] = code - 1;
            }
            code <<= 1;
        }
        Ok(h)
    }

    #[inline(always)]
    fn decode(&self, br: &mut BitReader) -> Result<u8> {
        let e = self.fast[br.peek(FAST) as usize];
        if e != 0 {
            br.skip((e >> 8) as u32);
            return Ok(e as u8);
        }
        for len in 1..=16u32 {
            let code = br.peek(len) as i32;
            if self.maxcode[len as usize] >= 0
                && code <= self.maxcode[len as usize]
                && code >= self.mincode[len as usize]
            {
                br.skip(len);
                let idx = self.valptr[len as usize] + code - self.mincode[len as usize];
                return Ok(self.vals[idx as usize]);
            }
        }
        Err(Error::invalid("bad JPEG huffman code"))
    }
}

#[inline(always)]
fn extend(v: u32, t: u32) -> i32 {
    if t == 0 {
        0
    } else if v < (1 << (t - 1)) {
        v as i32 - (1 << t) + 1
    } else {
        v as i32
    }
}

// ---------------------------------------------------------------- structures

struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// Storage dims in blocks (padded to whole MCUs).
    blocks_w: usize,
    blocks_h: usize,
    /// Dims in blocks that actually cover the image (used by non-interleaved scans).
    real_bw: usize,
    real_bh: usize,
    coefs: Vec<i16>,
}

struct ScanComp {
    idx: usize,
    dc: usize,
    ac: usize,
}

#[derive(Clone, Copy)]
struct ScanParams {
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    qt: [Option<[u16; 64]>; 4],
    dc_tabs: [Option<HuffDec>; 4],
    ac_tabs: [Option<HuffDec>; 4],
    comps: Vec<Component>,
    width: usize,
    height: usize,
    hmax: usize,
    vmax: usize,
    mcus_x: usize,
    mcus_y: usize,
    progressive: bool,
    restart_interval: usize,
    adobe_transform: Option<u8>,
    scans_seen: usize,
    /// ICC profile fragments as (sequence number, bytes).
    icc_parts: Vec<(u8, Vec<u8>)>,
    /// Descriptive segments (EXIF/XMP/Photoshop APP segments, comments) as (marker, body).
    extras: Vec<(u8, Vec<u8>)>,
}

const EXIF_PREFIX: &[u8] = b"Exif\0\0";
const XMP_PREFIX: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const PHOTOSHOP_PREFIX: &[u8] = b"Photoshop 3.0\0";
/// Descriptive metadata is capped so a hostile file cannot make us hoard memory.
const MAX_EXTRA_SEGMENTS: usize = 256;

/// Assemble the `Metadata` of a decoded file from the ICC fragments and descriptive segments it held.
fn build_meta(mut icc_parts: Vec<(u8, Vec<u8>)>, extras: Vec<(u8, Vec<u8>)>) -> Metadata {
    icc_parts.sort_by_key(|p| p.0);
    let mut meta = Metadata {
        icc: (!icc_parts.is_empty()).then(|| icc_parts.into_iter().flat_map(|p| p.1).collect()),
        ..Default::default()
    };
    for (marker, body) in extras {
        match marker {
            0xE1 if body.starts_with(EXIF_PREFIX) && meta.exif.is_none() => meta.exif = Some(body[6..].to_vec()),
            0xE1 if body.starts_with(XMP_PREFIX) && meta.xmp.is_none() => {
                meta.xmp = Some(body[XMP_PREFIX.len()..].to_vec());
            }
            0xED if body.starts_with(PHOTOSHOP_PREFIX) => meta.jpeg_extra.push((marker, body)),
            0xFE => meta.jpeg_extra.push((marker, body)),
            _ => {}
        }
    }
    meta
}

fn be16(d: &[u8], at: usize) -> Result<usize> {
    d.get(at..at + 2).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize).ok_or_else(|| Error::invalid("truncated JPEG"))
}

/// Facts about a parsed JPEG that drive the choice between DCT-domain and pixel-domain recompression.
#[derive(Debug, Clone)]
pub struct Info {
    pub width: usize,
    pub height: usize,
    /// Sampling factors (h, v) per component.
    pub sampling: Vec<(usize, usize)>,
    /// Estimated IJG quality of the source quantization tables.
    pub source_quality: u8,
    /// Components are stored as RGB rather than YCbCr.
    pub coded_rgb: bool,
    /// Four components: CMYK, or YCCK when `adobe_transform` is 2.
    pub cmyk: bool,
}

/// Pixels a viewer would show for these quantized coefficients (dequantize, IDCT, upsample).
pub fn reconstruct(ci: &CoefImage) -> Result<Image> {
    let hmax = ci.comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = ci.comps.iter().map(|c| c.v).max().unwrap_or(1);
    let mut qt: [Option<[u16; 64]>; 4] = [None; 4];
    for (slot, q) in ci.qtables.iter().enumerate() {
        qt[slot] = q.map(|t| t.map(|v| v as u16));
    }
    let comps = ci
        .comps
        .iter()
        .map(|c| Component {
            id: c.id,
            h: c.h,
            v: c.v,
            tq: c.tq,
            blocks_w: c.blocks_w,
            blocks_h: c.coefs.len() / 64 / c.blocks_w,
            real_bw: c.blocks_w,
            real_bh: c.coefs.len() / 64 / c.blocks_w,
            coefs: c.coefs.clone(),
        })
        .collect();
    let d = Decoder {
        data: &[],
        pos: 0,
        qt,
        dc_tabs: [None, None, None, None],
        ac_tabs: [None, None, None, None],
        comps,
        width: ci.width,
        height: ci.height,
        hmax,
        vmax,
        mcus_x: 0,
        mcus_y: 0,
        progressive: false,
        restart_interval: 0,
        adobe_transform: None,
        scans_seen: 1,
        icc_parts: Vec::new(),
        extras: Vec::new(),
    };
    d.finish()
}

/// A fully parsed JPEG whose coefficients can be turned into pixels or recompressed directly.
pub struct Parsed<'a>(Decoder<'a>);

pub fn decode(data: &[u8]) -> Result<Image> {
    parse(data)?.into_image()
}

pub fn parse(data: &[u8]) -> Result<Parsed<'_>> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(Error::invalid("not a JPEG file"));
    }
    let mut d = Decoder {
        data,
        pos: 2,
        qt: [None; 4],
        dc_tabs: [None, None, None, None],
        ac_tabs: [None, None, None, None],
        comps: Vec::new(),
        width: 0,
        height: 0,
        hmax: 1,
        vmax: 1,
        mcus_x: 0,
        mcus_y: 0,
        progressive: false,
        restart_interval: 0,
        adobe_transform: None,
        scans_seen: 0,
        icc_parts: Vec::new(),
        extras: Vec::new(),
    };
    d.run()?;
    if d.scans_seen == 0 || d.comps.is_empty() {
        return Err(Error::invalid("JPEG contains no image data"));
    }
    Ok(Parsed(d))
}

impl Parsed<'_> {
    pub fn info(&self) -> Info {
        let d = &self.0;
        let luma_q = d.qt[d.comps[0].tq].map(|t| {
            let mut t8 = [0u8; 64];
            for (o, v) in t8.iter_mut().zip(t) {
                *o = v.min(255) as u8;
            }
            t8
        });
        let ids: Vec<u8> = d.comps.iter().map(|c| c.id).collect();
        Info {
            width: d.width,
            height: d.height,
            sampling: d.comps.iter().map(|c| (c.h, c.v)).collect(),
            source_quality: luma_q.map_or(100, |t| super::quality::estimate_quality(&t)),
            coded_rgb: d.comps.len() == 3
                && (d.adobe_transform == Some(0) || (d.adobe_transform.is_none() && ids == b"RGB")),
            cmyk: d.comps.len() == 4,
        }
    }

    pub fn into_image(self) -> Result<Image> {
        self.0.finish()
    }

    /// Hand over the quantized coefficients for DCT-domain recompression.
    pub fn into_coefs(self) -> Result<CoefImage> {
        if self.info().coded_rgb {
            return Err(Error::unsupported("RGB-coded JPEG"));
        }
        let d = self.0;
        let mut qtables: [Option<[u8; 64]>; 4] = [None; 4];
        for c in &d.comps {
            let t = d.qt[c.tq].ok_or_else(|| Error::invalid("missing quantization table"))?;
            if t.iter().any(|&v| v > 255 || v == 0) {
                return Err(Error::unsupported("16-bit quantization table"));
            }
            let mut t8 = [0u8; 64];
            for (o, v) in t8.iter_mut().zip(t) {
                *o = v as u8;
            }
            qtables[c.tq] = Some(t8);
        }
        let meta = build_meta(d.icc_parts, d.extras);
        Ok(CoefImage {
            width: d.width,
            height: d.height,
            comps: d
                .comps
                .into_iter()
                .map(|c| CoefComponent { id: c.id, h: c.h, v: c.v, tq: c.tq, blocks_w: c.blocks_w, coefs: c.coefs })
                .collect(),
            qtables,
            meta,
            adobe_transform: d.adobe_transform,
        })
    }
}

/// EXIF orientation (1..=8) of a JPEG, or 1 when absent.
pub fn exif_orientation(data: &[u8]) -> u8 {
    let mut pos = 2;
    while pos + 4 <= data.len() && data[pos] == 0xFF {
        let m = data[pos + 1];
        if m == 0xD8 || m == 0x01 || (0xD0..=0xD7).contains(&m) {
            pos += 2;
            continue;
        }
        if m == 0xDA || m == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        if len < 2 || pos + 2 + len > data.len() {
            break;
        }
        let seg = &data[pos + 4..pos + 2 + len];
        if m == 0xE1 && seg.len() > 14 && &seg[..6] == b"Exif\0\0" {
            return parse_tiff_orientation(&seg[6..]).unwrap_or(1);
        }
        pos += 2 + len;
    }
    1
}

fn parse_tiff_orientation(t: &[u8]) -> Option<u8> {
    let le = match t.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = t.get(o..o + 2)?;
        Some(if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b: [u8; 4] = t.get(o..o + 4)?.try_into().ok()?;
        Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    };
    let ifd = u32_at(4)? as usize;
    let n = u16_at(ifd)? as usize;
    for i in 0..n {
        let e = ifd + 2 + i * 12;
        if u16_at(e)? == 0x0112 {
            let v = u16_at(e + 8)?;
            return (1..=8).contains(&v).then_some(v as u8);
        }
    }
    None
}

impl Decoder<'_> {
    fn next_marker(&mut self) -> Result<u8> {
        let d = self.data;
        loop {
            while self.pos < d.len() && d[self.pos] != 0xFF {
                self.pos += 1;
            }
            while self.pos < d.len() && d[self.pos] == 0xFF {
                self.pos += 1;
            }
            let Some(&m) = d.get(self.pos) else {
                return Err(Error::invalid("unexpected end of JPEG"));
            };
            self.pos += 1;
            if m != 0 && !(0xD0..=0xD7).contains(&m) {
                return Ok(m);
            }
        }
    }

    fn run(&mut self) -> Result<()> {
        loop {
            let m = self.next_marker()?;
            match m {
                0xD9 => return Ok(()),
                0x01 => continue,
                _ => {}
            }
            let len = be16(self.data, self.pos)?;
            if len < 2 || self.pos + len > self.data.len() {
                // Truncated segment: accept what we have if at least one scan was decoded.
                return if self.scans_seen > 0 { Ok(()) } else { Err(Error::invalid("truncated JPEG segment")) };
            }
            let body = &self.data[self.pos + 2..self.pos + len];
            self.pos += len;
            match m {
                0xDB => self.read_dqt(body)?,
                0xC4 => self.read_dht(body)?,
                0xC0..=0xC2 => {
                    if !self.comps.is_empty() {
                        return Err(Error::invalid("multiple SOF markers"));
                    }
                    self.read_sof(body, m == 0xC2)?;
                }
                0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                    return Err(Error::unsupported("JPEG process (lossless/hierarchical/arithmetic)"));
                }
                0xDD => {
                    if body.len() < 2 {
                        return Err(Error::invalid("bad DRI"));
                    }
                    self.restart_interval = be16(body, 0)?;
                }
                0xE2 if body.len() > 14 && &body[..12] == b"ICC_PROFILE\0" => {
                    self.icc_parts.push((body[12], body[14..].to_vec()));
                }
                0xEE if body.len() >= 12 && &body[..5] == b"Adobe" => self.adobe_transform = Some(body[11]),
                0xE1 | 0xED | 0xFE if self.extras.len() < MAX_EXTRA_SEGMENTS => self.extras.push((m, body.to_vec())),
                0xDA => {
                    self.read_sos(body)?;
                    self.scans_seen += 1;
                }
                _ => {}
            }
        }
    }

    fn read_dqt(&mut self, mut b: &[u8]) -> Result<()> {
        while !b.is_empty() {
            let (pq, tq) = (b[0] >> 4, (b[0] & 15) as usize);
            if tq > 3 || pq > 1 {
                return Err(Error::invalid("bad DQT"));
            }
            let need = 1 + 64 * (pq as usize + 1);
            if b.len() < need {
                return Err(Error::invalid("truncated DQT"));
            }
            let mut t = [0u16; 64];
            for k in 0..64 {
                let v = if pq == 0 { b[1 + k] as u16 } else { u16::from_be_bytes([b[1 + 2 * k], b[2 + 2 * k]]) };
                t[ZIGZAG[k]] = v;
            }
            self.qt[tq] = Some(t);
            b = &b[need..];
        }
        Ok(())
    }

    fn read_dht(&mut self, mut b: &[u8]) -> Result<()> {
        while !b.is_empty() {
            if b.len() < 17 {
                return Err(Error::invalid("truncated DHT"));
            }
            let (tc, th) = (b[0] >> 4, (b[0] & 15) as usize);
            if tc > 1 || th > 3 {
                return Err(Error::invalid("bad DHT"));
            }
            let bits: [u8; 16] = b[1..17].try_into().unwrap();
            let total: usize = bits.iter().map(|&x| x as usize).sum();
            if b.len() < 17 + total {
                return Err(Error::invalid("truncated DHT"));
            }
            let t = HuffDec::new(&bits, &b[17..17 + total])?;
            if tc == 0 {
                self.dc_tabs[th] = Some(t);
            } else {
                self.ac_tabs[th] = Some(t);
            }
            b = &b[17 + total..];
        }
        Ok(())
    }

    fn read_sof(&mut self, b: &[u8], progressive: bool) -> Result<()> {
        if b.len() < 6 {
            return Err(Error::invalid("bad SOF"));
        }
        if b[0] != 8 {
            return Err(Error::unsupported(format!("{}-bit JPEG", b[0])));
        }
        let (h, w, nc) = (be16(b, 1)?, be16(b, 3)?, b[5] as usize);
        if w == 0 || h == 0 {
            return Err(Error::invalid("zero JPEG dimension (DNL not supported)"));
        }
        if (w as u64) * (h as u64) > MAX_PIXELS {
            return Err(Error::invalid("unreasonable image dimensions"));
        }
        if nc != 1 && nc != 3 && nc != 4 {
            return Err(Error::unsupported(format!("{nc}-component JPEG")));
        }
        if b.len() < 6 + 3 * nc {
            return Err(Error::invalid("bad SOF"));
        }
        let mut raw = Vec::new();
        for i in 0..nc {
            let c = &b[6 + 3 * i..9 + 3 * i];
            let (hs, vs) = ((c[1] >> 4) as usize, (c[1] & 15) as usize);
            if !(1..=4).contains(&hs) || !(1..=4).contains(&vs) || c[2] > 3 {
                return Err(Error::invalid("bad component description"));
            }
            raw.push((c[0], hs, vs, c[2] as usize));
        }
        self.width = w;
        self.height = h;
        self.progressive = progressive;
        // Single-component images are always 1x1 sampled regardless of the declared factors.
        if nc == 1 {
            raw[0].1 = 1;
            raw[0].2 = 1;
        }
        self.hmax = raw.iter().map(|c| c.1).max().unwrap();
        self.vmax = raw.iter().map(|c| c.2).max().unwrap();
        self.mcus_x = w.div_ceil(8 * self.hmax);
        self.mcus_y = h.div_ceil(8 * self.vmax);
        for (id, hs, vs, tq) in raw {
            let blocks_w = self.mcus_x * hs;
            let blocks_h = self.mcus_y * vs;
            let cw = (w * hs).div_ceil(self.hmax);
            let ch = (h * vs).div_ceil(self.vmax);
            self.comps.push(Component {
                id,
                h: hs,
                v: vs,
                tq,
                blocks_w,
                blocks_h,
                real_bw: cw.div_ceil(8),
                real_bh: ch.div_ceil(8),
                coefs: vec![0i16; blocks_w * blocks_h * 64],
            });
        }
        Ok(())
    }

    fn read_sos(&mut self, b: &[u8]) -> Result<()> {
        if self.comps.is_empty() {
            return Err(Error::invalid("SOS before SOF"));
        }
        let ns = *b.first().ok_or_else(|| Error::invalid("bad SOS"))? as usize;
        if ns == 0 || ns > self.comps.len() || b.len() < 1 + 2 * ns + 3 {
            return Err(Error::invalid("bad SOS"));
        }
        let mut sc = Vec::new();
        for i in 0..ns {
            let (cid, t) = (b[1 + 2 * i], b[2 + 2 * i]);
            let idx = self
                .comps
                .iter()
                .position(|c| c.id == cid)
                .ok_or_else(|| Error::invalid("SOS references unknown component"))?;
            if sc.iter().any(|s: &ScanComp| s.idx == idx) {
                return Err(Error::invalid("duplicate component in scan"));
            }
            sc.push(ScanComp { idx, dc: (t >> 4) as usize & 3, ac: (t & 15) as usize & 3 });
        }
        let t = &b[1 + 2 * ns..];
        let p = if self.progressive {
            let sp =
                ScanParams { ss: t[0] as usize, se: t[1] as usize, ah: (t[2] >> 4) as u32, al: (t[2] & 15) as u32 };
            if sp.ss > sp.se || sp.se > 63 || sp.al > 13 || (sp.ss == 0 && sp.se != 0) || (sp.ss > 0 && ns != 1) {
                return Err(Error::invalid("bad progressive scan parameters"));
            }
            sp
        } else {
            ScanParams { ss: 0, se: 63, ah: 0, al: 0 }
        };
        let end = self.decode_scan(&sc, p)?;
        self.pos = end;
        Ok(())
    }

    fn decode_scan(&mut self, sc: &[ScanComp], p: ScanParams) -> Result<usize> {
        let need_dc = !self.progressive || (p.ss == 0 && p.ah == 0);
        let need_ac = !self.progressive || p.ss > 0;
        let mut dc_t: Vec<Option<HuffDec>> = Vec::new();
        let mut ac_t: Vec<Option<HuffDec>> = Vec::new();
        for s in sc {
            let dc = self.dc_tabs[s.dc].clone();
            let ac = self.ac_tabs[s.ac].clone();
            if need_dc && dc.is_none() || need_ac && ac.is_none() {
                return Err(Error::invalid("scan uses undefined huffman table"));
            }
            dc_t.push(dc);
            ac_t.push(ac);
        }
        let mut br = BitReader::new(self.data, self.pos);
        let mut pred = [0i32; 4];
        let mut eobrun = 0u32;
        let interleaved = sc.len() > 1;
        let (mcus_x, mcus_y) = if interleaved {
            (self.mcus_x, self.mcus_y)
        } else {
            (self.comps[sc[0].idx].real_bw, self.comps[sc[0].idx].real_bh)
        };
        let total = mcus_x * mcus_y;
        let ri = self.restart_interval;
        let progressive = self.progressive;
        for mcu in 0..total {
            if ri > 0 && mcu > 0 && mcu % ri == 0 {
                br.restart();
                pred = [0; 4];
                eobrun = 0;
            }
            let (mx, my) = (mcu % mcus_x, mcu / mcus_x);
            for (si, s) in sc.iter().enumerate() {
                let c = &mut self.comps[s.idx];
                let (bh, bv) = if interleaved { (c.h, c.v) } else { (1, 1) };
                for by in 0..bv {
                    for bx in 0..bh {
                        let (col, row) = (mx * bh + bx, my * bv + by);
                        let off = (row * c.blocks_w + col) * 64;
                        let blk = &mut c.coefs[off..off + 64];
                        let r = if !progressive {
                            decode_sequential(
                                &mut br,
                                dc_t[si].as_ref().unwrap(),
                                ac_t[si].as_ref().unwrap(),
                                blk,
                                &mut pred[si],
                            )
                        } else if p.ss == 0 {
                            if p.ah == 0 {
                                decode_dc_first(&mut br, dc_t[si].as_ref().unwrap(), blk, &mut pred[si], p.al)
                            } else {
                                blk[0] |= (br.bit() as i16) << p.al;
                                Ok(())
                            }
                        } else if p.ah == 0 {
                            decode_ac_first(&mut br, ac_t[si].as_ref().unwrap(), blk, &mut eobrun, p)
                        } else {
                            decode_ac_refine(&mut br, ac_t[si].as_ref().unwrap(), blk, &mut eobrun, p)
                        };
                        r?;
                    }
                }
            }
        }
        Ok(br.pos)
    }

    fn finish(self) -> Result<Image> {
        if self.scans_seen == 0 || self.comps.is_empty() {
            return Err(Error::invalid("JPEG contains no image data"));
        }
        let (w, h) = (self.width, self.height);
        // IDCT every component into an 8-bit plane.
        let mut planes: Vec<(Vec<u8>, usize)> = Vec::new();
        for c in &self.comps {
            let q = self.qt[c.tq].ok_or_else(|| Error::invalid("missing quantization table"))?;
            let mut mult = [[0f32; 8]; 8];
            for v in 0..8 {
                for u in 0..8 {
                    mult[v][u] = q[v * 8 + u] as f32 * AAN[u] * AAN[v];
                }
            }
            let stride = c.blocks_w * 8;
            let mut plane = vec![0u8; stride * c.blocks_h * 8];
            plane.par_chunks_mut(stride * 8).enumerate().for_each(|(by, rows)| {
                for bx in 0..c.blocks_w {
                    let src = &c.coefs[(by * c.blocks_w + bx) * 64..][..64];
                    let mut blk = [[0f32; 8]; 8];
                    for k in 0..64 {
                        let n = ZIGZAG[k];
                        blk[n / 8][n % 8] = src[k] as f32 * mult[n / 8][n % 8];
                    }
                    idct(&mut blk);
                    for y in 0..8 {
                        for x in 0..8 {
                            rows[y * stride + bx * 8 + x] = (round_f32(blk[y][x]) + 128).clamp(0, 255) as u8;
                        }
                    }
                }
            });
            planes.push((plane, stride));
        }
        // Upsample to full resolution.
        let mut full: Vec<Vec<u8>> = Vec::new();
        for (c, (plane, stride)) in self.comps.iter().zip(&planes) {
            let cw = (w * c.h).div_ceil(self.hmax);
            let ch = (h * c.v).div_ceil(self.vmax);
            if !self.hmax.is_multiple_of(c.h) || !self.vmax.is_multiple_of(c.v) {
                return Err(Error::unsupported("non-integral chroma subsampling ratio"));
            }
            full.push(upsample(plane, *stride, cw, ch, self.hmax / c.h, self.vmax / c.v, w, h));
        }
        let meta = build_meta(self.icc_parts, self.extras);
        if full.len() == 1 {
            let mut img = Image::new(w as u32, h as u32, ColorType::Gray, 8, full.pop().unwrap())?;
            img.meta = meta;
            return Ok(img);
        }
        if full.len() == 4 {
            let data = cmyk_to_rgb(&full, self.adobe_transform);
            let mut img = Image::new(w as u32, h as u32, ColorType::Rgb, 8, data)?;
            img.meta = meta;
            return Ok(img);
        }
        let ids = [self.comps[0].id, self.comps[1].id, self.comps[2].id];
        let rgb_direct = self.adobe_transform == Some(0) || (self.adobe_transform.is_none() && ids == *b"RGB");
        let mut data = vec![0u8; w * h * 3];
        data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                let (a, b, c) = (full[0][i] as i32, full[1][i] as i32, full[2][i] as i32);
                let px = &mut row[x * 3..x * 3 + 3];
                if rgb_direct {
                    px.copy_from_slice(&[a as u8, b as u8, c as u8]);
                } else {
                    let (cb, cr) = (b - 128, c - 128);
                    px[0] = (a + ((91881 * cr + 32768) >> 16)).clamp(0, 255) as u8;
                    px[1] = (a - ((22554 * cb + 46802 * cr + 32768) >> 16)).clamp(0, 255) as u8;
                    px[2] = (a + ((116130 * cb + 32768) >> 16)).clamp(0, 255) as u8;
                }
            }
        });
        let mut img = Image::new(w as u32, h as u32, ColorType::Rgb, 8, data)?;
        img.meta = meta;
        Ok(img)
    }
}

/// Convert four decoded planes to RGB. With an Adobe marker the values are stored inverted
/// (255 = no ink) and transform 2 means the first three planes are YCC-coded; without a marker the
/// values are plain ink amounts. This mirrors what libjpeg-based viewers do.
fn cmyk_to_rgb(planes: &[Vec<u8>], adobe_transform: Option<u8>) -> Vec<u8> {
    let n = planes[0].len();
    let mut out = vec![0u8; n * 3];
    let inverted = adobe_transform.is_some();
    let ycck = adobe_transform == Some(2);
    out.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
        let (mut c, mut m, mut y) = (planes[0][i] as i32, planes[1][i] as i32, planes[2][i] as i32);
        let k = planes[3][i] as i32;
        if ycck {
            let (yy, cb, cr) = (c, m - 128, y - 128);
            let r = (yy + ((91881 * cr + 32768) >> 16)).clamp(0, 255);
            let g = (yy - ((22554 * cb + 46802 * cr + 32768) >> 16)).clamp(0, 255);
            let b = (yy + ((116130 * cb + 32768) >> 16)).clamp(0, 255);
            (c, m, y) = (255 - r, 255 - g, 255 - b);
        }
        for (o, v) in px.iter_mut().zip([c, m, y]) {
            // `light` is how much light the ink lets through: the stored value when inverted.
            let (light_v, light_k) = if inverted { (v, k) } else { (255 - v, 255 - k) };
            *o = ((light_v * light_k + 127) / 255) as u8;
        }
    });
    out
}

// ---------------------------------------------------------------- block decoders

fn decode_sequential(br: &mut BitReader, dc: &HuffDec, ac: &HuffDec, blk: &mut [i16], pred: &mut i32) -> Result<()> {
    let t = dc.decode(br)? as u32;
    if t > 16 {
        return Err(Error::invalid("bad DC category"));
    }
    let diff = extend(br.bits(t), t);
    *pred = pred.wrapping_add(diff);
    blk[0] = *pred as i16;
    let mut k = 1;
    while k < 64 {
        let rs = ac.decode(br)?;
        let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
        if s == 0 {
            if r == 15 {
                k += 16;
                continue;
            }
            break;
        }
        k += r;
        if k > 63 {
            return Err(Error::invalid("AC coefficient index out of range"));
        }
        blk[k] = extend(br.bits(s), s) as i16;
        k += 1;
    }
    Ok(())
}

fn decode_dc_first(br: &mut BitReader, dc: &HuffDec, blk: &mut [i16], pred: &mut i32, al: u32) -> Result<()> {
    let t = dc.decode(br)? as u32;
    if t > 16 {
        return Err(Error::invalid("bad DC category"));
    }
    *pred = pred.wrapping_add(extend(br.bits(t), t));
    blk[0] = (*pred << al) as i16;
    Ok(())
}

fn decode_ac_first(br: &mut BitReader, ac: &HuffDec, blk: &mut [i16], eobrun: &mut u32, p: ScanParams) -> Result<()> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = p.ss;
    while k <= p.se {
        let rs = ac.decode(br)?;
        let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
        if s == 0 {
            if r < 15 {
                *eobrun = (1 << r) - 1;
                if r > 0 {
                    *eobrun += br.bits(r as u32);
                }
                break;
            }
            k += 16;
            continue;
        }
        k += r;
        if k > 63 {
            return Err(Error::invalid("AC coefficient index out of range"));
        }
        blk[k] = (extend(br.bits(s), s) << p.al) as i16;
        k += 1;
    }
    Ok(())
}

fn decode_ac_refine(br: &mut BitReader, ac: &HuffDec, blk: &mut [i16], eobrun: &mut u32, p: ScanParams) -> Result<()> {
    let p1: i16 = 1 << p.al;
    let m1: i16 = -1 << p.al;
    let mut k = p.ss;
    if *eobrun == 0 {
        while k <= p.se {
            let rs = ac.decode(br)?;
            let (mut r, s) = ((rs >> 4) as i32, (rs & 15) as u32);
            let mut value = 0i16;
            if s != 0 {
                value = if br.bit() { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = 1 << r;
                if r > 0 {
                    *eobrun += br.bits(r as u32);
                }
                break;
            }
            // Skip over already-nonzero coefficients (refining them) and r zero ones.
            while k <= p.se {
                let c = &mut blk[k];
                if *c != 0 {
                    if br.bit() && (*c & p1) == 0 {
                        *c += if *c >= 0 { p1 } else { m1 };
                    }
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
            }
            if value != 0 {
                if k > 63 {
                    return Err(Error::invalid("AC refine index out of range"));
                }
                blk[k] = value;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= p.se {
            let c = &mut blk[k];
            if *c != 0 && br.bit() && (*c & p1) == 0 {
                *c += if *c >= 0 { p1 } else { m1 };
            }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

// ---------------------------------------------------------------- upsampling

#[allow(clippy::too_many_arguments)]
fn upsample(p: &[u8], stride: usize, cw: usize, ch: usize, fx: usize, fy: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    let px = |x: usize, y: usize| p[y * stride + x] as i32;
    match (fx, fy) {
        (1, 1) => {
            for y in 0..h {
                out[y * w..(y + 1) * w].copy_from_slice(&p[y * stride..y * stride + w]);
            }
        }
        (2, 1) => {
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    let i = x / 2;
                    let near = px(i, y);
                    let far = if x % 2 == 0 { px(i.saturating_sub(1), y) } else { px((i + 1).min(cw - 1), y) };
                    *o = ((3 * near + far + if x % 2 == 0 { 1 } else { 2 }) >> 2) as u8;
                }
            });
        }
        (1, 2) => {
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let j = y / 2;
                let far_y = if y % 2 == 0 { j.saturating_sub(1) } else { (j + 1).min(ch - 1) };
                for (x, o) in row.iter_mut().enumerate() {
                    *o = ((3 * px(x, j) + px(x, far_y) + if y % 2 == 0 { 1 } else { 2 }) >> 2) as u8;
                }
            });
        }
        (2, 2) => {
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let j = y / 2;
                let far_y = if y % 2 == 0 { j.saturating_sub(1) } else { (j + 1).min(ch - 1) };
                let vert = |x: usize| 3 * px(x, j) + px(x, far_y);
                for (x, o) in row.iter_mut().enumerate() {
                    let i = x / 2;
                    let near = vert(i);
                    let far = if x % 2 == 0 { vert(i.saturating_sub(1)) } else { vert((i + 1).min(cw - 1)) };
                    *o = ((3 * near + far + if x % 2 == 0 { 8 } else { 7 }) >> 4) as u8;
                }
            });
        }
        _ => {
            for y in 0..h {
                for x in 0..w {
                    out[y * w + x] = p[(y / fy) * stride + x / fx];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod cmyk_tests {
    use super::cmyk_to_rgb;

    fn px(c: u8, m: u8, y: u8, k: u8, adobe: Option<u8>) -> [u8; 3] {
        let planes = [vec![c], vec![m], vec![y], vec![k]];
        cmyk_to_rgb(&planes, adobe).try_into().unwrap()
    }

    #[test]
    fn adobe_values_are_inverted_ink() {
        assert_eq!(px(255, 255, 255, 255, Some(0)), [255, 255, 255], "no ink is white");
        assert_eq!(px(0, 0, 0, 255, Some(0)), [0, 0, 0], "full CMY is black");
        assert_eq!(px(255, 255, 255, 0, Some(0)), [0, 0, 0], "full K is black");
        assert_eq!(px(0, 255, 255, 255, Some(0)), [0, 255, 255], "full cyan ink blocks red only");
    }

    #[test]
    fn plain_cmyk_without_an_adobe_marker_is_ink_amounts() {
        assert_eq!(px(0, 0, 0, 0, None), [255, 255, 255]);
        assert_eq!(px(0, 0, 0, 255, None), [0, 0, 0]);
        assert_eq!(px(255, 0, 0, 0, None), [0, 255, 255]);
    }

    #[test]
    fn ycck_is_decoded_to_cmy_first() {
        // Y=0 with neutral chroma means "no ink" once libjpeg's 255 - RGB step is applied.
        assert_eq!(px(0, 128, 128, 255, Some(2)), [255, 255, 255]);
        assert_eq!(px(255, 128, 128, 255, Some(2)), [0, 0, 0]);
        assert_eq!(px(0, 128, 128, 0, Some(2)), [0, 0, 0]);
    }
}
