//! Baseline JPEG encoder with per-image optimized Huffman tables.

use super::ZIGZAG;
use super::dct::{AAN, fdct, round_f32};
use super::quality::{chroma_table, luma_table};
use super::rdo::Rdo;
use rayon::prelude::*;
use rc_core::error::{Error, Result};
use rc_core::image::{ColorType, Image, Metadata};
use rc_core::util::huffman::build_lengths;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:4:4 for quality >= 90, otherwise 4:2:0.
    Auto,
    S444,
    S420,
}

#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// 1..=100.
    pub quality: u8,
    pub subsampling: Subsampling,
    /// Background used when flattening transparency.
    pub background: [u8; 3],
    /// Rate-distortion optimized (trellis) quantization.
    pub trellis: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self { quality: 80, subsampling: Subsampling::Auto, background: [255, 255, 255], trellis: true }
    }
}

/// Lambda scale for trellis quantization (relative to the squared median quantizer step).
pub const TRELLIS_KAPPA: f32 = 0.015;

/// `ZZ_T[k]`: index into the transposed (`[u][v]`) DCT output for zigzag position k.
const ZZ_T: [usize; 64] = {
    let mut t = [0usize; 64];
    let mut k = 0;
    while k < 64 {
        let n = ZIGZAG[k];
        t[k] = (n % 8) * 8 + n / 8;
        k += 1;
    }
    t
};

/// One color component's quantized DCT coefficients.
#[derive(Clone)]
pub struct CoefComponent {
    pub id: u8,
    /// Sampling factors.
    pub h: usize,
    pub v: usize,
    /// Quantization table slot (0..=3).
    pub tq: usize,
    /// Width of the block grid (padded to whole MCUs).
    pub blocks_w: usize,
    /// Zigzag-ordered coefficients, 64 per block, raster block order.
    pub coefs: Vec<i16>,
}

/// A JPEG in the DCT domain: everything needed to (re)write the file without touching pixels.
#[derive(Clone)]
pub struct CoefImage {
    pub width: usize,
    pub height: usize,
    pub comps: Vec<CoefComponent>,
    /// Quantization tables in natural order, indexed by slot.
    pub qtables: [Option<[u8; 64]>; 4],
    pub meta: Metadata,
    /// Adobe APP14 colour transform of a 4-component (CMYK/YCCK) file, written back so the colours
    /// keep their meaning. `None` for everything else.
    pub adobe_transform: Option<u8>,
}

impl CoefImage {
    pub(super) fn hv_max(&self) -> (usize, usize) {
        (self.comps.iter().map(|c| c.h).max().unwrap_or(1), self.comps.iter().map(|c| c.v).max().unwrap_or(1))
    }

    pub(super) fn mcus(&self) -> (usize, usize) {
        let (hmax, vmax) = self.hv_max();
        (self.width.div_ceil(8 * hmax), self.height.div_ceil(8 * vmax))
    }

    /// Which standard table a component is quantized toward: 0 luma, 1 chroma. Three-component files have
    /// one luma channel and two chroma channels. In a four-component file every channel of plain CMYK is an
    /// ink amount with no perceptual excuse for coarser steps, so all of them use the luma table; in YCCK the
    /// two color-difference channels are chroma and Y and K are luma-like.
    fn table_class(&self, ci: usize) -> usize {
        if self.comps.len() == 4 {
            usize::from(self.adobe_transform == Some(2) && (ci == 1 || ci == 2))
        } else {
            ci.min(1)
        }
    }

    /// Re-quantize in the DCT domain towards the standard tables at `quality`. Positions where the
    /// source is already coarser than the target are left untouched, so a source at or below the
    /// target quality is transcoded losslessly. With `trellis`, blocks that change are
    /// re-quantized from their exact source values with rate-distortion optimization.
    pub fn requantize(&mut self, quality: u8, trellis: bool) {
        let tables = [luma_table(quality), chroma_table(quality)];
        let mut new_slots: [Option<[u8; 64]>; 4] = [None; 4];
        let mut changed = false;
        // (slot, old/new ratio per zigzag position, new table) per component.
        let mut plan: Vec<(usize, [f32; 64], [u8; 64])> = Vec::new();
        let originals: Vec<Vec<i16>> =
            if trellis { self.comps.iter().map(|c| c.coefs.clone()).collect() } else { Vec::new() };
        let classes: Vec<usize> = (0..self.comps.len()).map(|ci| self.table_class(ci)).collect();
        for (ci, c) in self.comps.iter_mut().enumerate() {
            let old = self.qtables[c.tq].unwrap_or([1; 64]);
            let mut new = tables[classes[ci]];
            for (n, o) in new.iter_mut().zip(&old) {
                *n = (*n).max(*o);
            }
            changed |= new != old;
            // Scale factor per zigzag position.
            let mut ratio = [0f32; 64];
            for k in 0..64 {
                ratio[k] = old[ZIGZAG[k]] as f32 / new[ZIGZAG[k]] as f32;
            }
            c.coefs.par_chunks_mut(64 * 64).for_each(|chunk| {
                for blk in chunk.chunks_exact_mut(64) {
                    for k in 0..64 {
                        blk[k] = round_f32(blk[k] as f32 * ratio[k]).clamp(-2047, 2047) as i16;
                    }
                }
            });
            // Components that end up with the same table share a slot; a different table gets the next one.
            let slot = match new_slots.iter().position(|t| *t == Some(new)) {
                Some(s) => s,
                None => {
                    let s = new_slots.iter().position(Option::is_none).expect("at most four components");
                    new_slots[s] = Some(new);
                    s
                }
            };
            c.tq = slot;
            plan.push((slot, ratio, new));
        }
        self.qtables = new_slots;
        if trellis && changed {
            // The rate model follows the Huffman table a component is written with (the first component's, or
            // the shared one of all the others); the quantizer follows the component's own table.
            let lens = ac_code_lengths(self);
            let rdos: Vec<Rdo> =
                plan.iter().enumerate().map(|(ci, p)| Rdo::new(&p.2, TRELLIS_KAPPA, &lens[ci.min(1)])).collect();
            for (ci, ((c, orig), (_, ratio, _))) in self.comps.iter_mut().zip(&originals).zip(&plan).enumerate() {
                let rdo = &rdos[ci];
                c.coefs.par_chunks_mut(64 * 64).zip(orig.par_chunks(64 * 64)).for_each(|(out, inp)| {
                    for (o, i) in out.chunks_exact_mut(64).zip(inp.chunks_exact(64)) {
                        let mut a = [0f32; 64];
                        for k in 0..64 {
                            a[k] = i[k] as f32 * ratio[k];
                        }
                        rdo.quantize_ac(&a, o);
                    }
                });
            }
        }
    }
}

/// Divisors in the transposed DCT-output layout (folding in the AAN scale factors).
fn fdct_divisors(q: &[u8; 64]) -> [f32; 64] {
    let mut d = [0f32; 64];
    for v in 0..8 {
        for u in 0..8 {
            d[u * 8 + v] = 1.0 / (q[v * 8 + u] as f32 * AAN[u] * AAN[v] * 8.0);
        }
    }
    d
}

fn quantize_plane(plane: &[u8], stride: usize, bw: usize, bh: usize, div: &[f32; 64], rdo: Option<&Rdo>) -> Vec<i16> {
    let mut out = vec![0i16; bw * bh * 64];
    out.par_chunks_mut(bw * 64).enumerate().for_each(|(by, row)| {
        for bx in 0..bw {
            let mut blk = [[0f32; 8]; 8];
            for (y, line) in blk.iter_mut().enumerate() {
                let src = &plane[(by * 8 + y) * stride + bx * 8..][..8];
                for x in 0..8 {
                    line[x] = src[x] as f32 - 128.0;
                }
            }
            fdct(&mut blk);
            let flat = blk.as_flattened();
            let dst = &mut row[bx * 64..bx * 64 + 64];
            match rdo {
                None => {
                    for k in 0..64 {
                        let f = ZZ_T[k];
                        dst[k] = round_f32(flat[f] * div[f]).clamp(-2047, 2047) as i16;
                    }
                }
                Some(r) => {
                    let mut a = [0f32; 64];
                    for k in 0..64 {
                        let f = ZZ_T[k];
                        a[k] = flat[f] * div[f];
                    }
                    dst[0] = round_f32(a[0]).clamp(-2047, 2047) as i16;
                    r.quantize_ac(&a, dst);
                }
            }
        }
    });
    out
}

/// Huffman code lengths of every AC symbol for luma (index 0) and chroma (index 1), measured on
/// the coefficients as they currently stand. Used as the rate model for trellis quantization.
fn ac_code_lengths(ci: &CoefImage) -> [[u8; 256]; 2] {
    let (mcus_x, mcus_y) = ci.mcus();
    let mut stats = Stats { dc: [[0; 256]; 2], ac: [[0; 256]; 2], raw_bits: 0 };
    scan(&ci.comps, mcus_x, mcus_y, &mut stats);
    [optimal_table(&stats.ac[0]).size, optimal_table(&stats.ac[1]).size]
}

// ---------------------------------------------------------------- Huffman

pub(super) struct HuffTable {
    /// Number of codes of each length 1..=16 (index 0 unused).
    pub(super) bits: [u8; 17],
    pub(super) vals: Vec<u8>,
    pub(super) code: [u16; 256],
    pub(super) size: [u8; 256],
}

pub(super) fn optimal_table(freq: &[u32; 256]) -> HuffTable {
    // Reserve the all-ones code with a dummy symbol of frequency 1 (see build_lengths tie-breaking).
    let mut f = freq.to_vec();
    f.push(1);
    let lens = build_lengths(&f, 16);
    let mut syms: Vec<(u8, u8)> = (0..256).filter(|&s| lens[s] > 0).map(|s| (lens[s], s as u8)).collect();
    if syms.is_empty() {
        syms.push((1, 0));
    }
    syms.sort_unstable();
    let mut t = HuffTable { bits: [0; 17], vals: Vec::new(), code: [0; 256], size: [0; 256] };
    let mut code = 0u32;
    let mut prev_len = syms[0].0;
    for &(len, s) in &syms {
        code <<= len - prev_len;
        prev_len = len;
        t.bits[len as usize] += 1;
        t.vals.push(s);
        t.code[s as usize] = code as u16;
        t.size[s as usize] = len;
        code += 1;
    }
    t
}

trait Sink {
    fn dc(&mut self, tbl: usize, cat: u32, extra: u32);
    fn ac(&mut self, tbl: usize, sym: u8, extra: u32, nbits: u32);
}

struct Stats {
    dc: [[u32; 256]; 2],
    ac: [[u32; 256]; 2],
    /// Raw (non-Huffman) bits that would be emitted.
    raw_bits: u64,
}

impl Sink for Stats {
    #[inline(always)]
    fn dc(&mut self, tbl: usize, cat: u32, _: u32) {
        self.dc[tbl][cat as usize] += 1;
        self.raw_bits += cat as u64;
    }
    #[inline(always)]
    fn ac(&mut self, tbl: usize, sym: u8, _: u32, nbits: u32) {
        self.ac[tbl][sym as usize] += 1;
        self.raw_bits += nbits as u64;
    }
}

pub(super) struct MsbWriter {
    pub(super) out: Vec<u8>,
    pub(super) acc: u64,
    pub(super) n: u32,
}

impl MsbWriter {
    #[inline(always)]
    pub(super) fn put(&mut self, v: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.acc = (self.acc << len) | (v as u64 & ((1u64 << len) - 1));
        self.n += len;
        while self.n >= 8 {
            let byte = (self.acc >> (self.n - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0);
            }
            self.n -= 8;
        }
    }

    pub(super) fn finish(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1 << pad) - 1, pad);
        }
    }
}

struct Writer<'a> {
    w: MsbWriter,
    dc: &'a [HuffTable; 2],
    ac: &'a [HuffTable; 2],
}

impl Sink for Writer<'_> {
    #[inline(always)]
    fn dc(&mut self, tbl: usize, cat: u32, extra: u32) {
        let t = &self.dc[tbl];
        self.w.put(t.code[cat as usize] as u32, t.size[cat as usize] as u32);
        self.w.put(extra, cat);
    }
    #[inline(always)]
    fn ac(&mut self, tbl: usize, sym: u8, extra: u32, nbits: u32) {
        let t = &self.ac[tbl];
        self.w.put(t.code[sym as usize] as u32, t.size[sym as usize] as u32);
        self.w.put(extra, nbits);
    }
}

#[inline(always)]
fn category(v: i32) -> (u32, u32) {
    let a = v.unsigned_abs();
    let cat = 32 - a.leading_zeros();
    let extra = if v < 0 { (v - 1) as u32 } else { v as u32 } & ((1u32 << cat) - 1);
    (cat, extra)
}

#[inline(always)]
fn code_block<S: Sink>(s: &mut S, blk: &[i16], pred: &mut i16, tbl: usize) {
    let diff = blk[0] as i32 - *pred as i32;
    *pred = blk[0];
    let (cat, extra) = category(diff);
    s.dc(tbl, cat, extra);
    let mut run = 0u32;
    for &c in &blk[1..64] {
        if c == 0 {
            run += 1;
            continue;
        }
        while run >= 16 {
            s.ac(tbl, 0xF0, 0, 0);
            run -= 16;
        }
        let (cat, extra) = category(c as i32);
        s.ac(tbl, ((run << 4) | cat) as u8, extra, cat);
        run = 0;
    }
    if run > 0 {
        s.ac(tbl, 0x00, 0, 0);
    }
}

fn scan<S: Sink>(comps: &[CoefComponent], mcus_x: usize, mcus_y: usize, sink: &mut S) {
    let mut pred = [0i16; 4];
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            for (ci, c) in comps.iter().enumerate() {
                for by in 0..c.v {
                    for bx in 0..c.h {
                        let b = (my * c.v + by) * c.blocks_w + mx * c.h + bx;
                        code_block(sink, &c.coefs[b * 64..b * 64 + 64], &mut pred[ci], ci.min(1));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- output

pub(super) fn segment(out: &mut Vec<u8>, marker: u8, body: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(body);
}

pub(super) fn write_dht(out: &mut Vec<u8>, class_id: u8, t: &HuffTable) {
    let mut body = vec![class_id];
    body.extend_from_slice(&t.bits[1..17]);
    body.extend_from_slice(&t.vals);
    segment(out, 0xC4, &body);
}

/// Encode as baseline JPEG. Transparency is flattened onto `opts.background`.
pub fn encode(img: &Image, opts: &EncodeOptions) -> Result<Vec<u8>> {
    if img.width > 65535 || img.height > 65535 {
        return Err(Error::unsupported("JPEG dimensions are limited to 65535 pixels"));
    }
    let src = img.flatten(opts.background);
    Ok(write_best(&build_coefs(&src, opts)))
}

/// Quantize 8-bit Gray/Rgb pixels into DCT coefficients at `opts.quality` (color conversion,
/// subsampling, DCT, plain then trellis quantization), without entropy coding.
pub fn build_coefs(src: &Image, opts: &EncodeOptions) -> CoefImage {
    let (w, h) = (src.width as usize, src.height as usize);
    let gray = src.color == ColorType::Gray;
    let sub420 = !gray
        && match opts.subsampling {
            Subsampling::S420 => true,
            Subsampling::S444 => false,
            Subsampling::Auto => opts.quality < 90,
        };
    let (hmax, vmax) = if sub420 { (2, 2) } else { (1, 1) };
    let (mcu_w, mcu_h) = (8 * hmax, 8 * vmax);
    let mcus_x = w.div_ceil(mcu_w);
    let mcus_y = h.div_ceil(mcu_h);
    let (pw, ph) = (mcus_x * mcu_w, mcus_y * mcu_h);

    let qy = luma_table(opts.quality);
    let qc = chroma_table(opts.quality);

    let mut comps: Vec<CoefComponent> = Vec::new();
    // (plane, stride, blocks wide, blocks high, quant-class) for a possible second, RDO pass.
    let mut planes: Vec<(Vec<u8>, usize, usize, usize, usize)> = Vec::new();
    if gray {
        let mut y = vec![0u8; pw * ph];
        y.par_chunks_mut(pw).enumerate().for_each(|(r, row)| {
            let sr = r.min(h - 1);
            for (x, o) in row.iter_mut().enumerate() {
                *o = src.data[sr * w + x.min(w - 1)];
            }
        });
        let coefs = quantize_plane(&y, pw, pw / 8, ph / 8, &fdct_divisors(&qy), None);
        comps.push(CoefComponent { id: 1, h: 1, v: 1, tq: 0, blocks_w: pw / 8, coefs });
        planes.push((y, pw, pw / 8, ph / 8, 0));
    } else {
        let mut yp = vec![0u8; pw * ph];
        let mut cbp = vec![0u8; pw * ph];
        let mut crp = vec![0u8; pw * ph];
        yp.par_chunks_mut(pw).zip(cbp.par_chunks_mut(pw)).zip(crp.par_chunks_mut(pw)).enumerate().for_each(
            |(r, ((yr, cbr), crr))| {
                let sr = r.min(h - 1);
                for x in 0..pw {
                    let i = (sr * w + x.min(w - 1)) * 3;
                    let (rr, gg, bb) = (src.data[i] as i32, src.data[i + 1] as i32, src.data[i + 2] as i32);
                    yr[x] = ((19595 * rr + 38470 * gg + 7471 * bb + 32768) >> 16) as u8;
                    cbr[x] = ((-11059 * rr - 21709 * gg + 32768 * bb + (128 << 16) + 32767) >> 16) as u8;
                    crr[x] = ((32768 * rr - 27439 * gg - 5329 * bb + (128 << 16) + 32767) >> 16) as u8;
                }
            },
        );
        let (cb, cr, cw, ch) = if sub420 {
            let (cw, ch) = (pw / 2, ph / 2);
            let down = |p: &[u8]| -> Vec<u8> {
                let mut o = vec![0u8; cw * ch];
                o.par_chunks_mut(cw).enumerate().for_each(|(r, row)| {
                    for (x, d) in row.iter_mut().enumerate() {
                        let i = 2 * r * pw + 2 * x;
                        let s = p[i] as u32 + p[i + 1] as u32 + p[i + pw] as u32 + p[i + pw + 1] as u32;
                        *d = ((s + 2) >> 2) as u8;
                    }
                });
                o
            };
            (down(&cbp), down(&crp), cw, ch)
        } else {
            (cbp, crp, pw, ph)
        };
        let dy = fdct_divisors(&qy);
        let dc = fdct_divisors(&qc);
        let (ycoefs, (cbcoefs, crcoefs)) = rayon::join(
            || quantize_plane(&yp, pw, pw / 8, ph / 8, &dy, None),
            || {
                rayon::join(
                    || quantize_plane(&cb, cw, cw / 8, ch / 8, &dc, None),
                    || quantize_plane(&cr, cw, cw / 8, ch / 8, &dc, None),
                )
            },
        );
        comps.push(CoefComponent { id: 1, h: hmax, v: vmax, tq: 0, blocks_w: pw / 8, coefs: ycoefs });
        comps.push(CoefComponent { id: 2, h: 1, v: 1, tq: 1, blocks_w: cw / 8, coefs: cbcoefs });
        comps.push(CoefComponent { id: 3, h: 1, v: 1, tq: 1, blocks_w: cw / 8, coefs: crcoefs });
        planes.push((yp, pw, pw / 8, ph / 8, 0));
        planes.push((cb, cw, cw / 8, ch / 8, 1));
        planes.push((cr, cw, cw / 8, ch / 8, 1));
    }

    let mut ci = CoefImage {
        width: w,
        height: h,
        comps,
        qtables: [Some(qy), if gray { None } else { Some(qc) }, None, None],
        meta: src.meta.clone(),
        adobe_transform: None,
    };
    if opts.trellis && opts.quality < 100 {
        // Second pass: re-quantize every block with rate-distortion optimization, using code
        // lengths measured on the plain pass.
        let lens = ac_code_lengths(&ci);
        let tables = [qy, qc];
        let rdos = [Rdo::new(&tables[0], TRELLIS_KAPPA, &lens[0]), Rdo::new(&tables[1], TRELLIS_KAPPA, &lens[1])];
        let divs = [fdct_divisors(&tables[0]), fdct_divisors(&tables[1])];
        for (comp, (plane, stride, bw, bh, class)) in ci.comps.iter_mut().zip(&planes) {
            comp.coefs = quantize_plane(plane, *stride, *bw, *bh, &divs[*class], Some(&rdos[*class]));
        }
    }
    ci
}

/// Serialize as whichever of baseline and progressive is smaller. Both sizes are estimated from
/// symbol statistics first, so only the winner is actually written.
pub fn write_best(ci: &CoefImage) -> Vec<u8> {
    let (prog_est, script) = super::progressive::plan(ci);
    if baseline_estimate(ci) <= prog_est { write_jpeg(ci) } else { super::progressive::emit(ci, &script) }
}

/// Estimated size in bytes of the baseline encoding.
fn baseline_estimate(ci: &CoefImage) -> u64 {
    let (mcus_x, mcus_y) = ci.mcus();
    let mut stats = Stats { dc: [[0; 256]; 2], ac: [[0; 256]; 2], raw_bits: 0 };
    scan(&ci.comps, mcus_x, mcus_y, &mut stats);
    let mut bits = stats.raw_bits;
    let mut total = 0u64;
    for class in [&stats.dc, &stats.ac] {
        for freq in class.iter().take(if ci.comps.len() == 1 { 1 } else { 2 }) {
            let t = optimal_table(freq);
            for (sym, &f) in freq.iter().enumerate() {
                bits += f as u64 * t.size[sym] as u64;
            }
            total += 5 + 16 + t.vals.len() as u64;
        }
    }
    total + bits.div_ceil(8) + 20
}

/// Entropy-code and serialize a [`CoefImage`] as baseline JPEG with optimized Huffman tables.
pub fn write_jpeg(ci: &CoefImage) -> Vec<u8> {
    let comps = &ci.comps;
    let (mcus_x, mcus_y) = ci.mcus();
    let gray = comps.len() == 1;
    let ntables = if gray { 1 } else { 2 };
    // Pass 1: symbol statistics -> optimal Huffman tables.
    let mut stats = Stats { dc: [[0; 256]; 2], ac: [[0; 256]; 2], raw_bits: 0 };
    scan(comps, mcus_x, mcus_y, &mut stats);
    let dc_tables = [optimal_table(&stats.dc[0]), optimal_table(&stats.dc[1])];
    let ac_tables = [optimal_table(&stats.ac[0]), optimal_table(&stats.ac[1])];

    // Pass 2: entropy-code.
    let mut writer = Writer {
        w: MsbWriter { out: Vec::with_capacity(ci.width * ci.height / 4 + 1024), acc: 0, n: 0 },
        dc: &dc_tables,
        ac: &ac_tables,
    };
    scan(comps, mcus_x, mcus_y, &mut writer);
    writer.w.finish();
    let entropy = writer.w.out;

    let mut out = write_headers(ci, 0xC0, entropy.len() + 1024);
    for t in 0..ntables {
        write_dht(&mut out, t as u8, &dc_tables[t]);
        write_dht(&mut out, 0x10 | t as u8, &ac_tables[t]);
    }
    let mut sos = vec![comps.len() as u8];
    for (i, c) in comps.iter().enumerate() {
        sos.extend_from_slice(&[c.id, ((i.min(1)) << 4 | i.min(1)) as u8]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    segment(&mut out, 0xDA, &sos);
    out.extend_from_slice(&entropy);
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

/// SOI, JFIF, ICC, DQT and SOF segments. `sof` is 0xC0 (baseline) or 0xC2 (progressive).
pub(super) fn write_headers(ci: &CoefImage, sof_marker: u8, capacity: usize) -> Vec<u8> {
    let comps = &ci.comps;
    let (w, h) = (ci.width, ci.height);
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&[0xFF, 0xD8]);
    segment(&mut out, 0xE0, &[b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);
    if let (Some(t), 4) = (ci.adobe_transform, comps.len()) {
        // "Adobe", version 100, two flag words, colour transform.
        segment(&mut out, 0xEE, &[b'A', b'd', b'o', b'b', b'e', 0, 100, 0, 0, 0, 0, t]);
    }
    // Descriptive metadata (only present when the caller asked to keep it). A segment holds at most
    // 65533 bytes; anything larger (an EXIF block with a big thumbnail, say) cannot be carried.
    let fits = |body: &[u8]| body.len() <= 65533;
    if let Some(exif) = &ci.meta.exif {
        let body = [&b"Exif\0\0"[..], exif].concat();
        if fits(&body) {
            segment(&mut out, 0xE1, &body);
        }
    }
    if let Some(xmp) = &ci.meta.xmp {
        let body = [&b"http://ns.adobe.com/xap/1.0/\0"[..], xmp].concat();
        if fits(&body) {
            segment(&mut out, 0xE1, &body);
        }
    }
    for (marker, body) in &ci.meta.jpeg_extra {
        if fits(body) {
            segment(&mut out, *marker, body);
        }
    }
    if let Some(icc) = &ci.meta.icc {
        let chunks: Vec<&[u8]> = icc.chunks(65519).collect();
        for (i, c) in chunks.iter().enumerate() {
            let mut body = b"ICC_PROFILE\0".to_vec();
            body.extend_from_slice(&[(i + 1) as u8, chunks.len() as u8]);
            body.extend_from_slice(c);
            segment(&mut out, 0xE2, &body);
        }
    }
    let mut dqt = Vec::new();
    for (slot, q) in ci.qtables.iter().enumerate() {
        if let Some(q) = q {
            dqt.push(slot as u8);
            dqt.extend(ZIGZAG.iter().map(|&n| q[n]));
        }
    }
    segment(&mut out, 0xDB, &dqt);
    let mut sof = vec![8];
    sof.extend_from_slice(&(h as u16).to_be_bytes());
    sof.extend_from_slice(&(w as u16).to_be_bytes());
    sof.push(comps.len() as u8);
    for c in comps {
        sof.extend_from_slice(&[c.id, (c.h << 4 | c.v) as u8, c.tq as u8]);
    }
    segment(&mut out, sof_marker, &sof);
    out
}
