//! Progressive JPEG writer (spectral selection + successive approximation) with per-scan
//! optimized Huffman tables. Follows the structure of libjpeg's `jcphuff.c`.

use super::encode::{CoefImage, HuffTable, MsbWriter, optimal_table, segment, write_dht, write_headers};
use rayon::prelude::*;

/// Sink for the entropy-coder's output: Huffman-coded symbols and raw bits.
trait ScanSink {
    /// `class`: 0 = DC table, 1 = AC table.
    fn sym(&mut self, class: usize, slot: usize, sym: u8);
    fn bits(&mut self, v: u32, n: u32);
}

#[derive(Clone)]
pub struct Scan {
    comps: Vec<usize>,
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

struct Stats {
    freq: [[[u32; 256]; 2]; 2],
    /// Raw (non-Huffman) bits emitted.
    raw_bits: u64,
}

impl ScanSink for Stats {
    fn sym(&mut self, class: usize, slot: usize, sym: u8) {
        self.freq[class][slot][sym as usize] += 1;
    }
    fn bits(&mut self, _: u32, n: u32) {
        self.raw_bits += n as u64;
    }
}

struct Emit<'a> {
    w: MsbWriter,
    tables: &'a [[Option<HuffTable>; 2]; 2],
}

impl ScanSink for Emit<'_> {
    fn sym(&mut self, class: usize, slot: usize, sym: u8) {
        let t = self.tables[class][slot].as_ref().expect("table built from statistics");
        self.w.put(t.code[sym as usize] as u32, t.size[sym as usize] as u32);
    }
    fn bits(&mut self, v: u32, n: u32) {
        self.w.put(v, n);
    }
}

/// Entropy-coder state that persists across the blocks of one scan.
struct State {
    last_dc: [i32; 4],
    eobrun: u32,
    /// Correction bits owed to the pending end-of-band run.
    pending: Vec<u8>,
    /// Correction bits for the current block.
    br: Vec<u8>,
}

const MAX_CORR_BITS: usize = 1000;

fn nbits(v: u32) -> u32 {
    32 - v.leading_zeros()
}

fn emit_eobrun<S: ScanSink>(st: &mut State, sink: &mut S) {
    if st.eobrun > 0 {
        let nb = nbits(st.eobrun) - 1;
        sink.sym(1, 0, (nb << 4) as u8);
        if nb > 0 {
            sink.bits(st.eobrun & ((1 << nb) - 1), nb);
        }
        st.eobrun = 0;
        for b in st.pending.drain(..) {
            sink.bits(b as u32, 1);
        }
    }
}

fn dc_first<S: ScanSink>(st: &mut State, sink: &mut S, blk: &[i16], comp: usize, slot: usize, al: u32) {
    let v = (blk[0] >> al) as i32;
    let diff = v - st.last_dc[comp];
    st.last_dc[comp] = v;
    let nb = nbits(diff.unsigned_abs());
    sink.sym(0, slot, nb as u8);
    if nb > 0 {
        let extra = if diff < 0 { (diff - 1) as u32 } else { diff as u32 } & ((1 << nb) - 1);
        sink.bits(extra, nb);
    }
}

fn dc_refine<S: ScanSink>(sink: &mut S, blk: &[i16], al: u32) {
    sink.bits(((blk[0] >> al) & 1) as u32, 1);
}

fn ac_first<S: ScanSink>(st: &mut State, sink: &mut S, blk: &[i16], sc: &Scan) {
    let mut r = 0u32;
    for k in sc.ss..=sc.se {
        let v = blk[k] as i32;
        if v == 0 {
            r += 1;
            continue;
        }
        let (mag, ext) = if v < 0 {
            let m = (-v) >> sc.al;
            (m, !m)
        } else {
            let m = v >> sc.al;
            (m, m)
        };
        if mag == 0 {
            r += 1;
            continue;
        }
        emit_eobrun(st, sink);
        while r > 15 {
            sink.sym(1, 0, 0xF0);
            r -= 16;
        }
        let nb = nbits(mag as u32);
        sink.sym(1, 0, ((r << 4) + nb) as u8);
        sink.bits(ext as u32 & ((1 << nb) - 1), nb);
        r = 0;
    }
    if r > 0 {
        st.eobrun += 1;
        if st.eobrun == 0x7FFF {
            emit_eobrun(st, sink);
        }
    }
}

fn ac_refine<S: ScanSink>(st: &mut State, sink: &mut S, blk: &[i16], sc: &Scan) {
    let mut abs = [0i32; 64];
    let mut eob = 0usize;
    for k in sc.ss..=sc.se {
        let t = (blk[k] as i32).abs() >> sc.al;
        abs[k] = t;
        if t == 1 {
            eob = k;
        }
    }
    let mut r = 0u32;
    st.br.clear();
    for k in sc.ss..=sc.se {
        let t = abs[k];
        if t == 0 {
            r += 1;
            continue;
        }
        while r > 15 && k <= eob {
            emit_eobrun(st, sink);
            sink.sym(1, 0, 0xF0);
            r -= 16;
            for b in st.br.drain(..) {
                sink.bits(b as u32, 1);
            }
        }
        if t > 1 {
            st.br.push((t & 1) as u8);
            continue;
        }
        emit_eobrun(st, sink);
        sink.sym(1, 0, ((r << 4) + 1) as u8);
        sink.bits(if blk[k] < 0 { 0 } else { 1 }, 1);
        for b in st.br.drain(..) {
            sink.bits(b as u32, 1);
        }
        r = 0;
    }
    if r > 0 || !st.br.is_empty() {
        st.eobrun += 1;
        let br = std::mem::take(&mut st.br);
        st.pending.extend_from_slice(&br);
        st.br = br;
        st.br.clear();
        if st.eobrun == 0x7FFF || st.pending.len() > MAX_CORR_BITS - 63 {
            emit_eobrun(st, sink);
        }
    }
}

fn run_scan<S: ScanSink>(ci: &CoefImage, sc: &Scan, sink: &mut S) {
    let mut st = State { last_dc: [0; 4], eobrun: 0, pending: Vec::new(), br: Vec::new() };
    let (hmax, vmax) = ci.hv_max();
    let dc_scan = sc.ss == 0;
    let visit = |st: &mut State, sink: &mut S, ci_idx: usize, blk: &[i16]| {
        let slot = ci_idx.min(1);
        if dc_scan {
            if sc.ah == 0 {
                dc_first(st, sink, blk, ci_idx, slot, sc.al);
            } else {
                dc_refine(sink, blk, sc.al);
            }
        } else if sc.ah == 0 {
            ac_first(st, sink, blk, sc);
        } else {
            ac_refine(st, sink, blk, sc);
        }
    };
    if sc.comps.len() > 1 {
        let (mcus_x, mcus_y) = ci.mcus();
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                for &k in &sc.comps {
                    let c = &ci.comps[k];
                    for by in 0..c.v {
                        for bx in 0..c.h {
                            let b = (my * c.v + by) * c.blocks_w + mx * c.h + bx;
                            visit(&mut st, sink, k, &c.coefs[b * 64..b * 64 + 64]);
                        }
                    }
                }
            }
        }
    } else {
        let k = sc.comps[0];
        let c = &ci.comps[k];
        // Non-interleaved scans cover only the blocks that overlap the image.
        let real_w = (ci.width * c.h).div_ceil(hmax).div_ceil(8);
        let real_h = (ci.height * c.v).div_ceil(vmax).div_ceil(8);
        for by in 0..real_h {
            for bx in 0..real_w {
                let b = by * c.blocks_w + bx;
                visit(&mut st, sink, k, &c.coefs[b * 64..b * 64 + 64]);
            }
        }
    }
    if !dc_scan {
        emit_eobrun(&mut st, sink);
    }
}

/// A family of scan layouts: DC precision, luma band split, luma/chroma successive-approximation
/// depth. Only the *partition* of coefficients into scans affects size, not their order, so every
/// layout lists first scans before their refinements. libjpeg's and mozjpeg's default scripts are
/// members of the family.
fn scripts(ncomp: usize) -> Vec<Vec<Scan>> {
    let s = |comps: &[usize], ss, se, ah, al| Scan { comps: comps.to_vec(), ss, se, ah, al };
    let all: Vec<usize> = (0..ncomp).collect();
    // Every component after the first is coded like chroma (Cb/Cr, or the M/Y/K planes of CMYK).
    let chroma_als: &[u32] = if ncomp >= 3 { &[0, 1] } else { &[0] };
    let chroma_splits: &[Option<usize>] = if ncomp >= 3 { &[None, Some(5)] } else { &[None] };
    let mut out = Vec::new();
    for dc_al in [0u32, 1] {
        for split in [None, Some(5usize), Some(8)] {
            for luma_al in [0u32, 1, 2] {
                for (&chroma_al, chroma_split) in
                    chroma_als.iter().flat_map(|a| chroma_splits.iter().map(move |c| (a, *c)))
                {
                    let mut v = vec![s(&all, 0, 0, 0, dc_al)];
                    match split {
                        None => v.push(s(&[0], 1, 63, 0, luma_al)),
                        Some(k) => {
                            v.push(s(&[0], 1, k, 0, luma_al));
                            v.push(s(&[0], k + 1, 63, 0, luma_al));
                        }
                    }
                    if ncomp >= 3 {
                        for c in 1..ncomp {
                            match chroma_split {
                                None => v.push(s(&[c], 1, 63, 0, chroma_al)),
                                Some(k) => {
                                    v.push(s(&[c], 1, k, 0, chroma_al));
                                    v.push(s(&[c], k + 1, 63, 0, chroma_al));
                                }
                            }
                        }
                    }
                    if dc_al > 0 {
                        v.push(s(&all, 0, 0, 1, 0));
                    }
                    for ah in (1..=luma_al).rev() {
                        v.push(s(&[0], 1, 63, ah, ah - 1));
                    }
                    for ah in (1..=chroma_al).rev() {
                        for c in 1..ncomp {
                            v.push(s(&[c], 1, 63, ah, ah - 1));
                        }
                    }
                    out.push(v);
                }
            }
        }
    }
    out
}

fn build_tables(stats: &Stats) -> [[Option<HuffTable>; 2]; 2] {
    let mut tables: [[Option<HuffTable>; 2]; 2] = [[None, None], [None, None]];
    for class in 0..2 {
        for slot in 0..2 {
            if stats.freq[class][slot].iter().any(|&f| f > 0) {
                tables[class][slot] = Some(optimal_table(&stats.freq[class][slot]));
            }
        }
    }
    tables
}

/// Size in bytes of one scan (DHT + SOS + entropy data), without emitting it.
fn scan_cost(ci: &CoefImage, sc: &Scan) -> u64 {
    let mut stats = Stats { freq: [[[0; 256]; 2]; 2], raw_bits: 0 };
    run_scan(ci, sc, &mut stats);
    let tables = build_tables(&stats);
    let mut bits = stats.raw_bits;
    let mut total = 0u64;
    for class in 0..2 {
        for slot in 0..2 {
            if let Some(t) = &tables[class][slot] {
                for (sym, &f) in stats.freq[class][slot].iter().enumerate() {
                    bits += f as u64 * t.size[sym] as u64;
                }
                total += 5 + 16 + t.vals.len() as u64; // DHT segment
            }
        }
    }
    total + bits.div_ceil(8) + 14 // data + SOS segment
}

type ScanKey = (Vec<usize>, usize, usize, u32, u32);

fn key(sc: &Scan) -> ScanKey {
    (sc.comps.clone(), sc.ss, sc.se, sc.ah, sc.al)
}

/// The best scan layout and its estimated size. Scans are coded independently (each has its own
/// Huffman tables), so every distinct scan is costed once and layouts are compared by summing.
pub fn plan(ci: &CoefImage) -> (u64, Vec<Scan>) {
    let candidates = scripts(ci.comps.len());
    let mut distinct: Vec<&Scan> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for sc in candidates.iter().flatten() {
        if seen.insert(key(sc)) {
            distinct.push(sc);
        }
    }
    let costs: std::collections::HashMap<ScanKey, u64> =
        distinct.par_iter().map(|sc| (key(sc), scan_cost(ci, sc))).collect();
    let total = |script: &Vec<Scan>| script.iter().map(|sc| costs[&key(sc)]).sum::<u64>();
    let best = candidates.iter().min_by_key(|s| total(s)).expect("at least one scan script");
    (total(best), best.clone())
}

/// Entropy-code every scan of `script` (own optimized Huffman tables per scan).
fn encode_script(ci: &CoefImage, script: &[Scan]) -> Vec<u8> {
    let mut out = Vec::new();
    for sc in script {
        // Pass 1: statistics for this scan's tables.
        let mut stats = Stats { freq: [[[0; 256]; 2]; 2], raw_bits: 0 };
        run_scan(ci, sc, &mut stats);
        let tables = build_tables(&stats);
        for class in 0..2 {
            for slot in 0..2 {
                if let Some(t) = &tables[class][slot] {
                    write_dht(&mut out, ((class as u8) << 4) | slot as u8, t);
                }
            }
        }
        let mut sos = vec![sc.comps.len() as u8];
        for &k in &sc.comps {
            let slot = k.min(1) as u8;
            let (td, ta) = if sc.ss == 0 { (slot, 0) } else { (0, 0) };
            sos.extend_from_slice(&[ci.comps[k].id, (td << 4) | ta]);
        }
        sos.extend_from_slice(&[sc.ss as u8, sc.se as u8, ((sc.ah << 4) | sc.al) as u8]);
        segment(&mut out, 0xDA, &sos);
        // Pass 2: emit.
        let mut emit = Emit { w: MsbWriter { out: Vec::new(), acc: 0, n: 0 }, tables: &tables };
        run_scan(ci, sc, &mut emit);
        emit.w.finish();
        out.extend_from_slice(&emit.w.out);
    }
    out
}

/// Serialize as a progressive JPEG using the best scan layout.
pub fn write_progressive(ci: &CoefImage) -> Vec<u8> {
    let (_, script) = plan(ci);
    emit(ci, &script)
}

/// Serialize with a layout chosen by [`plan`].
pub fn emit(ci: &CoefImage, script: &[Scan]) -> Vec<u8> {
    let best = encode_script(ci, script);
    let mut out = write_headers(ci, 0xC2, best.len() + 1024);
    out.extend_from_slice(&best);
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}
