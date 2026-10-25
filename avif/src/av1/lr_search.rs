//! Encoder side of loop restoration: choosing the filter of every restoration unit.
//!
//! For each unit the candidates are "no filter", a separable symmetric Wiener filter whose taps come from a
//! least-squares fit against the source, and the 16 self-guided parameter sets with least-squares projection
//! weights. Every candidate is scored with the decoder's own filter code, so the reported error is exact. Units
//! are then switched on only where the error saved pays for the parameters.

use super::frame::{Frame, Plane};
use super::lr::{self, Block, LrPlane, LrType, LrUnit, PlaneCtx, SGR_PARAMS, Window, box_filter};
use super::obu::{RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER};
use rayon::prelude::*;

/// The outcome for one plane at one unit size.
#[derive(Clone, Debug)]
pub struct PlaneChoice {
    pub unit_size: usize,
    /// Frame level restoration type (`RESTORE_*`).
    pub kind: u8,
    pub units_per_row: usize,
    pub units_per_col: usize,
    pub units: Vec<LrUnit>,
    /// Cost (error + lambda * bits) of coding the plane this way.
    pub cost: f64,
    /// Cost with no restoration at all.
    pub cost_none: f64,
}

/// Restoration chosen for a whole frame.
#[derive(Clone, Debug)]
pub struct FrameChoice {
    pub planes: Vec<PlaneChoice>,
}

impl FrameChoice {
    pub fn uses_lr(&self) -> bool {
        self.planes.iter().any(|p| p.kind != RESTORE_NONE)
    }
}

type UnitBlocks = Vec<(Block, Window)>;
type Flts = (Option<Vec<i32>>, Option<Vec<i32>>);
type UnitEval = (f64, Option<([[i8; 3]; 2], f64)>, Option<(u8, [i8; 2], f64)>);

/// The blocks (one per stripe and unit column) of each unit, with their source windows.
fn gather_units(ctx: &PlaneCtx, unit_size: usize, sub_y: usize) -> (LrPlane, Vec<UnitBlocks>) {
    let shape = LrPlane::new(unit_size, ctx.plane_w, ctx.plane_h);
    let mut units: Vec<UnitBlocks> = (0..shape.units.len()).map(|_| Vec::new()).collect();
    let stripe_h = 64 >> sub_y;
    let mut stripe = 0usize;
    loop {
        let start = ((64 * stripe) as isize - 8) >> sub_y;
        let end = start + stripe_h as isize - 1;
        if start >= ctx.plane_h as isize {
            break;
        }
        let y0 = start.max(0) as usize;
        let y1 = end.min(ctx.plane_h as isize - 1) as usize;
        let unit_row = ((stripe * stripe_h) / unit_size).min(shape.units_per_col - 1);
        for unit_col in 0..shape.units_per_row {
            let x0 = unit_col * unit_size;
            let x1 = if unit_col + 1 == shape.units_per_row { ctx.plane_w } else { (unit_col + 1) * unit_size };
            if x0 >= x1 {
                continue;
            }
            let block = Block { x0, y0, w: x1 - x0, h: y1 - y0 + 1, stripe_start: start, stripe_end: end };
            let window = Window::gather(ctx, &block);
            units[unit_row * shape.units_per_row + unit_col].push((block, window));
        }
        stripe += 1;
    }
    (shape, units)
}

fn sse_of(blocks: &UnitBlocks, src: &Plane, out: impl Fn(usize, usize, usize) -> i32) -> f64 {
    let mut sse = 0f64;
    for (bi, (b, _)) in blocks.iter().enumerate() {
        let mut acc = 0i64;
        for r in 0..b.h {
            let s = &src.row(b.y0 + r)[b.x0..b.x0 + b.w];
            for (j, &sv) in s.iter().enumerate() {
                let d = sv as i64 - out(bi, r, j) as i64;
                acc += d * d;
            }
        }
        sse += acc as f64;
    }
    sse
}

/// Solve the `n x n` system `a x = b` (n <= 3) by Gaussian elimination, None when singular.
fn solve(n: usize, a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let mut m = [[0f64; 4]; 3];
    for i in 0..n {
        m[i][..n].copy_from_slice(&a[i][..n]);
        m[i][n] = b[i];
    }
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if m[p][c].abs() < 1e-9 {
            return None;
        }
        m.swap(c, p);
        for r in 0..n {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..=n {
                    m[r][k] -= f * m[c][k];
                }
            }
        }
    }
    let mut x = [0f64; 3];
    for i in 0..n {
        x[i] = m[i][n] / m[i][i];
    }
    Some(x)
}

const W_MIN: [i32; 3] = [-5, -23, -17];
const W_MAX: [i32; 3] = [10, 8, 46];

/// Least-squares Wiener taps (vertical pass first in the result, as the decoder's `LrWiener`).
fn wiener_fit(blocks: &UnitBlocks, src: &Plane, chroma: bool) -> [[i8; 3]; 2] {
    let first = chroma as usize;
    let n = 3 - first;
    // Horizontal taps from the pass-through vertical filter.
    let (mut a, mut b) = ([[0f64; 3]; 3], [0f64; 3]);
    for (blk, win) in blocks {
        for i in 0..blk.h {
            let srow = &src.row(blk.y0 + i)[blk.x0..blk.x0 + blk.w];
            for j in 0..blk.w {
                let x = win.at(j as isize, i as isize) as f64;
                let t = srow[j] as f64 - x;
                let mut r = [0f64; 3];
                for k in first..3 {
                    let d = (3 - k) as isize;
                    r[k] =
                        win.at(j as isize - d, i as isize) as f64 + win.at(j as isize + d, i as isize) as f64 - 2.0 * x;
                }
                for k in first..3 {
                    b[k] += r[k] * t;
                    for l in first..3 {
                        a[k][l] += r[k] * r[l];
                    }
                }
            }
        }
    }
    let quant = |sol: Option<[f64; 3]>| -> [i32; 3] {
        let mut c = [0i32; 3];
        if let Some(s) = sol {
            for k in 0..n {
                c[first + k] = ((s[k] * 128.0).round() as i32).clamp(W_MIN[first + k], W_MAX[first + k]);
            }
        }
        c
    };
    let solve_sub = |a: &[[f64; 3]; 3], b: &[f64; 3]| -> Option<[f64; 3]> {
        let mut sa = [[0f64; 3]; 3];
        let mut sb = [0f64; 3];
        for k in 0..n {
            sb[k] = b[first + k];
            for l in 0..n {
                sa[k][l] = a[first + k][first + l] + if k == l { 1e-3 } else { 0.0 };
            }
        }
        solve(n, &sa, &sb)
    };
    let h = quant(solve_sub(&a, &b));
    let htaps = [h[0] as f64 / 128.0, h[1] as f64 / 128.0, h[2] as f64 / 128.0];

    // Vertical taps on the horizontally filtered rows (three rows of margin above and below).
    let (mut a, mut b) = ([[0f64; 3]; 3], [0f64; 3]);
    for (blk, win) in blocks {
        let (w, hh) = (blk.w, blk.h);
        let rows = hh + 6;
        let mut y = vec![0f64; rows * w];
        for ri in 0..rows {
            let i = ri as isize - 3;
            for j in 0..w {
                let x = win.at(j as isize, i) as f64;
                let mut v = x;
                for k in 0..3 {
                    let d = (3 - k) as isize;
                    v += htaps[k] * (win.at(j as isize - d, i) as f64 + win.at(j as isize + d, i) as f64 - 2.0 * x);
                }
                y[ri * w + j] = v;
            }
        }
        for i in 0..hh {
            let srow = &src.row(blk.y0 + i)[blk.x0..blk.x0 + w];
            for j in 0..w {
                let c = y[(i + 3) * w + j];
                let t = srow[j] as f64 - c;
                let mut r = [0f64; 3];
                for k in first..3 {
                    let d = 3 - k;
                    r[k] = y[(i + 3 - d) * w + j] + y[(i + 3 + d) * w + j] - 2.0 * c;
                }
                for k in first..3 {
                    b[k] += r[k] * t;
                    for l in first..3 {
                        a[k][l] += r[k] * r[l];
                    }
                }
            }
        }
    }
    let v = quant(solve_sub(&a, &b));
    [[v[0] as i8, v[1] as i8, v[2] as i8], [h[0] as i8, h[1] as i8, h[2] as i8]]
}

/// Best self-guided parameters of a unit over the given parameter sets: (set, weights, error).
fn sgr_best(blocks: &UnitBlocks, src: &Plane, bit_depth: u32, sets: &[usize]) -> Option<(u8, [i8; 2], f64)> {
    let max = (1i32 << bit_depth) - 1;
    let mut best: Option<(u8, [i8; 2], f64)> = None;
    for &set in sets {
        let p = &SGR_PARAMS[set];
        let (r0, e0, r1, e1) = (p[0] as usize, p[1] as u32, p[2] as usize, p[3] as u32);
        let flts: Vec<Flts> = blocks
            .iter()
            .map(|(b, win)| {
                (
                    (r0 != 0).then(|| box_filter(win, bit_depth, r0, e0, b.w, b.h)),
                    (r1 != 0).then(|| box_filter(win, bit_depth, r1, e1, b.w, b.h)),
                )
            })
            .collect();
        let (mut saa, mut sab, mut sbb, mut sat, mut sbt) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for ((b, win), (f0, f1)) in blocks.iter().zip(&flts) {
            for i in 0..b.h {
                let srow = &src.row(b.y0 + i)[b.x0..b.x0 + b.w];
                for j in 0..b.w {
                    let u = (win.at(j as isize, i as isize) as i32) << 4;
                    let t = (((srow[j] as i32) << 4) - u) as f64;
                    let a = f0.as_ref().map_or(0, |f| f[i * b.w + j] - u) as f64;
                    let c = f1.as_ref().map_or(0, |f| f[i * b.w + j] - u) as f64;
                    saa += a * a;
                    sab += a * c;
                    sbb += c * c;
                    sat += a * t;
                    sbt += c * t;
                }
            }
        }
        let (w0, w1) = if r0 != 0 && r1 != 0 {
            let det = saa * sbb - sab * sab;
            if det.abs() < 1e-6 {
                continue;
            }
            let alpha = (sat * sbb - sbt * sab) / det;
            let beta = (saa * sbt - sab * sat) / det;
            let w0 = ((alpha * 128.0).round() as i32).clamp(-96, 31);
            let w2 = (beta * 128.0).round() as i32;
            (w0, (128 - w0 - w2).clamp(-32, 95))
        } else if r1 != 0 {
            if sbb < 1e-6 {
                continue;
            }
            let w2 = ((sbt / sbb * 128.0).round() as i32).clamp(-63, 160);
            (0, (128 - w2).clamp(-32, 95))
        } else {
            if saa < 1e-6 {
                continue;
            }
            let w0 = ((sat / saa * 128.0).round() as i32).clamp(-96, 31);
            (w0, (128 - w0).clamp(-32, 95))
        };
        let w2 = 128 - w0 - w1;
        let sse = sse_of(blocks, src, |bi, i, j| {
            let (b, win) = &blocks[bi];
            let u = (win.at(j as isize, i as isize) as i32) << 4;
            let (f0, f1) = &flts[bi];
            let a = f0.as_ref().map_or(u, |f| f[i * b.w + j]);
            let c = f1.as_ref().map_or(u, |f| f[i * b.w + j]);
            let v = w1 * u + w0 * a + w2 * c;
            ((v + (1 << 10)) >> 11).clamp(0, max)
        });
        if best.is_none_or(|bb| sse < bb.2) {
            best = Some((set as u8, [w0 as i8, w1 as i8], sse));
        }
    }
    best
}

/// Estimated bits of a unit's parameters (flag symbols are added by the caller).
fn wiener_bits(chroma: bool) -> f64 {
    if chroma { 22.0 } else { 32.0 }
}

fn sgr_bits(set: u8) -> f64 {
    let p = &SGR_PARAMS[set as usize];
    if p[0] != 0 && p[2] != 0 { 4.0 + 14.0 } else { 4.0 + 7.0 }
}

/// Search one plane at one unit size.
#[allow(clippy::too_many_arguments)]
pub fn search_plane(
    cdef: &Plane,
    deblocked: &Plane,
    src: &Plane,
    plane_w: usize,
    plane_h: usize,
    bit_depth: u32,
    sub_y: usize,
    chroma: bool,
    unit_size: usize,
    lambda: f64,
    effort: u8,
) -> PlaneChoice {
    let ctx = PlaneCtx { cdef, deblocked, plane_w, plane_h, bit_depth };
    let (shape, units) = gather_units(&ctx, unit_size, sub_y);
    let all_sets: Vec<usize> = (0..16).collect();
    let few_sets: Vec<usize> = vec![0, 3, 6, 10, 14];
    let sets = if effort >= 8 { &all_sets } else { &few_sets };
    // (error without filter, wiener (taps, error), sgr (set, weights, error)) per unit.
    let evals: Vec<UnitEval> = units
        .par_iter()
        .map(|blocks| {
            let none = sse_of(blocks, src, |bi, i, j| blocks[bi].1.at(j as isize, i as isize) as i32);
            let taps = wiener_fit(blocks, src, chroma);
            let max = (1i32 << bit_depth) - 1;
            let mut outs: Vec<Vec<u16>> = Vec::with_capacity(blocks.len());
            for (b, win) in blocks {
                let mut o = vec![0u16; b.w * b.h];
                lr::wiener(win, bit_depth, &taps, &mut o);
                outs.push(o);
            }
            let _ = max;
            let w_sse = sse_of(blocks, src, |bi, i, j| outs[bi][i * blocks[bi].0.w + j] as i32);
            let sgr = sgr_best(blocks, src, bit_depth, sets);
            (none, Some((taps, w_sse)), sgr)
        })
        .collect();

    // Frame level type: score each option from the per-unit best choices.
    let n = evals.len() as f64;
    let cost_none: f64 = evals.iter().map(|e| e.0).sum();
    let wb = wiener_bits(chroma);
    let mut best_kind = RESTORE_NONE;
    let mut best_cost = cost_none;
    let mut best_units = vec![LrUnit::default(); evals.len()];
    for kind in [RESTORE_WIENER, RESTORE_SGRPROJ, RESTORE_SWITCHABLE] {
        let flag_bits = if kind == RESTORE_SWITCHABLE { 1.6 } else { 1.0 };
        let mut cost = 0f64;
        let mut chosen = Vec::with_capacity(evals.len());
        for (none, w, s) in &evals {
            let mut best = (*none + lambda * flag_bits, LrUnit::default());
            if kind != RESTORE_SGRPROJ
                && let Some((taps, sse)) = w
            {
                let c = sse + lambda * (flag_bits + wb);
                if c < best.0 {
                    best = (c, LrUnit { kind: LrType::Wiener, wiener: *taps, ..LrUnit::default() });
                }
            }
            if kind != RESTORE_WIENER
                && let Some((set, xqd, sse)) = s
            {
                let c = sse + lambda * (flag_bits + sgr_bits(*set));
                if c < best.0 {
                    best = (c, LrUnit { kind: LrType::Sgrproj, sgr_set: *set, sgr_xqd: *xqd, ..LrUnit::default() });
                }
            }
            cost += best.0;
            chosen.push(best.1);
        }
        if cost < best_cost {
            best_cost = cost;
            best_kind = kind;
            best_units = chosen;
        }
    }
    let _ = n;
    PlaneChoice {
        unit_size,
        kind: best_kind,
        units_per_row: shape.units_per_row,
        units_per_col: shape.units_per_col,
        units: best_units,
        cost: best_cost,
        cost_none,
    }
}

/// Search a whole frame: `cdef` is the post-CDEF picture, `deblocked` the picture before CDEF, `src` the
/// original. Picks the unit sizes too.
pub fn search_frame(cdef: &Frame, deblocked: &Frame, src: &Frame, lambda: f64, effort: u8) -> FrameChoice {
    let np = cdef.planes.len();
    let bd = cdef.bit_depth as u32;
    let dims = |p: usize| {
        let (sx, sy) = if p == 0 { (0, 0) } else { (cdef.sub_x as usize, cdef.sub_y as usize) };
        ((cdef.width + sx) >> sx, (cdef.height + sy) >> sy, sy)
    };
    let run = |p: usize, unit: usize| {
        let (w, h, sy) = dims(p);
        search_plane(&cdef.planes[p], &deblocked.planes[p], &src.planes[p], w, h, bd, sy, p > 0, unit, lambda, effort)
    };
    // Unit sizes: searched only at the top effort levels; otherwise one size by picture area.
    let area = cdef.width * cdef.height;
    let default_unit = if area <= 352 * 288 {
        64
    } else if area <= 1_600_000 {
        128
    } else {
        256
    };
    let default_sizes = [default_unit];
    let luma_sizes: &[usize] = if effort >= 8 { &[64, 128, 256] } else { &default_sizes };
    let luma: Vec<PlaneChoice> = luma_sizes.iter().map(|&u| run(0, u)).collect();
    let can_shift = cdef.sub_x == 1 && cdef.sub_y == 1;
    let mut best: Option<(f64, usize, Vec<PlaneChoice>)> = None;
    for (li, lc) in luma.iter().enumerate() {
        let l = lc.unit_size;
        let mut chroma_total = 0f64;
        let mut chroma = Vec::new();
        if np > 1 {
            let cands: Vec<usize> = if can_shift && effort >= 8 { vec![l, l / 2] } else { vec![l] };
            let mut pick: Option<(f64, Vec<PlaneChoice>)> = None;
            for &c in &cands {
                let ch: Vec<PlaneChoice> = (1..np).map(|p| run(p, c)).collect();
                let cost: f64 = ch.iter().map(|x| x.cost).sum();
                if pick.as_ref().is_none_or(|b| cost < b.0) {
                    pick = Some((cost, ch));
                }
            }
            let (c, ch) = pick.expect("a chroma unit size");
            chroma_total = c;
            chroma = ch;
        }
        let total = lc.cost + chroma_total;
        if best.as_ref().is_none_or(|b| total < b.0) {
            best = Some((total, li, chroma));
        }
    }
    let (_, li, chroma) = best.expect("a luma unit size");
    let mut planes = vec![luma[li].clone()];
    planes.extend(chroma);
    FrameChoice { planes }
}

#[allow(dead_code)]
fn _kinds() -> [u8; 3] {
    [RESTORE_NONE, RESTORE_WIENER, RESTORE_SGRPROJ]
}
