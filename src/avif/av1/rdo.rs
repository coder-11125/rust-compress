//! Rate-distortion search for the intra encoder: partition tree, intra modes, transform size and type.
//!
//! Costs are `J = distortion (sum of squared errors) + lambda * bits`. Rates come from the live adaptive CDFs
//! (static estimates of the symbols about to be coded), distortion from actually reconstructing the candidate.

use super::cdf_tables::Cdfs;
use super::consts::*;
use super::entropy::{Coder, prob_cost};
use super::tile::{Blk, CoefBlock, StoredCoefs, TileCoder};

/// Result of searching a partition node: its cost and the decisions in decoder traversal order.
#[derive(Clone, Default)]
pub(crate) struct Node {
    pub j: f64,
    pub parts: Vec<u8>,
    pub blks: Vec<Blk>,
}

fn cdf_cost(cdf: &[u16], v: usize) -> u64 {
    let lo = if v > 0 { cdf[v - 1] as u32 } else { 0 };
    prob_cost(cdf[v] as u32 - lo) as u64
}

const MODES: [u8; 13] = [
    DC_PRED,
    V_PRED,
    H_PRED,
    D45_PRED,
    D135_PRED,
    D113_PRED,
    D157_PRED,
    D203_PRED,
    D67_PRED,
    SMOOTH_PRED,
    SMOOTH_V_PRED,
    SMOOTH_H_PRED,
    PAETH_PRED,
];

impl<C: Coder> TileCoder<'_, C> {
    fn lambda(&self) -> f64 {
        self.enc_src.as_ref().map_or(1.0, |e| e.lambda)
    }

    /// Cost of the partition symbol at a node (1/256 bits), from the live CDFs.
    fn partition_bits(&self, r: usize, c: usize, bsize: u8, partition: u8, has_rows: bool, has_cols: bool) -> u64 {
        if bsize < BLOCK_8X8 {
            return 0;
        }
        let bsl = (NUM_4X4_WIDE[bsize as usize] as usize).ilog2() as usize;
        let avail_u = self.mi_row_start < r;
        let avail_l = self.mi_col_start < c;
        let above = avail_u && (NUM_4X4_WIDE[self.f.mi_size[self.mi(r - 1, c)] as usize] as usize).ilog2() < bsl as u32;
        let left = avail_l && (NUM_4X4_HIGH[self.f.mi_size[self.mi(r, c - 1)] as usize] as usize).ilog2() < bsl as u32;
        let ctx = left as usize * 2 + above as usize;
        if has_rows && has_cols {
            let cdf: &[u16] = match bsl {
                1 => &self.cdf.partition_w8[ctx],
                5 => &self.cdf.partition_w128[ctx],
                n => &self.cdf.partition[(n - 2) * 4 + ctx],
            };
            cdf_cost(cdf, partition as usize)
        } else if has_rows || has_cols {
            256
        } else {
            0
        }
    }

    /// Bits of the block-level syntax other than coefficients (1/256 bit).
    fn mode_bits(&self, b: &Blk, skip: bool) -> u64 {
        let mut bits = 0;
        let (r, c) = (b.mi_row, b.mi_col);
        let mut ctx = 0;
        if b.avail_u {
            ctx += self.f.skip[self.mi(r - 1, c)] as usize;
        }
        if b.avail_l {
            ctx += self.f.skip[self.mi(r, c - 1)] as usize;
        }
        bits += cdf_cost(&self.cdf.skip[ctx], skip as usize);
        let above = if b.avail_u { self.f.y_mode[self.mi(r - 1, c)] } else { DC_PRED };
        let left = if b.avail_l { self.f.y_mode[self.mi(r, c - 1)] } else { DC_PRED };
        let (ac, lc) = (INTRA_MODE_CONTEXT[above as usize] as usize, INTRA_MODE_CONTEXT[left as usize] as usize);
        bits += cdf_cost(&self.cdf.kf_y_mode[ac][lc], b.y_mode as usize);
        if b.bsize >= BLOCK_8X8 && is_directional(b.y_mode) {
            bits += cdf_cost(&self.cdf.angle_delta[(b.y_mode - V_PRED) as usize], (b.angle_y + 3) as usize);
        }
        if b.has_chroma {
            let cfl_allowed = block_width(b.bsize).max(block_height(b.bsize)) <= 32;
            let ym = b.y_mode as usize;
            bits += if cfl_allowed {
                cdf_cost(&self.cdf.uv_mode_cfl[ym], b.uv_mode as usize)
            } else {
                cdf_cost(&self.cdf.uv_mode[ym], b.uv_mode as usize)
            };
            if b.uv_mode == UV_CFL_PRED {
                let sign_of = |a: i8| {
                    if a == 0 {
                        0
                    } else if a < 0 {
                        1
                    } else {
                        2
                    }
                };
                let (su, sv) = (sign_of(b.cfl_alpha_u), sign_of(b.cfl_alpha_v));
                let joint = (su * 3 + sv) - 1;
                bits += cdf_cost(&self.cdf.cfl_sign, joint);
                if su != 0 {
                    let ctx = (su - 1) * 3 + sv;
                    bits += cdf_cost(&self.cdf.cfl_alpha[ctx], b.cfl_alpha_u.unsigned_abs() as usize - 1);
                }
                if sv != 0 {
                    let ctx = (sv - 1) * 3 + su;
                    bits += cdf_cost(&self.cdf.cfl_alpha[ctx], b.cfl_alpha_v.unsigned_abs() as usize - 1);
                }
            } else if b.bsize >= BLOCK_8X8 && is_directional(b.uv_mode) {
                bits += cdf_cost(&self.cdf.angle_delta[(b.uv_mode - V_PRED) as usize], (b.angle_uv + 3) as usize);
            }
        }
        if self.f.seq.enable_filter_intra
            && b.y_mode == DC_PRED
            && block_width(b.bsize).max(block_height(b.bsize)) <= 32
        {
            bits += cdf_cost(&self.cdf.use_filter_intra[b.bsize as usize], b.use_filter_intra as usize);
            if b.use_filter_intra {
                bits += cdf_cost(&self.cdf.filter_intra_mode, b.filter_intra_mode as usize);
            }
        }
        if self.f.hdr.tx_mode_select && b.bsize > BLOCK_4X4 && !b.lossless {
            let max_rect = MAX_TX_SIZE_RECT[b.bsize as usize];
            let mut depth = 0usize;
            let mut t = max_rect;
            while t != b.tx_size && depth < 2 {
                t = SPLIT_TX_SIZE[t as usize];
                depth += 1;
            }
            let ctx = 0;
            bits += match MAX_TX_DEPTH[b.bsize as usize] {
                1 => cdf_cost(&self.cdf.tx_size_8x8[ctx], depth),
                2 => cdf_cost(&self.cdf.tx_size[0][ctx], depth),
                3 => cdf_cost(&self.cdf.tx_size[1][ctx], depth),
                _ => cdf_cost(&self.cdf.tx_size[2][ctx], depth),
            };
        }
        bits
    }

    /// Run the analysis on selected planes and return (coefficients, distortion, coefficient bits).
    fn analyze_planes(&mut self, b: &mut Blk, planes: u8) -> (Vec<StoredCoefs>, f64, u64) {
        self.an_planes = planes;
        self.an_dist = 0.0;
        self.an_bits = 0;
        let log = self.analyze_block(b);
        self.an_planes = 7;
        (log, self.an_dist, self.an_bits)
    }

    fn sad_of(&self, plane: usize, x: usize, y: usize, w: usize, h: usize) -> u64 {
        let src = &self.enc_src.as_ref().expect("encoder source").planes[plane];
        let cur = &self.f.planes[plane];
        let mut sad = 0u64;
        for i in 0..h {
            let (rs, rc) = (&src.data[(y + i) * src.stride + x..][..w], &cur.data[(y + i) * cur.stride + x..][..w]);
            for j in 0..w {
                sad += (rs[j] as i64 - rc[j] as i64).unsigned_abs();
            }
        }
        sad
    }

    /// Choose and evaluate the best leaf block of the given size. Returns its cost and syntax.
    pub(crate) fn eval_leaf(&mut self, r: usize, c: usize, bsize: u8) -> (f64, Blk) {
        let lambda = self.lambda();
        let mut base = Blk { mi_row: r, mi_col: c, bsize, ..Default::default() };
        self.set_block_geometry(&mut base);
        let max_rect = MAX_TX_SIZE_RECT[bsize as usize];
        base.tx_size = max_rect;
        let (x, y) = (c * 4, r * 4);
        let (tw, th) = (block_width(bsize).min(64), block_height(bsize).min(64));
        let _ = (tw, th);

        // --- Luma mode ranking by prediction error (whole block at once).
        let (txw, txh) = (TX_W[max_rect as usize] as usize, TX_H[max_rect as usize] as usize);
        let mut ranked: Vec<(u64, u8)> = Vec::with_capacity(13);
        for &mode in &MODES {
            let mut t = base.clone();
            t.y_mode = mode;
            self.predict_intra_block(&t, 0, x, y, max_rect, base.avail_l, base.avail_u, false, false, mode);
            ranked.push((self.sad_of(0, x, y, txw, txh), mode));
        }
        ranked.sort();
        let mut cands: Vec<(u8, i8)> = Vec::new();
        let effort = self.enc_src.as_ref().map_or(9, |e| e.effort);
        let n_modes = if effort >= 6 {
            3
        } else if effort >= 3 {
            2
        } else {
            1
        };
        for &(_, mode) in ranked.iter().take(n_modes) {
            if is_directional(mode) && base.bsize >= BLOCK_8X8 {
                // Refine the angle by prediction error.
                let mut best = (u64::MAX, 0i8);
                for d in -3i8..=3 {
                    let mut t = base.clone();
                    t.y_mode = mode;
                    t.angle_y = d;
                    self.predict_intra_block(&t, 0, x, y, max_rect, base.avail_l, base.avail_u, false, false, mode);
                    let sad = self.sad_of(0, x, y, txw, txh);
                    if sad < best.0 {
                        best = (sad, d);
                    }
                }
                cands.push((mode, best.1));
            } else {
                cands.push((mode, 0));
            }
        }
        // --- Full evaluation of the candidates (luma only), keeping the best. Mode decisions use DCT only.
        self.full_tx = effort >= 8;
        let mut best: Option<(f64, Blk)> = None;
        for &(mode, angle) in &cands {
            let mut t = base.clone();
            t.y_mode = mode;
            t.angle_y = angle;
            let (log, dist, coef_bits) = self.analyze_planes(&mut t, 1);
            let skip = log.iter().all(|s| s.eob == 0);
            let bits = coef_bits + self.mode_bits(&t, skip);
            let j = dist + lambda * bits as f64 / 256.0;
            if best.as_ref().is_none_or(|b| j < b.0) {
                best = Some((j, t));
            }
        }
        let (_, mut blk) = best.expect("at least one candidate");

        // --- Transform size for the winner.
        if self.f.hdr.tx_mode_select && bsize > BLOCK_4X4 && !blk.lossless && effort >= 4 {
            let max_depth = if MAX_TX_DEPTH[bsize as usize] == 1 { 1 } else { 2 };
            let mut t = max_rect;
            let mut best_tx = (f64::MAX, max_rect);
            for depth in 0..=max_depth {
                if depth > 0 {
                    t = SPLIT_TX_SIZE[t as usize];
                }
                let mut tb = blk.clone();
                tb.tx_size = t;
                let (log, dist, coef_bits) = self.analyze_planes(&mut tb, 1);
                let skip = log.iter().all(|s| s.eob == 0);
                let bits = coef_bits + self.mode_bits(&tb, skip);
                let j = dist + lambda * bits as f64 / 256.0;
                if j < best_tx.0 {
                    best_tx = (j, t);
                }
                if t == TX_4X4 {
                    break;
                }
            }
            blk.tx_size = best_tx.1;
        }

        // --- Chroma.
        if blk.has_chroma {
            let (sx, sy) = (self.f.sub_x, self.f.sub_y);
            let uv_tx = self.get_tx_size(&blk, 1);
            let (cx, cy) = ((blk.mi_col >> sx) * 4, (blk.mi_row >> sy) * 4);
            let (cw, ch) = (TX_W[uv_tx as usize] as usize, TX_H[uv_tx as usize] as usize);
            let mut ranked: Vec<(u64, u8)> = Vec::new();
            for &mode in &MODES {
                let mut t = blk.clone();
                t.uv_mode = mode;
                self.predict_intra_block(
                    &t,
                    1,
                    cx,
                    cy,
                    uv_tx,
                    blk.avail_l_chroma,
                    blk.avail_u_chroma,
                    false,
                    false,
                    mode,
                );
                ranked.push((self.sad_of(1, cx, cy, cw, ch), mode));
            }
            ranked.sort();
            let mut uv_cands: Vec<(u8, i8)> = ranked.iter().take(2).map(|&(_, m)| (m, 0i8)).collect();
            // Chroma-from-luma needs the reconstructed luma of this block: analyse luma once.
            let cfl_allowed = block_width(bsize).max(block_height(bsize)) <= 32;
            if cfl_allowed {
                uv_cands.push((UV_CFL_PRED, 0));
            }
            let mut best_uv: Option<(f64, Blk)> = None;
            if cfl_allowed {
                let mut t = blk.clone();
                self.analyze_planes(&mut t, 1);
            }
            for &(mode, _) in &uv_cands {
                let mut t = blk.clone();
                t.uv_mode = mode;
                if mode == UV_CFL_PRED {
                    // Fit the scale factors from the prediction error, then evaluate.
                    self.fit_cfl(&mut t);
                    if t.cfl_alpha_u == 0 && t.cfl_alpha_v == 0 {
                        continue;
                    }
                }
                let (log, dist, coef_bits) = self.analyze_planes(&mut t, 6);
                let skip = log.iter().all(|s| s.eob == 0);
                let bits = coef_bits + self.mode_bits(&t, skip);
                let j = dist + lambda * bits as f64 / 256.0;
                if best_uv.as_ref().is_none_or(|b| j < b.0) {
                    best_uv = Some((j, t));
                }
            }
            if let Some((_, t)) = best_uv {
                blk.uv_mode = t.uv_mode;
                blk.angle_uv = t.angle_uv;
                blk.cfl_alpha_u = t.cfl_alpha_u;
                blk.cfl_alpha_v = t.cfl_alpha_v;
            }
        }

        // --- Final cost of the chosen configuration, with the complete transform-type search.
        self.full_tx = true;
        let (log, dist, coef_bits) = self.analyze_planes(&mut blk, 7);
        let skip = log.iter().all(|s| s.eob == 0);
        let bits = if skip {
            self.mode_bits(&blk, true) + self.skip_residual_bits(&blk)
        } else {
            coef_bits + self.mode_bits(&blk, false)
        };
        blk.skip = skip;
        blk.pre = Some(std::sync::Arc::new(log));
        let j = dist + lambda * bits as f64 / 256.0;
        (j, blk)
    }

    /// Bits for the transform blocks of a skipped block: none.
    fn skip_residual_bits(&self, _b: &Blk) -> u64 {
        0
    }

    /// Choose CfL alphas by least squares between the AC part of the subsampled luma and the source chroma.
    fn fit_cfl(&mut self, t: &mut Blk) {
        let (sx, sy) = (self.f.sub_x, self.f.sub_y);
        let uv_tx = self.get_tx_size(t, 1);
        let (w, h) = (TX_W[uv_tx as usize] as usize, TX_H[uv_tx as usize] as usize);
        let (cx, cy) = ((t.mi_col >> sx) * 4, (t.mi_row >> sy) * 4);
        // Luma AC contribution as the decoder would form it (before scaling).
        let mut luma = vec![0i32; w * h];
        let mut sum = 0i64;
        {
            let lp = &self.f.planes[0];
            for i in 0..h {
                for j in 0..w {
                    let (ly, lx) = ((cy + i) << sy, (cx + j) << sx);
                    let mut s = 0i32;
                    for dy in 0..=sy {
                        for dx in 0..=sx {
                            s += lp.at(lx + dx, ly + dy) as i32;
                        }
                    }
                    let v = s << (3 - sx - sy);
                    luma[i * w + j] = v;
                    sum += v as i64;
                }
            }
        }
        let avg = (sum + (w * h) as i64 / 2) / (w * h) as i64;
        for v in luma.iter_mut() {
            *v -= avg as i32;
        }
        t.uv_mode = DC_PRED;
        let mut alphas = [0i8; 2];
        for (pi, plane) in [1usize, 2].iter().enumerate() {
            // DC prediction first (the decoder starts from it).
            self.predict_intra_block(
                t,
                *plane,
                cx,
                cy,
                uv_tx,
                t.avail_l_chroma,
                t.avail_u_chroma,
                false,
                false,
                DC_PRED,
            );
            let src = &self.enc_src.as_ref().expect("src").planes[*plane];
            let cur = &self.f.planes[*plane];
            let (mut num, mut den) = (0i64, 0i64);
            for i in 0..h {
                for j in 0..w {
                    let e = src.at(cx + j, cy + i) as i64 - cur.at(cx + j, cy + i) as i64;
                    let l = luma[i * w + j] as i64;
                    num += e * l;
                    den += l * l;
                }
            }
            // alpha is in 1/8 steps: scaled = Round2(alpha * ac, 6) with ac carrying a factor 8.
            let a = if den == 0 { 0.0 } else { num as f64 / den as f64 * 64.0 };
            alphas[pi] = a.round().clamp(-16.0, 16.0) as i8;
        }
        t.uv_mode = UV_CFL_PRED;
        t.cfl_alpha_u = alphas[0];
        t.cfl_alpha_v = alphas[1];
    }

    /// Search the partition tree rooted at a node.
    pub(crate) fn search_node(&mut self, r: usize, c: usize, bsize: u8) -> Node {
        if r >= self.f.mi_rows || c >= self.f.mi_cols {
            return Node::default();
        }
        let lambda = self.lambda();
        let num4 = NUM_4X4_WIDE[bsize as usize] as usize;
        let half = num4 >> 1;
        let has_rows = (r + half) < self.f.mi_rows;
        let has_cols = (c + half) < self.f.mi_cols;
        if bsize < BLOCK_8X8 {
            let (j, b) = self.eval_leaf(r, c, bsize);
            return Node { j, parts: Vec::new(), blks: vec![b] };
        }
        let mut options: Vec<u8> = Vec::new();
        if has_rows && has_cols {
            options.extend([PARTITION_NONE, PARTITION_SPLIT, PARTITION_HORZ, PARTITION_VERT]);
            if bsize == BLOCK_8X8 {
                // 8x4 / 4x8 halves are allowed, but are rarely worth the extra evaluation.
            }
        } else if has_cols {
            options.extend([PARTITION_HORZ, PARTITION_SPLIT]);
        } else if has_rows {
            options.extend([PARTITION_VERT, PARTITION_SPLIT]);
        } else {
            options.push(PARTITION_SPLIT);
        }
        let effort = self.enc_src.as_ref().map_or(9, |e| e.effort);
        let (mut none_j, mut split_j): (Option<f64>, Option<f64>) = (None, None);
        let mut best: Option<Node> = None;
        let mut last: Option<Node> = None;
        for &p in &options {
            if effort < 8 && (p == PARTITION_HORZ || p == PARTITION_VERT) {
                // Rectangular shapes only pay off when the square options are close; otherwise skip the work.
                let (jn, js) = (none_j.unwrap_or(f64::MAX), split_j.unwrap_or(f64::MAX));
                if (jn - js).abs() > 0.15 * jn.min(js) {
                    continue;
                }
            }
            let pbits = self.partition_bits(r, c, bsize, p, has_rows, has_cols);
            let mut node = Node { j: lambda * pbits as f64 / 256.0, parts: vec![p], blks: Vec::new() };
            let sub = partition_subsize(p, bsize);
            match p {
                PARTITION_NONE => {
                    let (j, b) = self.eval_leaf(r, c, sub);
                    node.j += j;
                    node.blks.push(b);
                }
                PARTITION_HORZ => {
                    let (j, b) = self.eval_leaf(r, c, sub);
                    node.j += j;
                    node.blks.push(b);
                    if has_rows {
                        let (j2, b2) = self.eval_leaf(r + half, c, sub);
                        node.j += j2;
                        node.blks.push(b2);
                    }
                }
                PARTITION_VERT => {
                    let (j, b) = self.eval_leaf(r, c, sub);
                    node.j += j;
                    node.blks.push(b);
                    if has_cols {
                        let (j2, b2) = self.eval_leaf(r, c + half, sub);
                        node.j += j2;
                        node.blks.push(b2);
                    }
                }
                _ => {
                    // Split: four children, each re-applied before the next is evaluated.
                    for (dr, dc) in [(0, 0), (0, half), (half, 0), (half, half)] {
                        let child = self.search_node(r + dr, c + dc, sub);
                        node.j += child.j;
                        node.parts.extend(child.parts.iter().copied());
                        node.blks.extend(child.blks.iter().cloned());
                    }
                }
            }
            if p == PARTITION_NONE {
                none_j = Some(node.j);
            } else if p == PARTITION_SPLIT {
                split_j = Some(node.j);
            }
            let skipped_whole = p == PARTITION_NONE && effort < 8 && bsize >= BLOCK_16X16 && node.blks[0].skip;
            if best.as_ref().is_none_or(|b| node.j < b.j) {
                best = Some(node.clone());
            }
            last = Some(node);
            if skipped_whole {
                // A flat block that codes no residual: finer partitions cannot beat it by enough to matter.
                break;
            }
        }
        let best = best.expect("a partition option");
        // Leave the frame buffer in the state of the winning option.
        let wins_last = last.as_ref().is_some_and(|l| l.parts == best.parts && l.j == best.j);
        if !wins_last {
            self.apply_blocks(&best.blks);
        }
        best
    }

    /// Reconstruct planned blocks into the frame buffer (so later predictions see them).
    pub(crate) fn apply_blocks(&mut self, blks: &[Blk]) {
        let full = std::mem::replace(&mut self.full_tx, true);
        for b in blks {
            let mut t = b.clone();
            self.set_block_geometry(&mut t);
            self.analyze_planes(&mut t, 7);
        }
        self.full_tx = full;
    }

    /// Search one superblock and queue its decisions for the traversal.
    pub(crate) fn plan_sb(&mut self, r: usize, c: usize) {
        let bs = self.sb_size();
        let node = self.search_node(r, c, bs);
        self.part_cache = node.parts.into();
        self.blk_cache = node.blks.into();
    }
}

/// Unused helper kept for symmetry with the decoder-side context code.
#[allow(dead_code)]
fn unused(_: &Cdfs, _: &CoefBlock) {}
