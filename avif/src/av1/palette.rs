//! Palette mode for the encoder: candidate palettes for a block (exact when the block has few colours, clustered
//! otherwise), the index map, and the rate of the palette syntax. The syntax itself (colour lists, cache use,
//! colour map) is the shared code in `tile.rs`.

use super::consts::*;
use super::entropy::{Coder, prob_cost};
use super::tile::{Blk, TileCoder, palette_color_context, plan_palette_colors, plan_v_deltas};
use std::sync::Arc;

/// Colour clusters of at most this many distinct values are tried; busier blocks (photographs) are not palette material.
const MAX_DISTINCT: usize = 64;

/// A candidate for one plane pair: sorted colours (U with the matching V for chroma) and the index map.
pub(crate) struct PalCand {
    pub colors: Vec<u16>,
    pub colors_v: Vec<u16>,
    pub map: Arc<Vec<u8>>,
}

fn cdf_cost(cdf: &[u16], v: usize) -> u64 {
    let lo = if v > 0 { cdf[v - 1] as u32 } else { 0 };
    prob_cost(cdf[v] as u32 - lo) as u64
}

/// Weighted 1-D k-means over sorted distinct values; centres are returned sorted and distinct.
fn cluster_1d(vals: &[(u16, u32)], k: usize) -> Vec<u16> {
    let total: u32 = vals.iter().map(|v| v.1).sum();
    let mut centers: Vec<f64> = Vec::with_capacity(k);
    let mut acc = 0u32;
    let mut next = 0;
    for &(v, w) in vals {
        acc += w;
        while next < k && acc as u64 * k as u64 >= (2 * next as u64 + 1) * total as u64 / 2 {
            centers.push(v as f64);
            next += 1;
        }
    }
    centers.dedup();
    for _ in 0..8 {
        let (mut sum, mut cnt) = (vec![0f64; centers.len()], vec![0f64; centers.len()]);
        for &(v, w) in vals {
            let i = nearest_1d(&centers, v as f64);
            sum[i] += v as f64 * w as f64;
            cnt[i] += w as f64;
        }
        for i in 0..centers.len() {
            if cnt[i] > 0.0 {
                centers[i] = sum[i] / cnt[i];
            }
        }
    }
    let mut out: Vec<u16> = centers.iter().map(|c| c.round() as u16).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn nearest_1d(centers: &[f64], v: f64) -> usize {
    (0..centers.len()).min_by(|&a, &b| (centers[a] - v).abs().total_cmp(&(centers[b] - v).abs())).unwrap_or(0)
}

/// Weighted 2-D k-means over distinct (u, v) pairs, seeded with the most frequent pairs.
fn cluster_2d(vals: &[((u16, u16), u32)], k: usize) -> Vec<(u16, u16)> {
    let mut by_count: Vec<_> = vals.to_vec();
    by_count.sort_by_key(|v| std::cmp::Reverse(v.1));
    let mut centers: Vec<(f64, f64)> = by_count.iter().take(k).map(|v| (v.0.0 as f64, v.0.1 as f64)).collect();
    let dist = |c: &(f64, f64), p: (u16, u16)| (c.0 - p.0 as f64).powi(2) + (c.1 - p.1 as f64).powi(2);
    let nearest = |centers: &[(f64, f64)], p: (u16, u16)| {
        (0..centers.len()).min_by(|&a, &b| dist(&centers[a], p).total_cmp(&dist(&centers[b], p))).unwrap_or(0)
    };
    for _ in 0..8 {
        let mut acc = vec![(0f64, 0f64, 0f64); centers.len()];
        for &(p, w) in vals {
            let i = nearest(&centers, p);
            acc[i].0 += p.0 as f64 * w as f64;
            acc[i].1 += p.1 as f64 * w as f64;
            acc[i].2 += w as f64;
        }
        for (c, a) in centers.iter_mut().zip(&acc) {
            if a.2 > 0.0 {
                *c = (a.0 / a.2, a.1 / a.2);
            }
        }
    }
    let mut out: Vec<(u16, u16)> = centers.iter().map(|c| (c.0.round() as u16, c.1.round() as u16)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Run-length distinct values of a sample list, or `None` when there are more than `MAX_DISTINCT`.
fn distinct<T: Ord + Copy>(mut v: Vec<T>) -> Option<Vec<(T, u32)>> {
    v.sort_unstable();
    let mut out: Vec<(T, u32)> = Vec::new();
    for x in v {
        match out.last_mut() {
            Some(l) if l.0 == x => l.1 += 1,
            _ => {
                if out.len() == MAX_DISTINCT {
                    return None;
                }
                out.push((x, 1));
            }
        }
    }
    Some(out)
}

/// Extend an `on_w` x `on_h` index map to the whole `bw` x `bh` block the way the decoder does, at stride 64.
fn extend_map(small: &[u8], on_w: usize, on_h: usize, bw: usize, bh: usize) -> Vec<u8> {
    let mut map = vec![0u8; 64 * 64];
    for i in 0..on_h {
        for j in 0..bw {
            map[i * 64 + j] = small[i * on_w + j.min(on_w - 1)];
        }
    }
    for i in on_h..bh {
        let (head, tail) = map.split_at_mut(i * 64);
        tail[..bw].copy_from_slice(&head[(on_h - 1) * 64..(on_h - 1) * 64 + bw]);
    }
    map
}

impl<C: Coder> TileCoder<'_, C> {
    /// Palette syntax exists for this block size when screen content tools are on.
    pub(crate) fn palette_allowed(&self, bsize: u8) -> bool {
        self.f.hdr.allow_screen_content_tools
            && bsize >= BLOCK_8X8
            && block_width(bsize) <= 64
            && block_height(bsize) <= 64
    }

    /// Visible part of the block in luma samples: (x, y, width, height).
    fn visible(&self, b: &Blk) -> (usize, usize, usize, usize) {
        let (bw, bh) = (block_width(b.bsize), block_height(b.bsize));
        let on_w = bw.min((self.f.mi_cols - b.mi_col) * 4);
        let on_h = bh.min((self.f.mi_rows - b.mi_row) * 4);
        (b.mi_col * 4, b.mi_row * 4, on_w, on_h)
    }

    /// Luma palettes worth trying for the block: its exact colours when there are 2..=8, else clusters.
    pub(crate) fn luma_palette_candidates(&self, b: &Blk) -> Vec<PalCand> {
        let Some(src) = self.enc_src.as_ref() else { return Vec::new() };
        let effort = src.effort;
        let (x0, y0, on_w, on_h) = self.visible(b);
        let plane = &src.planes[0];
        let mut pix = Vec::with_capacity(on_w * on_h);
        for i in 0..on_h {
            pix.extend_from_slice(&plane.data[(y0 + i) * plane.stride + x0..][..on_w]);
        }
        let Some(vals) = distinct(pix.clone()) else { return Vec::new() };
        if vals.len() < 2 {
            return Vec::new();
        }
        let mut sets: Vec<Vec<u16>> = Vec::new();
        if vals.len() <= 8 {
            sets.push(vals.iter().map(|v| v.0).collect());
        } else {
            sets.push(cluster_1d(&vals, 8));
            if effort >= 8 {
                sets.push(cluster_1d(&vals, 4));
            }
        }
        let (bw, bh) = (block_width(b.bsize), block_height(b.bsize));
        sets.into_iter()
            .filter(|c| c.len() >= 2)
            .map(|colors| {
                let centers: Vec<f64> = colors.iter().map(|&c| c as f64).collect();
                let small: Vec<u8> = pix.iter().map(|&p| nearest_1d(&centers, p as f64) as u8).collect();
                PalCand { colors, colors_v: Vec::new(), map: Arc::new(extend_map(&small, on_w, on_h, bw, bh)) }
            })
            .collect()
    }

    /// Chroma palettes (U with its V) for the block.
    pub(crate) fn chroma_palette_candidates(&self, b: &Blk) -> Vec<PalCand> {
        let Some(src) = self.enc_src.as_ref() else { return Vec::new() };
        if !b.has_chroma || src.planes.len() < 3 {
            return Vec::new();
        }
        let (sx, sy) = (self.f.sub_x, self.f.sub_y);
        let (x0, y0, on_w, on_h) = self.visible(b);
        let (bw, bh) = (block_width(b.bsize) >> sx, block_height(b.bsize) >> sy);
        let (cx, cy, cw, ch) = (x0 >> sx, y0 >> sy, on_w >> sx, on_h >> sy);
        if cw == 0 || ch == 0 || bw < 4 || bh < 4 {
            return Vec::new();
        }
        let (pu, pv) = (&src.planes[1], &src.planes[2]);
        let mut pix = Vec::with_capacity(cw * ch);
        for i in 0..ch {
            for j in 0..cw {
                let o = (cy + i) * pu.stride + cx + j;
                pix.push((pu.data[o], pv.data[o]));
            }
        }
        let Some(vals) = distinct(pix.clone()) else { return Vec::new() };
        if vals.len() < 2 {
            return Vec::new();
        }
        let pairs: Vec<(u16, u16)> =
            if vals.len() <= 8 { vals.iter().map(|v| v.0).collect() } else { cluster_2d(&vals, 8) };
        if pairs.len() < 2 {
            return Vec::new();
        }
        let d = |c: &(u16, u16), p: (u16, u16)| (c.0 as i64 - p.0 as i64).pow(2) + (c.1 as i64 - p.1 as i64).pow(2);
        let small: Vec<u8> =
            pix.iter().map(|&p| (0..pairs.len()).min_by_key(|&i| d(&pairs[i], p)).unwrap_or(0) as u8).collect();
        vec![PalCand {
            colors: pairs.iter().map(|p| p.0).collect(),
            colors_v: pairs.iter().map(|p| p.1).collect(),
            map: Arc::new(extend_map(&small, cw, ch, bw, bh)),
        }]
    }

    /// Copy a luma candidate into a block.
    pub(crate) fn set_luma_palette(&self, b: &mut Blk, cand: &PalCand) {
        b.y_mode = DC_PRED;
        b.use_filter_intra = false;
        b.pal_size_y = cand.colors.len() as u8;
        b.pal_y = [0; 8];
        b.pal_y[..cand.colors.len()].copy_from_slice(&cand.colors);
        b.pal_map_y = Some(cand.map.clone());
    }

    /// Copy a chroma candidate into a block, choosing how V is coded.
    pub(crate) fn set_chroma_palette(&self, b: &mut Blk, cand: &PalCand) {
        let bd = self.f.seq.color.bit_depth as u32;
        let n = cand.colors.len();
        b.uv_mode = DC_PRED;
        b.pal_size_uv = n as u8;
        b.pal_u = [0; 8];
        b.pal_v = [0; 8];
        b.pal_u[..n].copy_from_slice(&cand.colors);
        b.pal_v[..n].copy_from_slice(&cand.colors_v);
        b.pal_map_uv = Some(cand.map.clone());
        b.pal_v_delta_encode = plan_v_deltas(&b.pal_v[..n], bd).is_some_and(|(e, d)| {
            bd + 2 + d.iter().map(|&x| bd - 4 + e + (x != 0) as u32).sum::<u32>() < n as u32 * bd
        });
    }

    fn color_cdf(&self, luma: bool, n: usize, ctx: usize) -> &[u16] {
        let c = &self.cdf;
        match (luma, n) {
            (true, 2) => &c.palette_color_y_2[ctx],
            (true, 3) => &c.palette_color_y_3[ctx],
            (true, 4) => &c.palette_color_y_4[ctx],
            (true, 5) => &c.palette_color_y_5[ctx],
            (true, 6) => &c.palette_color_y_6[ctx],
            (true, 7) => &c.palette_color_y_7[ctx],
            (true, _) => &c.palette_color_y_8[ctx],
            (false, 2) => &c.palette_color_uv_2[ctx],
            (false, 3) => &c.palette_color_uv_3[ctx],
            (false, 4) => &c.palette_color_uv_4[ctx],
            (false, 5) => &c.palette_color_uv_5[ctx],
            (false, 6) => &c.palette_color_uv_6[ctx],
            (false, 7) => &c.palette_color_uv_7[ctx],
            (false, _) => &c.palette_color_uv_8[ctx],
        }
    }

    /// Rate (1/256 bit) of coding an index map in the decoder's wavefront order, from the current CDFs.
    fn map_bits(&self, map: &[u8], n: usize, on_w: usize, on_h: usize, luma: bool) -> u64 {
        let mut bits = ((n as f64).log2() * 256.0) as u64;
        for i in 1..(on_h + on_w - 1) {
            let jmax = i.min(on_w - 1);
            let jmin = (i + 1).saturating_sub(on_h);
            for j in (jmin..=jmax).rev() {
                let (row, col) = (i - j, j);
                let (order, ctx) = palette_color_context(map, 64, row, col, n);
                let sym = order.iter().position(|&o| o == map[row * 64 + col]).unwrap_or(0);
                bits += cdf_cost(self.color_cdf(luma, n, ctx), sym);
            }
        }
        bits
    }

    /// Rate (1/256 bit) of everything the palette syntax of a block takes: the has-palette flags, and for palette
    /// planes the sizes, colours and index maps.
    pub(crate) fn palette_bits(&self, b: &Blk) -> u64 {
        if !self.palette_allowed(b.bsize) {
            return 0;
        }
        let bd = self.f.seq.color.bit_depth as u32;
        let ctx_idx = (NUM_4X4_WIDE[b.bsize as usize].ilog2() + NUM_4X4_HIGH[b.bsize as usize].ilog2()) as usize - 2;
        let (r, c) = (b.mi_row, b.mi_col);
        let (_, _, on_w, on_h) = self.visible(b);
        let mut bits = 0;
        if b.y_mode == DC_PRED {
            let ctx = (b.avail_u && self.f.pal_size[0][self.mi(r - 1, c)] > 0) as usize
                + (b.avail_l && self.f.pal_size[0][self.mi(r, c - 1)] > 0) as usize;
            bits += cdf_cost(&self.cdf.palette_y_mode[ctx_idx][ctx], (b.pal_size_y > 0) as usize);
            if b.pal_size_y > 0 {
                let n = b.pal_size_y as usize;
                bits += cdf_cost(&self.cdf.palette_size_y[ctx_idx], n - 2);
                let plan = plan_palette_colors(&b.pal_y[..n], &self.palette_cache(b, 0), bd, true);
                bits += plan.bits as u64 * 256;
                if let Some(m) = &b.pal_map_y {
                    bits += self.map_bits(m, n, on_w, on_h, true);
                }
            }
        }
        if b.has_chroma && b.uv_mode == DC_PRED {
            let ctx = (b.pal_size_y > 0) as usize;
            bits += cdf_cost(&self.cdf.palette_uv_mode[ctx], (b.pal_size_uv > 0) as usize);
            if b.pal_size_uv > 0 {
                let n = b.pal_size_uv as usize;
                bits += cdf_cost(&self.cdf.palette_size_uv[ctx_idx], n - 2);
                let plan = plan_palette_colors(&b.pal_u[..n], &self.palette_cache(b, 1), bd, false);
                bits += plan.bits as u64 * 256;
                let raw = n as u32 * bd;
                let v_bits = match plan_v_deltas(&b.pal_v[..n], bd) {
                    Some((e, d)) if b.pal_v_delta_encode => {
                        bd + 2 + d.iter().map(|&x| bd - 4 + e + (x != 0) as u32).sum::<u32>()
                    }
                    _ => raw,
                };
                bits += (v_bits as u64 + 1) * 256;
                if let Some(m) = &b.pal_map_uv {
                    let (sx, sy) = (self.f.sub_x, self.f.sub_y);
                    bits += self.map_bits(m, n, on_w >> sx, on_h >> sy, false);
                }
            }
        }
        bits
    }
}
