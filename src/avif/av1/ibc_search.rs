//! Encoder side of intra block copy: finding displacement vectors that copy an earlier part of the picture.
//!
//! Candidates come from a hash of every 8x8 luma window of the source, so an exact duplicate of a block's top-left
//! window anywhere earlier in the picture is found directly. Each candidate must satisfy the specification's
//! `is_mv_valid` (the copied area is inside the tile, already decoded and far enough behind the current
//! superblock); the best few by prediction error are then coded for real and compared with the intra choice.

use super::consts::*;
use super::entropy::Coder;
use super::frame::Plane;
use super::intrabc::{INTRA_FRAME, INTRABC_DELAY_PIXELS, INTRABC_DELAY_SB64, Mv};
use super::tile::{Blk, TileCoder};

const WIN: usize = 8;

/// Index of the 8x8 luma windows of the source picture.
pub struct IbcIndex {
    /// (hash, x, y) of every non-flat window, sorted.
    entries: Vec<(u64, u16, u16)>,
}

impl IbcIndex {
    /// Pictures with more windows than this are not indexed (the search would cost too much memory).
    pub const MAX_WINDOWS: usize = 12_000_000;

    /// Index the windows of `luma` (`w` x `h` visible samples). Returns None for pictures too large to index.
    pub fn build(luma: &Plane, w: usize, h: usize) -> Option<IbcIndex> {
        if w < WIN || h < WIN || w > 65535 || h > 65535 || w * h > Self::MAX_WINDOWS {
            return None;
        }
        const B: u64 = 0x9e37_79b1_85eb_ca87;
        const C: u64 = 0xc2b2_ae3d_27d4_eb4f;
        let pow = |base: u64, n: usize| (0..n).fold(1u64, |a, _| a.wrapping_mul(base));
        let (bw, cw) = (pow(B, WIN - 1), pow(C, WIN - 1));
        let (nx, ny) = (w - WIN + 1, h - WIN + 1);
        // Horizontal rolling hash of each row, then vertical rolling hash of those.
        let mut row_hash = vec![0u64; nx * h];
        for y in 0..h {
            let row = luma.row(y);
            let mut hv = 0u64;
            for &p in &row[..WIN] {
                hv = hv.wrapping_mul(B).wrapping_add(p as u64 + 1);
            }
            row_hash[y * nx] = hv;
            for x in 1..nx {
                hv = hv
                    .wrapping_sub((row[x - 1] as u64 + 1).wrapping_mul(bw))
                    .wrapping_mul(B)
                    .wrapping_add(row[x + WIN - 1] as u64 + 1);
                row_hash[y * nx + x] = hv;
            }
        }
        let mut entries = Vec::new();
        for x in 0..nx {
            let mut hv = 0u64;
            for y in 0..WIN {
                hv = hv.wrapping_mul(C).wrapping_add(row_hash[y * nx + x]);
            }
            for y in 0..ny {
                if y > 0 {
                    hv = hv
                        .wrapping_sub(row_hash[(y - 1) * nx + x].wrapping_mul(cw))
                        .wrapping_mul(C)
                        .wrapping_add(row_hash[(y + WIN - 1) * nx + x]);
                }
                if !is_flat(luma, x, y) {
                    entries.push((hv, x as u16, y as u16));
                }
            }
        }
        entries.sort_unstable();
        Some(IbcIndex { entries })
    }

    /// The hash of the window at (x, y), computed directly (used for the block being coded).
    fn window_hash(luma: &Plane, x: usize, y: usize) -> u64 {
        const B: u64 = 0x9e37_79b1_85eb_ca87;
        const C: u64 = 0xc2b2_ae3d_27d4_eb4f;
        let mut hv = 0u64;
        for j in 0..WIN {
            let row = luma.row(y + j);
            let mut r = 0u64;
            for &p in &row[x..x + WIN] {
                r = r.wrapping_mul(B).wrapping_add(p as u64 + 1);
            }
            hv = hv.wrapping_mul(C).wrapping_add(r);
        }
        hv
    }

    /// Positions of windows equal to the one at (x, y) of `luma`, excluding the window itself.
    pub fn matches(&self, luma: &Plane, x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let key = Self::window_hash(luma, x, y);
        let start = self.entries.partition_point(|e| e.0 < key);
        self.entries[start..]
            .iter()
            .take_while(move |e| e.0 == key)
            .take(64)
            .map(|&(_, rx, ry)| (rx as usize, ry as usize))
            .filter(move |&p| p != (x, y))
    }
}

fn is_flat(luma: &Plane, x: usize, y: usize) -> bool {
    let first = luma.at(x, y);
    (0..WIN).all(|j| luma.row(y + j)[x..x + WIN].iter().all(|&p| p == first))
}

/// Whether a picture looks like it has repeated content worth a block copy trial: a good share of its 8x8 windows
/// occur again further along.
pub fn looks_repetitive(index: &IbcIndex) -> bool {
    let e = &index.entries;
    if e.len() < 64 {
        return false;
    }
    let dup = e.windows(2).filter(|p| p[0].0 == p[1].0).count();
    dup * 10 >= e.len()
}

impl<C: Coder> TileCoder<'_, C> {
    /// The specification's `is_mv_valid` for an intra block copy vector `mv` of block `b`.
    pub(crate) fn dv_is_valid(&self, b: &Blk, mv: Mv) -> bool {
        if mv[0].abs() >= 1 << 14 || mv[1].abs() >= 1 << 14 || (mv[0] & 7) != 0 || (mv[1] & 7) != 0 {
            return false;
        }
        let (bw, bh) = (block_width(b.bsize) as i32, block_height(b.bsize) as i32);
        let (dr, dc) = (mv[0] >> 3, mv[1] >> 3);
        let mut src_top = b.mi_row as i32 * 4 + dr;
        let mut src_left = b.mi_col as i32 * 4 + dc;
        let src_bottom = src_top + bh;
        let src_right = src_left + bw;
        if b.has_chroma {
            if bw < 8 && self.f.sub_x != 0 {
                src_left -= 4;
            }
            if bh < 8 && self.f.sub_y != 0 {
                src_top -= 4;
            }
        }
        if src_top < self.mi_row_start as i32 * 4
            || src_left < self.mi_col_start as i32 * 4
            || src_bottom > self.mi_row_end as i32 * 4
            || src_right > self.mi_col_end as i32 * 4
        {
            return false;
        }
        let sb128 = self.f.seq.use_128x128_superblock;
        let sb_h = if sb128 { 128 } else { 64 };
        let active_sb_row = (b.mi_row as i32 * 4) / sb_h;
        let active_sb64_col = (b.mi_col as i32 * 4) >> 6;
        let src_sb_row = (src_bottom - 1) / sb_h;
        let src_sb64_col = (src_right - 1) >> 6;
        let total_sb64_per_row = (((self.mi_col_end - self.mi_col_start) as i32 - 1) >> 4) + 1;
        let active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
        let src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
        if src_sb64 >= active_sb64 - INTRABC_DELAY_SB64 {
            return false;
        }
        let gradient = 1 + INTRABC_DELAY_SB64 + sb128 as i32;
        let wf_offset = gradient * (active_sb_row - src_sb_row);
        if src_sb_row > active_sb_row || src_sb64_col >= active_sb64_col - INTRABC_DELAY_SB64 + wf_offset {
            return false;
        }
        let _ = (INTRABC_DELAY_PIXELS, INTRA_FRAME);
        true
    }

    /// Candidate vectors for block `b`: exact duplicates of its top-left 8x8 window, nearest first, each checked for
    /// validity and ranked by the error of copying the reconstruction.
    pub(crate) fn ibc_candidates(&self, b: &Blk, limit: usize) -> Vec<Mv> {
        let Some(src) = self.enc_src.as_ref() else { return Vec::new() };
        let Some(index) = src.ibc.as_ref() else { return Vec::new() };
        let (bw, bh) = (block_width(b.bsize), block_height(b.bsize));
        if bw < WIN || bh < WIN {
            return Vec::new();
        }
        let (x, y) = (b.mi_col * 4, b.mi_row * 4);
        let luma = &src.planes[0];
        if x + WIN > luma.width || y + WIN > luma.height {
            return Vec::new();
        }
        let mut cands: Vec<(usize, Mv)> = Vec::new();
        for (rx, ry) in index.matches(luma, x, y) {
            let mv = [(ry as i32 - y as i32) * 8, (rx as i32 - x as i32) * 8];
            if !self.dv_is_valid(b, mv) {
                continue;
            }
            // Error of the whole block when copied from the reconstruction (luma).
            let cur = &self.f.planes[0];
            let mut sad = 0usize;
            'rows: for j in 0..bh {
                let s = &luma.row(y + j)[x..x + bw];
                let r = &cur.data[(ry + j) * cur.stride + rx..][..bw];
                for (a, c) in s.iter().zip(r) {
                    sad += (*a as i32 - *c as i32).unsigned_abs() as usize;
                }
                if sad > 64 * bw * bh {
                    break 'rows;
                }
            }
            cands.push((sad, mv));
        }
        cands.sort_by_key(|&(sad, mv)| (sad, mv[0].abs() + mv[1].abs()));
        cands.into_iter().take(limit).map(|(_, mv)| mv).collect()
    }
}
