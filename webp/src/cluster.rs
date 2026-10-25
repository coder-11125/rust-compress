//! Grouping of image tiles into a few sets that share prefix codes (the "meta prefix" image).
//!
//! Every tile of the image gets a histogram of the symbols coded inside it. Tiles whose statistics
//! are alike should share one set of five prefix codes, since each distinct set costs a header but
//! each shared set costs coding efficiency. Clustering is greedy and entropy-driven: a tile joins
//! the cluster it makes cheapest, or starts a new one when joining would cost more than the new
//! cluster's own header; a couple of reassignment passes then polish the result.

/// The five alphabets coded per pixel: green/length/cache, red, blue, alpha, distance.
pub const ALPHABETS: usize = 5;

/// Sparse symbol counts of one tile: (alphabet, symbol, count), sorted.
pub type Sparse = Vec<(u8, u16, u32)>;

/// Dense running counts of one cluster.
struct Cluster {
    counts: [Vec<u32>; ALPHABETS],
    /// Total symbols per alphabet and Σ f·log2 f per alphabet.
    total: [f64; ALPHABETS],
    flogf: [f64; ALPHABETS],
    used: usize,
}

#[inline(always)]
fn flog(f: f64) -> f64 {
    if f > 0.0 { f * f.log2() } else { 0.0 }
}

/// Approximate header bits for one newly used symbol of a prefix code description.
const SYMBOL_HEADER_BITS: f64 = 3.5;
/// Fixed header bits of one group (five codes' preambles).
const GROUP_HEADER_BITS: f64 = 5.0 * 24.0;

impl Cluster {
    fn new(sizes: &[usize; ALPHABETS]) -> Self {
        Cluster { counts: sizes.map(|n| vec![0; n]), total: [0.0; ALPHABETS], flogf: [0.0; ALPHABETS], used: 0 }
    }

    /// Entropy-coded size of the cluster's symbols plus its code descriptions, in bits.
    fn cost(&self) -> f64 {
        let body: f64 = (0..ALPHABETS).map(|a| flog(self.total[a]) - self.flogf[a]).sum();
        body + GROUP_HEADER_BITS + self.used as f64 * SYMBOL_HEADER_BITS
    }

    /// Change in `cost()` if `tile` were added (`sign` = 1) or removed (`sign` = -1).
    fn delta(&self, tile: &Sparse, sign: i64) -> f64 {
        let mut dtotal = [0f64; ALPHABETS];
        let mut dflogf = [0f64; ALPHABETS];
        let mut dused = 0i64;
        for &(a, s, c) in tile {
            let (a, s) = (a as usize, s as usize);
            let f = self.counts[a][s] as i64;
            let g = f + sign * c as i64;
            dtotal[a] += (sign * c as i64) as f64;
            dflogf[a] += flog(g as f64) - flog(f as f64);
            if f == 0 && g > 0 {
                dused += 1;
            } else if f > 0 && g == 0 {
                dused -= 1;
            }
        }
        let mut d = 0.0;
        for a in 0..ALPHABETS {
            d += flog(self.total[a] + dtotal[a]) - flog(self.total[a]) - dflogf[a];
        }
        d + dused as f64 * SYMBOL_HEADER_BITS
    }

    fn apply(&mut self, tile: &Sparse, sign: i64) {
        for &(a, s, c) in tile {
            let (a, s) = (a as usize, s as usize);
            let f = self.counts[a][s] as i64;
            let g = f + sign * c as i64;
            self.flogf[a] += flog(g as f64) - flog(f as f64);
            self.total[a] += (sign * c as i64) as f64;
            if f == 0 && g > 0 {
                self.used += 1;
            } else if f > 0 && g == 0 {
                self.used -= 1;
            }
            self.counts[a][s] = g as u32;
        }
    }
}

/// Result of clustering: `assign[tile]` is a dense group index in `0..groups`.
pub struct Grouping {
    pub assign: Vec<u16>,
    pub groups: usize,
    /// Estimated bits saved over coding everything with one group (positive = grouping helps).
    pub saving_bits: f64,
}

/// Cluster tiles. `sizes` are the alphabet sizes, `max_groups` bounds the result, `passes` is the
/// number of reassignment sweeps after the initial greedy pass.
pub fn cluster(tiles: &[Sparse], sizes: &[usize; ALPHABETS], max_groups: usize, passes: usize) -> Grouping {
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut assign = vec![0u16; tiles.len()];
    let mut single = Cluster::new(sizes);
    for t in tiles {
        single.apply(t, 1);
    }
    let single_cost = single.cost();

    for (i, tile) in tiles.iter().enumerate() {
        let mut best: Option<(f64, usize)> = None;
        for (ci, c) in clusters.iter().enumerate() {
            let d = c.delta(tile, 1);
            if best.is_none_or(|b| d < b.0) {
                best = Some((d, ci));
            }
        }
        // Cost of giving this tile a cluster of its own.
        let mut fresh = Cluster::new(sizes);
        fresh.apply(tile, 1);
        let own = fresh.cost();
        match best {
            Some((d, ci)) if d <= own || clusters.len() >= max_groups => {
                clusters[ci].apply(tile, 1);
                assign[i] = ci as u16;
            }
            _ => {
                assign[i] = clusters.len() as u16;
                clusters.push(fresh);
            }
        }
    }

    for _ in 0..passes {
        let mut moved = false;
        for (i, tile) in tiles.iter().enumerate() {
            let cur = assign[i] as usize;
            // Cost of the tile where it is, versus anywhere else (measured against the other
            // clusters as they are, and the current one without the tile).
            let stay = clusters[cur].delta(tile, -1);
            let mut best = (-stay, cur);
            for (ci, c) in clusters.iter().enumerate() {
                if ci == cur {
                    continue;
                }
                let d = c.delta(tile, 1);
                if d < best.0 {
                    best = (d, ci);
                }
            }
            if best.1 != cur {
                clusters[cur].apply(tile, -1);
                clusters[best.1].apply(tile, 1);
                assign[i] = best.1 as u16;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    // Drop empty clusters and renumber.
    let mut remap = vec![u16::MAX; clusters.len()];
    let mut groups = 0usize;
    for &a in &assign {
        if remap[a as usize] == u16::MAX {
            remap[a as usize] = groups as u16;
            groups += 1;
        }
    }
    for a in &mut assign {
        *a = remap[*a as usize];
    }
    let total: f64 = clusters.iter().filter(|c| c.total.iter().any(|&t| t > 0.0)).map(Cluster::cost).sum();
    Grouping { assign, groups, saving_bits: single_cost - total }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(sym: u16, n: u32) -> Sparse {
        vec![(0, sym, n), (1, sym, n)]
    }

    #[test]
    fn similar_tiles_merge_and_different_ones_separate() {
        let sizes = [300, 256, 256, 256, 40];
        // Two kinds of tiles, interleaved.
        let tiles: Vec<Sparse> = (0..40).map(|i| if i % 2 == 0 { tile(10, 200) } else { tile(200, 200) }).collect();
        let g = cluster(&tiles, &sizes, 16, 2);
        // Distinct statistics cannot share a group without paying for it.
        assert!(g.groups >= 2);
        assert!((0..40).all(|i| g.assign[i] == g.assign[i % 2]), "{:?}", g.assign);
        assert_ne!(g.assign[0], g.assign[1]);
    }

    #[test]
    fn identical_tiles_share_a_single_group() {
        let sizes = [300, 256, 256, 256, 40];
        let tiles: Vec<Sparse> = (0..30).map(|_| tile(7, 100)).collect();
        let g = cluster(&tiles, &sizes, 16, 2);
        assert_eq!(g.groups, 1);
        assert!(g.assign.iter().all(|&a| a == 0));
    }

    #[test]
    fn group_limit_is_respected() {
        let sizes = [300, 256, 256, 256, 40];
        let tiles: Vec<Sparse> = (0..50).map(|i| tile(i as u16 * 5, 100)).collect();
        let g = cluster(&tiles, &sizes, 4, 1);
        assert!(g.groups <= 4);
    }
}
