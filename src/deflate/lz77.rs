//! LZ77 match finder: hash chains with greedy (levels 1-3) or lazy (4-9) parsing.

pub const WINDOW: usize = 32768;
pub const MIN_MATCH: usize = 3;
pub const MAX_MATCH: usize = 258;

const HASH_BITS: u32 = 16;
const NONE: u32 = u32::MAX;
/// Length-3 matches farther than this are cheaper as literals.
const TOO_FAR: usize = 4096;

/// A literal byte or a (length, distance) pair packed in a u32.
#[derive(Clone, Copy, Debug)]
pub struct Token(pub u32);

impl Token {
    const MATCH_FLAG: u32 = 1 << 31;

    #[inline(always)]
    pub fn literal(b: u8) -> Self {
        Token(b as u32)
    }

    #[inline(always)]
    pub fn matched(len: usize, dist: usize) -> Self {
        debug_assert!((MIN_MATCH..=MAX_MATCH).contains(&len) && (1..=WINDOW).contains(&dist));
        Token(Self::MATCH_FLAG | (len as u32) << 16 | dist as u32)
    }

    #[inline(always)]
    pub fn is_match(self) -> bool {
        self.0 & Self::MATCH_FLAG != 0
    }

    #[inline(always)]
    pub fn length(self) -> usize {
        ((self.0 >> 16) & 0x1FF) as usize
    }

    #[inline(always)]
    pub fn dist(self) -> usize {
        (self.0 & 0xFFFF) as usize
    }

    #[inline(always)]
    pub fn lit(self) -> u8 {
        self.0 as u8
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub good: usize,
    pub max_lazy: usize,
    pub nice: usize,
    pub chain: usize,
    pub lazy: bool,
    /// Optimal parsing (levels 7-9): `iters` cost-model refinement passes.
    pub optimal: bool,
    pub iters: u32,
}

impl Params {
    pub fn for_level(level: u32) -> Self {
        let (good, max_lazy, nice, chain, lazy) = match level {
            0 | 1 => (4, 4, 8, 4, false),
            2 => (4, 5, 16, 8, false),
            3 => (4, 6, 32, 32, false),
            4 => (4, 4, 16, 16, true),
            5 => (8, 16, 32, 32, true),
            6 => (8, 16, 128, 128, true),
            _ => (32, 258, 258, 0, true),
        };
        // Optimal parsing: `chain` is the binary-tree search depth and `nice` the match length at which
        // a search stops; `iters` bounds the cost-model refinement passes.
        let (optimal, iters, chain, nice) = match level {
            7 => (true, 2, 48, 128),
            8 => (true, 4, 96, 128),
            9.. => (true, 15, 512, 258),
            _ => (false, 0, chain, nice),
        };
        Params { good, max_lazy, nice, chain, lazy, optimal, iters }
    }
}

pub struct Matcher {
    head: Vec<u32>,
    prev: Vec<u32>,
    /// Optimal-parse tables: a binary tree of suffixes per 4-byte hash (`son[2i]` = smaller child,
    /// `son[2i + 1]` = larger child of the position `i` mod WINDOW) plus the nearest 3-byte candidate.
    head4: Vec<u32>,
    son: Vec<u32>,
    head3: Vec<u32>,
    params: Params,
}

#[inline(always)]
fn hash3(d: &[u8], p: usize) -> usize {
    let v = d[p] as u32 | (d[p + 1] as u32) << 8 | (d[p + 2] as u32) << 16;
    (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
}

#[inline(always)]
fn match_len(d: &[u8], a: usize, b: usize, max: usize) -> usize {
    extend_len(d, a, b, 0, max)
}

/// Length of the common prefix of `d[a..]` and `d[b..]` (at most `max`), given that the first `from`
/// bytes are already known to be equal. a < b.
#[inline(always)]
fn extend_len(d: &[u8], a: usize, b: usize, from: usize, max: usize) -> usize {
    // Compare 8 bytes at a time.
    let mut l = from;
    while l + 8 <= max {
        let x = u64::from_le_bytes(d[a + l..a + l + 8].try_into().unwrap());
        let y = u64::from_le_bytes(d[b + l..b + l + 8].try_into().unwrap());
        let z = x ^ y;
        if z != 0 {
            return l + (z.trailing_zeros() >> 3) as usize;
        }
        l += 8;
    }
    while l < max && d[a + l] == d[b + l] {
        l += 1;
    }
    l
}

impl Matcher {
    pub fn new(params: Params) -> Self {
        let (h, p) = if params.optimal { (0, 0) } else { (1 << HASH_BITS, WINDOW) };
        let (h4, p4, h3) = if params.optimal { (1 << HASH_BITS, 2 * WINDOW, 1 << HASH_BITS) } else { (0, 0, 0) };
        Self {
            head: vec![NONE; h],
            prev: vec![NONE; p],
            head4: vec![NONE; h4],
            son: vec![NONE; p4],
            head3: vec![NONE; h3],
            params,
        }
    }

    /// Index positions `from..to` as match candidates without emitting tokens (shared history for
    /// independently compressed segments).
    pub fn prime(&mut self, d: &[u8], from: usize, to: usize) {
        let mut scratch = Vec::new();
        for p in from..to {
            if self.params.optimal {
                scratch.clear();
                self.collect_matches(d, p, d.len(), &mut scratch);
            } else {
                self.insert(d, p);
            }
        }
    }

    #[inline(always)]
    fn insert(&mut self, d: &[u8], pos: usize) -> u32 {
        if pos + MIN_MATCH > d.len() {
            return NONE;
        }
        let h = hash3(d, pos);
        let old = self.head[h];
        self.prev[pos & (WINDOW - 1)] = old;
        self.head[h] = pos as u32;
        old
    }

    fn longest(&self, d: &[u8], pos: usize, mut cand: u32, prev_len: usize) -> (usize, usize) {
        let max_len = (d.len() - pos).min(MAX_MATCH);
        if max_len < MIN_MATCH || cand == NONE {
            return (0, 0);
        }
        let p = &self.params;
        let mut chain = p.chain;
        if prev_len >= p.good {
            chain >>= 2;
        }
        let nice = p.nice.min(max_len);
        let (mut best_len, mut best_dist) = (prev_len.max(MIN_MATCH - 1), 0);
        while chain > 0 && best_len < max_len {
            let c = cand as usize;
            if pos - c > WINDOW {
                break;
            }
            if d[c + best_len] == d[pos + best_len] && d[c] == d[pos] {
                let l = match_len(d, c, pos, max_len);
                if l > best_len {
                    best_len = l;
                    best_dist = pos - c;
                    if l >= nice {
                        break;
                    }
                }
            }
            let next = self.prev[c & (WINDOW - 1)];
            if next == NONE || next >= cand {
                break;
            }
            cand = next;
            chain -= 1;
        }
        if best_dist == 0 {
            return (0, 0);
        }
        if best_len == MIN_MATCH && best_dist > TOO_FAR {
            return (0, 0);
        }
        (best_len, best_dist)
    }

    /// Insert `pos` into the suffix tree and append its Pareto-optimal matches to `out` as (longest
    /// length, smallest distance achieving at least the previous entry's length + 1), lengths and
    /// distances both increasing. Lengths are capped by `limit`; the tree itself always looks at all
    /// of `d` so its ordering does not depend on where the current chunk ends.
    pub fn collect_matches(&mut self, d: &[u8], pos: usize, limit: usize, out: &mut Vec<(u16, u16)>) {
        let first = out.len();
        if pos + MIN_MATCH > d.len() {
            return;
        }
        // Nearest 3-byte candidate.
        let h3 = hash3(d, pos);
        let c3 = self.head3[h3];
        self.head3[h3] = pos as u32;
        let out_max = (limit - pos).min(MAX_MATCH);
        let mut rec_best = MIN_MATCH - 1;
        if c3 != NONE && out_max >= MIN_MATCH {
            let c = c3 as usize;
            if pos - c < WINDOW && d[c] == d[pos] && d[c + 1] == d[pos + 1] && d[c + 2] == d[pos + 2] {
                let l = match_len(d, c, pos, out_max);
                rec_best = l;
                out.push((l as u16, (pos - c) as u16));
            }
        }
        if pos + 4 > d.len() {
            return;
        }
        // Binary-tree search (LZMA-style): walking down from the hash bucket's root, every visited node
        // is re-linked as a child of `pos`, so inserting and searching are one pass.
        let mask = WINDOW - 1;
        let tree_max = (d.len() - pos).min(MAX_MATCH);
        let nice = self.params.nice.min(tree_max);
        let v = u32::from_le_bytes(d[pos..pos + 4].try_into().unwrap());
        let h4 = (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize;
        let mut cur = self.head4[h4];
        self.head4[h4] = pos as u32;
        let (mut slot_small, mut slot_large) = ((pos & mask) * 2, (pos & mask) * 2 + 1);
        let (mut len_small, mut len_large) = (0usize, 0usize);
        let mut depth = self.params.chain;
        loop {
            if cur == NONE || depth == 0 || pos - cur as usize >= WINDOW {
                self.son[slot_small] = NONE;
                self.son[slot_large] = NONE;
                break;
            }
            depth -= 1;
            let c = cur as usize;
            let node = (c & mask) * 2;
            let len = extend_len(d, c, pos, len_small.min(len_large), tree_max);
            let l = len.min(out_max);
            if l > rec_best {
                rec_best = l;
                let dist = (pos - c) as u16;
                // A longer match at a smaller distance makes the earlier entries redundant.
                while out.len() > first && out[out.len() - 1].1 >= dist {
                    out.pop();
                }
                out.push((l as u16, dist));
            }
            if len >= nice {
                // Good enough: `pos` takes this node's place in the tree.
                self.son[slot_small] = self.son[node];
                self.son[slot_large] = self.son[node + 1];
                break;
            }
            if d[c + len] < d[pos + len] {
                self.son[slot_small] = cur;
                slot_small = node + 1;
                cur = self.son[slot_small];
                len_small = len;
            } else {
                self.son[slot_large] = cur;
                slot_large = node;
                cur = self.son[slot_large];
                len_large = len;
            }
        }
        // The prefix lengths above are inferred from the tree order; never emit a match unchecked.
        let mut keep = first;
        for i in first..out.len() {
            let (l, dist) = out[i];
            if match_len(d, pos - dist as usize, pos, l as usize) == l as usize {
                out.swap(keep, i);
                keep += 1;
            }
        }
        out.truncate(keep);
        if out_max < MIN_MATCH {
            out.truncate(first);
        }
    }

    /// Tokenize `d` starting at `*pos` until `max_tokens` tokens were produced
    /// or the input is exhausted. Hash state persists across calls.
    pub fn tokenize(&mut self, d: &[u8], pos: &mut usize, tokens: &mut Vec<Token>, max_tokens: usize) {
        let n = d.len();
        let params = self.params;
        while *pos < n && tokens.len() < max_tokens {
            let cand = self.insert(d, *pos);
            let (mut len, mut dist) = self.longest(d, *pos, cand, 0);
            if len == 0 {
                tokens.push(Token::literal(d[*pos]));
                *pos += 1;
                continue;
            }
            // `inserted` = number of positions after *pos already in the hash.
            let mut inserted = 0;
            if params.lazy {
                while len < params.max_lazy && *pos + 1 < n {
                    let c2 = self.insert(d, *pos + 1);
                    let (l2, d2) = self.longest(d, *pos + 1, c2, len);
                    if l2 > len {
                        tokens.push(Token::literal(d[*pos]));
                        *pos += 1;
                        len = l2;
                        dist = d2;
                        inserted = 0;
                    } else {
                        inserted = 1;
                        break;
                    }
                }
            }
            tokens.push(Token::matched(len, dist));
            let end = *pos + len;
            // Greedy fast levels skip hashing inside long matches.
            if params.lazy || len <= params.max_lazy {
                for p in *pos + 1 + inserted..end {
                    self.insert(d, p);
                }
            }
            *pos = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Repeats, runs and noise, so there are matches at many lengths and distances.
    fn sample(n: usize) -> Vec<u8> {
        let mut s = 99u32;
        let mut v: Vec<u8> = Vec::with_capacity(n);
        while v.len() < n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            match (s >> 29) & 3 {
                0 => v.extend(std::iter::repeat_n((s >> 8) as u8 & 7, (s >> 16) as usize % 40)),
                1 if v.len() > 64 => {
                    let st = (s >> 8) as usize % (v.len() - 32);
                    for k in 0..3 + (s >> 4) as usize % 50 {
                        v.push(v[st + k % 32]);
                    }
                }
                _ => v.extend((0..(s >> 20) % 12).map(|i| (s >> (i % 20)) as u8 & 15)),
            }
        }
        v.truncate(n);
        v
    }

    fn brute_longest(d: &[u8], pos: usize) -> usize {
        (pos.saturating_sub(WINDOW - 1)..pos)
            .map(|c| match_len(d, c, pos, (d.len() - pos).min(MAX_MATCH)))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn tree_matches_are_valid_monotone_and_find_the_longest() {
        let d = sample(6000);
        let mut m = Matcher::new(Params::for_level(9));
        let mut out = Vec::new();
        for pos in 0..d.len() {
            out.clear();
            m.collect_matches(&d, pos, d.len(), &mut out);
            for w in out.windows(2) {
                assert!(w[0].0 < w[1].0 && w[0].1 < w[1].1, "not monotone at {pos}: {out:?}");
            }
            for &(l, dist) in &out {
                let (l, dist) = (l as usize, dist as usize);
                assert!(l >= MIN_MATCH && (1..WINDOW).contains(&dist) && dist <= pos, "bad entry at {pos}");
                assert_eq!(match_len(&d, pos - dist, pos, l), l, "invalid match at {pos}");
            }
            // The tree is keyed on 4 bytes, so a longest match of 4+ bytes must be found exactly.
            let best = brute_longest(&d, pos);
            if best >= 4 {
                assert_eq!(out.last().map(|e| e.0 as usize), Some(best), "longest match missed at {pos}");
            }
        }
    }

    #[test]
    fn lengths_are_capped_by_the_chunk_limit() {
        let d = vec![5u8; 2000];
        let mut m = Matcher::new(Params::for_level(7));
        let mut out = Vec::new();
        for pos in 0..1000 {
            out.clear();
            m.collect_matches(&d, pos, 1000, &mut out);
            assert!(out.iter().all(|e| pos + e.0 as usize <= 1000), "match crosses the limit at {pos}");
        }
    }
}
