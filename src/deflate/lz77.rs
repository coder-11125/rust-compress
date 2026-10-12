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
    /// How many recently productive match distances are tested directly at every position.
    pub hints: usize,
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
        // Optimal parsing: `chain`/`nice` bound the per-position match search, `iters` the refinement
        // passes, `hints` the repeat-distance candidates tested directly.
        let (optimal, iters, chain, nice, hints) = match level {
            7 => (true, 2, 24, 48, 4),
            8 => (true, 4, 96, 128, 8),
            9.. => (true, 8, 512, 258, 16),
            _ => (false, 0, chain, nice, 0),
        };
        Params { good, max_lazy, nice, chain, lazy, optimal, iters, hints }
    }
}

pub struct Matcher {
    head: Vec<u32>,
    prev: Vec<u32>,
    /// Optimal-parse tables: 4-byte hash chains plus the single nearest 3-byte candidate.
    head4: Vec<u32>,
    prev4: Vec<u32>,
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
    // a < b; compare 8 bytes at a time.
    let mut l = 0;
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
        let (h4, p4, h3) = if params.optimal { (1 << HASH_BITS, WINDOW, 1 << HASH_BITS) } else { (0, 0, 0) };
        Self {
            head: vec![NONE; h],
            prev: vec![NONE; p],
            head4: vec![NONE; h4],
            prev4: vec![NONE; p4],
            head3: vec![NONE; h3],
            params,
        }
    }

    /// Insert into the optimal-parse tables; returns (nearest 3-byte candidate, 4-byte chain head).
    #[inline(always)]
    fn insert_opt(&mut self, d: &[u8], pos: usize) -> (u32, u32) {
        if pos + MIN_MATCH > d.len() {
            return (NONE, NONE);
        }
        let h3 = hash3(d, pos);
        let c3 = self.head3[h3];
        self.head3[h3] = pos as u32;
        if pos + 4 > d.len() {
            return (c3, NONE);
        }
        let v = u32::from_le_bytes(d[pos..pos + 4].try_into().unwrap());
        let h4 = (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize;
        let c4 = self.head4[h4];
        self.prev4[pos & (WINDOW - 1)] = c4;
        self.head4[h4] = pos as u32;
        (c3, c4)
    }

    /// Index positions `from..to` as match candidates without emitting tokens (shared history for
    /// independently compressed segments).
    pub fn prime(&mut self, d: &[u8], from: usize, to: usize) {
        for p in from..to {
            if self.params.optimal {
                self.insert_opt(d, p);
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

    /// Insert `pos` and append its Pareto-optimal matches to `out` as (longest length, smallest
    /// distance achieving at least the previous entry's length + 1). Lengths are capped by `limit`.
    pub fn collect_matches(&mut self, d: &[u8], pos: usize, limit: usize, hints: &[u16], out: &mut Vec<(u16, u16)>) {
        let first = out.len();
        let (c3, c4) = self.insert_opt(d, pos);
        let max_len = (limit - pos).min(MAX_MATCH);
        if max_len < MIN_MATCH {
            return;
        }
        let nice = self.params.nice.min(max_len);
        let mut best_len = MIN_MATCH - 1;
        // Nearest triple match first (any 4-byte match is at least this far away).
        if c3 != NONE {
            let c = c3 as usize;
            if pos - c <= WINDOW && d[c] == d[pos] && d[c + 1] == d[pos + 1] && d[c + 2] == d[pos + 2] {
                let l = match_len(d, c, pos, max_len);
                best_len = l;
                out.push((l as u16, (pos - c) as u16));
            }
        }
        let mut chain = self.params.chain;
        let mut cand = c4;
        while cand != NONE && chain > 0 && best_len < nice {
            let c = cand as usize;
            if pos - c > WINDOW {
                break;
            }
            if d[c + best_len] == d[pos + best_len] {
                let l = match_len(d, c, pos, max_len);
                if l > best_len {
                    best_len = l;
                    out.push((l as u16, (pos - c) as u16));
                }
            }
            let next = self.prev4[c & (WINDOW - 1)];
            if next == NONE || next >= cand {
                break;
            }
            cand = next;
            chain -= 1;
        }
        // Distances that recent solutions used a lot (e.g. the row above in images) are tested
        // directly: finding them through the hash chain can take thousands of steps in flat areas.
        for &dd in hints {
            let dd = dd as usize;
            if dd > pos {
                continue;
            }
            let l = match_len(d, pos - dd, pos, max_len);
            if l < MIN_MATCH {
                continue;
            }
            // Longest match already known at distance <= dd; the hint only matters if it beats it.
            let entries = &out[first..];
            let below = entries.iter().take_while(|e| e.1 as usize <= dd).count();
            if below > 0 && entries[below - 1].0 as usize >= l {
                continue;
            }
            let at = first + below;
            while at < out.len() && out[at].0 as usize <= l {
                out.remove(at); // farther and not longer: dominated
            }
            out.insert(at, (l as u16, dd as u16));
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
