use super::{CL_ORDER, DIST_BASE, DIST_EXTRA, LEN_BASE, LEN_EXTRA};
use rc_core::error::{Error, Result};
use rc_core::util::bitio::LsbReader;
use rc_core::util::checksum::adler32;
use rc_core::util::huffman::Decoder;

fn fixed_decoders() -> (Decoder, Decoder) {
    let mut ll = [8u8; 288];
    ll[144..256].fill(9);
    ll[256..280].fill(7);
    (Decoder::new(&ll).unwrap(), Decoder::new(&[5u8; 30]).unwrap())
}

/// Decompress a raw DEFLATE stream. `max_out` bounds the output size.
pub fn inflate(data: &[u8], size_hint: usize, max_out: usize) -> Result<Vec<u8>> {
    inflate_with_end(data, size_hint, max_out).map(|(v, _)| v)
}

/// Like [`inflate`], also returning how many input bytes the stream occupied.
pub fn inflate_with_end(data: &[u8], size_hint: usize, max_out: usize) -> Result<(Vec<u8>, usize)> {
    let mut r = LsbReader::new(data);
    let mut out: Vec<u8> = Vec::with_capacity(size_hint.min(max_out).min(1 << 26));
    let mut fixed: Option<(Decoder, Decoder)> = None;
    loop {
        r.refill();
        let last = r.take(1) == 1;
        let kind = r.take(2);
        match kind {
            0 => {
                let hdr = r.align_and_take(4).ok_or_else(|| Error::invalid("truncated stored block"))?;
                let len = u16::from_le_bytes([hdr[0], hdr[1]]);
                let nlen = u16::from_le_bytes([hdr[2], hdr[3]]);
                if len != !nlen {
                    return Err(Error::invalid("stored block length mismatch"));
                }
                let raw = r.align_and_take(len as usize).ok_or_else(|| Error::invalid("truncated stored block"))?;
                if out.len() + raw.len() > max_out {
                    return Err(Error::invalid("output exceeds size limit"));
                }
                out.extend_from_slice(raw);
            }
            1 => {
                let (ll, d) = fixed.get_or_insert_with(fixed_decoders);
                inflate_block(&mut r, &mut out, ll, d, max_out)?;
            }
            2 => {
                let (ll, d) = read_dynamic_tables(&mut r)?;
                inflate_block(&mut r, &mut out, &ll, &d, max_out)?;
            }
            _ => return Err(Error::invalid("reserved deflate block type")),
        }
        if r.overran() {
            return Err(Error::invalid("truncated deflate stream"));
        }
        if last {
            break;
        }
    }
    Ok((out, r.byte_pos()))
}

fn read_dynamic_tables(r: &mut LsbReader) -> Result<(Decoder, Decoder)> {
    r.refill();
    let hlit = r.take(5) as usize + 257;
    let hdist = r.take(5) as usize + 1;
    let hclen = r.take(4) as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(Error::invalid("too many length/distance codes"));
    }
    let mut cl = [0u8; 19];
    for &idx in CL_ORDER.iter().take(hclen) {
        r.refill();
        cl[idx] = r.take(3) as u8;
    }
    let cl_dec = Decoder::new(&cl)?;
    let mut lengths = [0u8; 286 + 30];
    let total = hlit + hdist;
    let mut i = 0;
    while i < total {
        r.refill();
        let sym = cl_dec.decode(r)?;
        let (val, rep) = match sym {
            0..=15 => (sym as u8, 1),
            16 => {
                if i == 0 {
                    return Err(Error::invalid("repeat with no previous length"));
                }
                (lengths[i - 1], 3 + r.take(2) as usize)
            }
            17 => (0, 3 + r.take(3) as usize),
            18 => (0, 11 + r.take(7) as usize),
            _ => return Err(Error::invalid("bad code-length symbol")),
        };
        if i + rep > total {
            return Err(Error::invalid("code lengths overflow"));
        }
        lengths[i..i + rep].fill(val);
        i += rep;
    }
    if lengths[256] == 0 {
        return Err(Error::invalid("missing end-of-block code"));
    }
    Ok((Decoder::new(&lengths[..hlit])?, Decoder::new(&lengths[hlit..total])?))
}

fn inflate_block(r: &mut LsbReader, out: &mut Vec<u8>, ll: &Decoder, dist: &Decoder, max_out: usize) -> Result<()> {
    loop {
        r.refill();
        let sym = ll.decode(r)? as usize;
        if sym < 256 {
            if out.len() >= max_out {
                return Err(Error::invalid("output exceeds size limit"));
            }
            out.push(sym as u8);
            continue;
        }
        if sym == 256 {
            return Ok(());
        }
        let li = sym - 257;
        if li >= 29 {
            return Err(Error::invalid("bad length symbol"));
        }
        let len = LEN_BASE[li] as usize + r.take(LEN_EXTRA[li] as u32) as usize;
        r.refill();
        let ds = dist.decode(r)? as usize;
        if ds >= 30 {
            return Err(Error::invalid("bad distance symbol"));
        }
        let d = DIST_BASE[ds] as usize + r.take(DIST_EXTRA[ds] as u32) as usize;
        if d > out.len() {
            return Err(Error::invalid("distance too far back"));
        }
        if out.len() + len > max_out {
            return Err(Error::invalid("output exceeds size limit"));
        }
        let start = out.len() - d;
        if d >= len {
            out.extend_from_within(start..start + len);
        } else {
            for k in 0..len {
                let b = out[start + k];
                out.push(b);
            }
        }
        if r.overran() {
            return Err(Error::invalid("truncated deflate stream"));
        }
    }
}

/// Decompress a zlib stream (verifying the Adler-32 trailer).
pub fn zlib_decompress(data: &[u8], size_hint: usize, max_out: usize) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(Error::invalid("zlib stream too short"));
    }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0F != 8 || (cmf >> 4) > 7 || !((cmf as u16) << 8 | flg as u16).is_multiple_of(31) {
        return Err(Error::invalid("bad zlib header"));
    }
    if flg & 0x20 != 0 {
        return Err(Error::unsupported("zlib preset dictionary"));
    }
    let (out, used) = inflate_with_end(&data[2..], size_hint, max_out)?;
    let tail = data.get(2 + used..2 + used + 4).ok_or_else(|| Error::invalid("missing adler32"))?;
    if u32::from_be_bytes(tail.try_into().unwrap()) != adler32(&out) {
        return Err(Error::invalid("adler32 mismatch"));
    }
    Ok(out)
}

/// Structure of a DEFLATE stream, for tuning the encoder against other implementations.
#[derive(Debug, Default)]
pub struct StreamStats {
    pub blocks: usize,
    pub stored_blocks: usize,
    pub fixed_blocks: usize,
    pub literals: u64,
    pub matches: u64,
    pub match_bytes: u64,
    /// Matches by length bucket: 3, 4, 5-8, 9-16, 17-64, 65-257, 258.
    pub len_buckets: [u64; 7],
    /// Matches by distance: <=3, <=16, <=256, <=4096, larger.
    pub dist_buckets: [u64; 5],
    /// Bits spent on dynamic-block headers.
    pub header_bits: u64,
}

/// Walk a raw DEFLATE stream and collect [`StreamStats`].
pub fn stream_stats(data: &[u8]) -> Result<StreamStats> {
    let mut st = StreamStats::default();
    let mut r = LsbReader::new(data);
    let mut out_len = 0usize;
    let fixed = fixed_decoders();
    loop {
        r.refill();
        let last = r.take(1) == 1;
        let kind = r.take(2);
        st.blocks += 1;
        let (ll, dist) = match kind {
            0 => {
                st.stored_blocks += 1;
                let hdr = r.align_and_take(4).ok_or_else(|| Error::invalid("truncated"))?;
                let len = u16::from_le_bytes([hdr[0], hdr[1]]) as usize;
                r.align_and_take(len).ok_or_else(|| Error::invalid("truncated"))?;
                out_len += len;
                if last {
                    break;
                }
                continue;
            }
            1 => {
                st.fixed_blocks += 1;
                (None, None)
            }
            _ => {
                let before = r.byte_pos() * 8;
                let (l, d) = read_dynamic_tables(&mut r)?;
                st.header_bits += (r.byte_pos() * 8 - before) as u64;
                (Some(l), Some(d))
            }
        };
        let (ll, dist) = match (&ll, &dist) {
            (Some(l), Some(d)) => (l, d),
            _ => (&fixed.0, &fixed.1),
        };
        loop {
            r.refill();
            let sym = ll.decode(&mut r)? as usize;
            if sym < 256 {
                st.literals += 1;
                out_len += 1;
                continue;
            }
            if sym == 256 {
                break;
            }
            let li = sym - 257;
            let len = LEN_BASE[li] as usize + r.take(LEN_EXTRA[li] as u32) as usize;
            r.refill();
            let ds = dist.decode(&mut r)? as usize;
            let d = DIST_BASE[ds] as usize + r.take(DIST_EXTRA[ds] as u32) as usize;
            st.matches += 1;
            st.match_bytes += len as u64;
            st.len_buckets[match len {
                3 => 0,
                4 => 1,
                5..=8 => 2,
                9..=16 => 3,
                17..=64 => 4,
                65..=257 => 5,
                _ => 6,
            }] += 1;
            st.dist_buckets[match d {
                0..=3 => 0,
                4..=16 => 1,
                17..=256 => 2,
                257..=4096 => 3,
                _ => 4,
            }] += 1;
            out_len += len;
        }
        if last {
            break;
        }
    }
    let _ = out_len;
    Ok(st)
}
