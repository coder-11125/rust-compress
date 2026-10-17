//! AV1 still-picture decoder: OBU stream in, picture out.

use super::bits::BitReader;
use super::cdf_tables::Cdfs;
use super::entropy::SymbolDecoder;
use super::frame::{Frame, Plane};
use super::obu::{self, FrameHeader, SequenceHeader};
use super::tile::{FrameState, TileCoder};
use crate::error::{Error, Result};

/// A decoded picture and the headers it came with.
pub struct Decoded {
    pub frame: Frame,
    pub seq: SequenceHeader,
    pub hdr: FrameHeader,
}

/// Decode the first shown (intra) frame of a low-overhead OBU stream.
pub fn decode(data: &[u8]) -> Result<Decoded> {
    let obus = obu::split_obus(data)?;
    let mut seq: Option<SequenceHeader> = None;
    let mut hdr: Option<FrameHeader> = None;
    let mut state: Option<FrameState> = None;
    let mut tiles_done = 0usize;
    for o in &obus {
        match o.kind {
            obu::OBU_SEQUENCE_HEADER => {
                seq = Some(SequenceHeader::parse(o.payload)?);
            }
            obu::OBU_FRAME_HEADER | obu::OBU_FRAME => {
                if state.is_some() {
                    // A redundant copy of the header, or a second frame: only the first frame is decoded.
                    if o.kind == obu::OBU_FRAME_HEADER {
                        continue;
                    }
                    break;
                }
                let s = seq.as_ref().ok_or_else(|| Error::invalid("AV1 frame before sequence header"))?;
                let mut r = BitReader::new(o.payload);
                let h = FrameHeader::parse(s, &mut r)?;
                check_limits(s, &h)?;
                let fs = FrameState::new(s, &h)?;
                hdr = Some(h);
                state = Some(fs);
                if o.kind == obu::OBU_FRAME {
                    r.byte_align();
                    let off = r.byte_pos();
                    decode_tile_group(
                        state.as_mut().unwrap(),
                        hdr.as_ref().unwrap(),
                        &o.payload[off..],
                        &mut tiles_done,
                    )?;
                }
            }
            obu::OBU_TILE_GROUP => {
                let (Some(fs), Some(h)) = (state.as_mut(), hdr.as_ref()) else {
                    return Err(Error::invalid("AV1 tile group before frame header"));
                };
                decode_tile_group(fs, h, o.payload, &mut tiles_done)?;
            }
            _ => {}
        }
        if let (Some(h), true) = (hdr.as_ref(), state.is_some()) {
            if tiles_done == h.tile_info.cols * h.tile_info.rows {
                break;
            }
        }
    }
    let (Some(seq), Some(hdr), Some(fs)) = (seq, hdr, state) else {
        return Err(Error::invalid("no AV1 frame in stream"));
    };
    if tiles_done != hdr.tile_info.cols * hdr.tile_info.rows {
        return Err(Error::invalid("AV1 frame is missing tiles"));
    }
    let frame = finish_frame(fs)?;
    Ok(Decoded { frame, seq, hdr })
}

fn check_limits(seq: &SequenceHeader, h: &FrameHeader) -> Result<()> {
    let pixels = h.upscaled_width as u64 * h.frame_height as u64;
    if pixels > crate::image::MAX_PIXELS as u64 {
        return Err(Error::unsupported("AV1 picture too large"));
    }
    if seq.color.mono_chrome && h.frame_type > 3 {
        return Err(Error::invalid("bad frame type"));
    }
    Ok(())
}

fn decode_tile_group(fs: &mut FrameState, hdr: &FrameHeader, data: &[u8], done: &mut usize) -> Result<()> {
    let ti = &hdr.tile_info;
    let num_tiles = ti.cols * ti.rows;
    let mut r = BitReader::new(data);
    let mut start = 0;
    let mut end = num_tiles - 1;
    if num_tiles > 1 && r.flag()? {
        let bits = ti.cols_log2 + ti.rows_log2;
        start = r.bits(bits)? as usize;
        end = r.bits(bits)? as usize;
    }
    r.byte_align();
    let mut pos = r.byte_pos();
    if start >= num_tiles || end >= num_tiles || end < start {
        return Err(Error::invalid("bad AV1 tile group"));
    }
    for tile_num in start..=end {
        let tile_row = tile_num / ti.cols;
        let tile_col = tile_num % ti.cols;
        let size = if tile_num == end {
            data.len().checked_sub(pos).ok_or_else(|| Error::invalid("truncated AV1 tile"))?
        } else {
            let n = ti.tile_size_bytes;
            let b = data.get(pos..pos + n).ok_or_else(|| Error::invalid("truncated AV1 tile size"))?;
            pos += n;
            let mut v = 0usize;
            for (i, &x) in b.iter().enumerate() {
                v |= (x as usize) << (8 * i);
            }
            v + 1
        };
        let tile = data.get(pos..pos + size).ok_or_else(|| Error::invalid("truncated AV1 tile"))?;
        pos += size;
        let cdf = Cdfs::new(hdr.quant.base_q_idx as u32);
        let dec = SymbolDecoder::new(tile, !hdr.disable_cdf_update);
        let rows = (ti.mi_row_starts[tile_row], ti.mi_row_starts[tile_row + 1]);
        let cols = (ti.mi_col_starts[tile_col], ti.mi_col_starts[tile_col + 1]);
        let mut tc = TileCoder::new(fs, cdf, dec, rows, cols);
        tc.decode_tile()?;
        *done += 1;
    }
    Ok(())
}

/// Post-filter and crop the decoded frame to its visible size.
fn finish_frame(fs: FrameState) -> Result<Frame> {
    let hdr = &fs.hdr;
    let seq = &fs.seq;
    let (w, h) = (hdr.upscaled_width, hdr.frame_height);
    let mut planes = Vec::new();
    for (p, plane) in fs.planes.iter().enumerate() {
        let (sx, sy) = if p == 0 { (0, 0) } else { (fs.sub_x, fs.sub_y) };
        let (pw, ph) = ((w + sx) >> sx, (h + sy) >> sy);
        let mut out = Plane::new(pw, ph);
        for y in 0..ph {
            out.row_mut(y).copy_from_slice(&plane.row(y)[..pw]);
        }
        planes.push(out);
    }
    Ok(Frame {
        planes,
        bit_depth: seq.color.bit_depth,
        sub_x: seq.color.sub_x,
        sub_y: seq.color.sub_y,
        width: w,
        height: h,
    })
}
