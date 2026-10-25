//! Minimal EXIF (TIFF) editing, so kept metadata stays truthful when the pixels change shape:
//! the orientation tag is reset once the rotation is baked into the pixels, the recorded pixel
//! dimensions follow a resize, and a thumbnail that no longer matches the image is dropped.
//!
//! The block is the TIFF structure itself (what a PNG `eXIf` chunk holds and what follows the
//! `Exif\0\0` prefix of a JPEG APP1 segment). Anything malformed leaves the block untouched.

const TAG_ORIENTATION: u16 = 0x0112;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_GPS_IFD: u16 = 0x8825;
const TAG_INTEROP_IFD: u16 = 0xA005;
const TAG_PIXEL_X: u16 = 0xA002;
const TAG_PIXEL_Y: u16 = 0xA003;

const MAX_ENTRIES: usize = 2048;
const MAX_DEPTH: usize = 4;

/// What to change in an EXIF block.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fix {
    /// The rotation/flip is now part of the pixels: set the orientation tag to 1 (upright).
    pub reset_orientation: bool,
    /// New pixel dimensions to record in the Exif sub-IFD (only when the tags already exist).
    pub dims: Option<(u32, u32)>,
    /// The pixels changed shape, so the embedded thumbnail no longer matches: unlink and drop it.
    pub drop_thumbnail: bool,
}

struct Reader<'a> {
    b: &'a [u8],
    le: bool,
}

impl Reader<'_> {
    fn u16(&self, at: usize) -> Option<u16> {
        let v: [u8; 2] = self.b.get(at..at + 2)?.try_into().ok()?;
        Some(if self.le { u16::from_le_bytes(v) } else { u16::from_be_bytes(v) })
    }

    fn u32(&self, at: usize) -> Option<u32> {
        let v: [u8; 4] = self.b.get(at..at + 4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(v) } else { u32::from_be_bytes(v) })
    }
}

/// Edits found by the read-only pass, applied only if the whole walk succeeded.
#[derive(Default)]
struct Plan {
    /// (offset, bytes) overwrites.
    patches: Vec<(usize, Vec<u8>)>,
    /// Highest byte referenced by any IFD or value in the tree walked from IFD0.
    max_end: usize,
    /// A value of an unknown type was seen, so `max_end` cannot be trusted.
    unknown_type: bool,
    visited: Vec<usize>,
}

fn type_size(t: u16) -> Option<usize> {
    Some(match t {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => return None,
    })
}

fn put(r: &Reader, v: u32, size: usize) -> Vec<u8> {
    match (size, r.le) {
        (2, true) => (v as u16).to_le_bytes().to_vec(),
        (2, false) => (v as u16).to_be_bytes().to_vec(),
        (_, true) => v.to_le_bytes().to_vec(),
        (_, false) => v.to_be_bytes().to_vec(),
    }
}

/// Walk one IFD (and the sub-IFDs it points to). Returns the offset of the next IFD in the chain.
fn walk(r: &Reader, ifd: usize, depth: usize, fix: &Fix, plan: &mut Plan, in_exif: bool) -> Option<u32> {
    if depth > MAX_DEPTH || plan.visited.contains(&ifd) {
        return None;
    }
    plan.visited.push(ifd);
    let n = r.u16(ifd)? as usize;
    if n > MAX_ENTRIES {
        return None;
    }
    plan.max_end = plan.max_end.max(ifd + 2 + n * 12 + 4);
    for i in 0..n {
        let e = ifd + 2 + i * 12;
        let tag = r.u16(e)?;
        let ty = r.u16(e + 2)?;
        let count = r.u32(e + 4)? as usize;
        match type_size(ty) {
            Some(sz) => {
                let bytes = sz.checked_mul(count)?;
                if bytes > 4 {
                    let off = r.u32(e + 8)? as usize;
                    let end = off.checked_add(bytes)?;
                    if end > r.b.len() {
                        return None;
                    }
                    plan.max_end = plan.max_end.max(end);
                }
            }
            None => plan.unknown_type = true,
        }
        match tag {
            TAG_ORIENTATION if depth == 0 && fix.reset_orientation && ty == 3 && count == 1 => {
                plan.patches.push((e + 8, put(r, 1, 2)));
            }
            TAG_PIXEL_X | TAG_PIXEL_Y if in_exif && count == 1 => {
                if let Some((w, h)) = fix.dims {
                    let v = if tag == TAG_PIXEL_X { w } else { h };
                    match ty {
                        3 if v <= u16::MAX as u32 => plan.patches.push((e + 8, put(r, v, 2))),
                        4 => plan.patches.push((e + 8, put(r, v, 4))),
                        _ => {}
                    }
                }
            }
            TAG_EXIF_IFD | TAG_GPS_IFD | TAG_INTEROP_IFD if ty == 4 && count == 1 => {
                let sub = r.u32(e + 8)? as usize;
                walk(r, sub, depth + 1, fix, plan, tag == TAG_EXIF_IFD || (in_exif && tag == TAG_INTEROP_IFD))?;
            }
            _ => {}
        }
    }
    r.u32(ifd + 2 + n * 12)
}

/// Apply `fix` to the EXIF block in place. Malformed blocks are left as they are.
pub fn tidy(tiff: &mut Vec<u8>, fix: Fix) {
    let le = match tiff.get(..4) {
        Some(b"II*\0") => true,
        Some(b"MM\0*") => false,
        _ => return,
    };
    let r = Reader { b: tiff, le };
    let Some(ifd0) = r.u32(4) else { return };
    let mut plan = Plan { max_end: 8, ..Default::default() };
    let Some(next) = walk(&r, ifd0 as usize, 0, &fix, &mut plan, false) else { return };
    let n0 = r.u16(ifd0 as usize).unwrap_or(0) as usize;
    let next_ptr_at = ifd0 as usize + 2 + n0 * 12;
    let Plan { patches, max_end, unknown_type, .. } = plan;
    for (at, bytes) in patches {
        tiff[at..at + bytes.len()].copy_from_slice(&bytes);
    }
    if fix.drop_thumbnail && next != 0 {
        tiff[next_ptr_at..next_ptr_at + 4].copy_from_slice(&[0; 4]);
        // The thumbnail normally sits after everything else; drop the bytes too when that is provably so.
        if !unknown_type && next as usize >= max_end && (next as usize) < tiff.len() {
            tiff.truncate(next as usize);
        }
    }
}

/// Orientation recorded in an EXIF block (1..=8), if any.
pub fn orientation(tiff: &[u8]) -> Option<u8> {
    let le = match tiff.get(..4)? {
        b"II*\0" => true,
        b"MM\0*" => false,
        _ => return None,
    };
    let r = Reader { b: tiff, le };
    let ifd = r.u32(4)? as usize;
    let n = r.u16(ifd)? as usize;
    (0..n.min(MAX_ENTRIES)).find_map(|i| {
        let e = ifd + 2 + i * 12;
        (r.u16(e)? == TAG_ORIENTATION).then(|| r.u16(e + 8)).flatten().filter(|v| (1..=8).contains(v)).map(|v| v as u8)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Little-endian TIFF: IFD0 {orientation=6, ExifIFD ptr}, ExifIFD {PixelX=4000, PixelY=3000},
    /// then IFD1 (thumbnail) and 8 bytes of "thumbnail" data after everything else.
    fn sample(le: bool) -> Vec<u8> {
        let w16 = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let w32 = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let mut b = Vec::new();
        b.extend_from_slice(if le { b"II*\0" } else { b"MM\0*" });
        b.extend_from_slice(&w32(8));
        // IFD0 at 8: 2 entries, next -> IFD1 at 68.
        b.extend_from_slice(&w16(2));
        for (tag, ty, val) in [(TAG_ORIENTATION, 3u16, 6u32), (TAG_EXIF_IFD, 4, 38)] {
            b.extend_from_slice(&w16(tag));
            b.extend_from_slice(&w16(ty));
            b.extend_from_slice(&w32(1));
            if ty == 3 {
                b.extend_from_slice(&w16(val as u16));
                b.extend_from_slice(&[0, 0]);
            } else {
                b.extend_from_slice(&w32(val));
            }
        }
        b.extend_from_slice(&w32(68));
        assert_eq!(b.len(), 38);
        // ExifIFD at 38: PixelX, PixelY (LONG).
        b.extend_from_slice(&w16(2));
        for (tag, v) in [(TAG_PIXEL_X, 4000u32), (TAG_PIXEL_Y, 3000)] {
            b.extend_from_slice(&w16(tag));
            b.extend_from_slice(&w16(4));
            b.extend_from_slice(&w32(1));
            b.extend_from_slice(&w32(v));
        }
        b.extend_from_slice(&w32(0));
        assert_eq!(b.len(), 68);
        // IFD1 at 68: no entries, then thumbnail bytes.
        b.extend_from_slice(&w16(0));
        b.extend_from_slice(&w32(0));
        b.extend_from_slice(b"THUMBNAIL");
        b
    }

    #[test]
    fn orientation_is_read_in_both_byte_orders() {
        assert_eq!(orientation(&sample(true)), Some(6));
        assert_eq!(orientation(&sample(false)), Some(6));
        assert_eq!(orientation(b"garbage"), None);
    }

    #[test]
    fn fix_resets_orientation_updates_dims_and_drops_thumbnail() {
        for le in [true, false] {
            let mut t = sample(le);
            tidy(&mut t, Fix { reset_orientation: true, dims: Some((1000, 750)), drop_thumbnail: true });
            assert_eq!(orientation(&t), Some(1));
            let r = Reader { b: &t, le };
            assert_eq!(r.u32(38 + 2 + 8), Some(1000));
            assert_eq!(r.u32(38 + 2 + 12 + 8), Some(750));
            assert_eq!(r.u32(8 + 2 + 24), Some(0), "IFD1 unlinked");
            assert_eq!(t.len(), 68, "IFD1 and thumbnail bytes removed");
        }
    }

    #[test]
    fn untouched_when_nothing_is_requested() {
        let mut t = sample(true);
        let before = t.clone();
        tidy(&mut t, Fix::default());
        assert_eq!(t, before);
    }

    #[test]
    fn malformed_blocks_are_left_alone() {
        let fix = Fix { reset_orientation: true, dims: Some((1, 1)), drop_thumbnail: true };
        let good = sample(true);
        // Every truncation and a self-referencing IFD must not panic or half-apply.
        for cut in 0..good.len() {
            let mut t = good[..cut].to_vec();
            tidy(&mut t, fix);
        }
        let mut looped = good.clone();
        looped[8 + 2 + 12 + 8..8 + 2 + 12 + 12].copy_from_slice(&8u32.to_le_bytes());
        let mut t = looped.clone();
        tidy(&mut t, fix);
        assert_eq!(t, looped);
    }
}
