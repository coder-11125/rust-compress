//! The `ALPH` chunk: the alpha plane of a lossy WebP image, stored raw or as a headerless
//! lossless image stream, optionally predicted from neighbouring alpha values.

use super::vp8l;
use crate::error::{Error, Result};

/// How alpha samples are predicted before coding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    None = 0,
    Horizontal = 1,
    Vertical = 2,
    Gradient = 3,
}

impl Filter {
    fn from_bits(b: u8) -> Filter {
        match b & 3 {
            0 => Filter::None,
            1 => Filter::Horizontal,
            2 => Filter::Vertical,
            _ => Filter::Gradient,
        }
    }
}

#[inline(always)]
fn gradient(a: u8, b: u8, c: u8) -> u8 {
    (a as i32 + b as i32 - c as i32).clamp(0, 255) as u8
}

/// Undo the prediction filter in place: each sample holds the difference to its prediction.
pub fn unfilter(plane: &mut [u8], width: usize, filter: Filter) {
    if filter == Filter::None {
        return;
    }
    let height = plane.len() / width;
    for y in 0..height {
        let (done, rest) = plane.split_at_mut(y * width);
        let row = &mut rest[..width];
        let prev = (y > 0).then(|| &done[(y - 1) * width..]);
        match (filter, prev) {
            (Filter::Vertical, Some(p)) => {
                for x in 0..width {
                    row[x] = row[x].wrapping_add(p[x]);
                }
            }
            (Filter::Gradient, Some(p)) => {
                let (mut top_left, mut left) = (p[0], p[0]);
                for x in 0..width {
                    let top = p[x];
                    left = row[x].wrapping_add(gradient(left, top, top_left));
                    top_left = top;
                    row[x] = left;
                }
            }
            // Horizontal prediction, and the first row of every filter: from the left, with the
            // first sample of a row predicted from the one above it.
            _ => {
                let mut pred = prev.map_or(0, |p| p[0]);
                for v in row.iter_mut() {
                    *v = v.wrapping_add(pred);
                    pred = *v;
                }
            }
        }
    }
}

/// Apply the prediction filter (the inverse of [`unfilter`]).
pub fn filter(plane: &[u8], width: usize, kind: Filter) -> Vec<u8> {
    if kind == Filter::None {
        return plane.to_vec();
    }
    let height = plane.len() / width;
    let mut out = vec![0u8; plane.len()];
    for y in 0..height {
        let row = &plane[y * width..(y + 1) * width];
        let prev = (y > 0).then(|| &plane[(y - 1) * width..y * width]);
        for x in 0..width {
            let pred = match (kind, prev) {
                (Filter::Vertical, Some(p)) => p[x],
                (Filter::Gradient, Some(p)) => {
                    if x == 0 {
                        p[0]
                    } else {
                        gradient(row[x - 1], p[x], p[x - 1])
                    }
                }
                _ => {
                    if x == 0 {
                        prev.map_or(0, |p| p[0])
                    } else {
                        row[x - 1]
                    }
                }
            };
            out[y * width + x] = row[x].wrapping_sub(pred);
        }
    }
    out
}

/// Decode an `ALPH` chunk body into a `width` x `height` alpha plane.
pub fn decode(data: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let (&head, body) = data.split_first().ok_or_else(|| Error::invalid("empty ALPH chunk"))?;
    let method = head & 3;
    let filter = Filter::from_bits(head >> 2);
    let n = width.checked_mul(height).ok_or_else(|| Error::invalid("alpha plane too large"))?;
    let mut plane = match method {
        0 => body.get(..n).ok_or_else(|| Error::invalid("truncated raw alpha plane"))?.to_vec(),
        1 => vp8l::decode_headerless(body, width, height)?.iter().map(|p| (p >> 8) as u8).collect(),
        _ => return Err(Error::unsupported("alpha compression method")),
    };
    unfilter(&mut plane, width, filter);
    Ok(plane)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_filter_roundtrips() {
        let (w, h) = (13usize, 9usize);
        let plane: Vec<u8> = (0..w * h).map(|i| ((i * 37 + i / w * 11) % 251) as u8).collect();
        for kind in [Filter::None, Filter::Horizontal, Filter::Vertical, Filter::Gradient] {
            let mut f = filter(&plane, w, kind);
            unfilter(&mut f, w, kind);
            assert_eq!(f, plane, "{kind:?}");
        }
    }

    #[test]
    fn raw_alpha_is_read_back() {
        let mut chunk = vec![0u8]; // raw, unfiltered
        chunk.extend((0..12u8).map(|v| v * 20));
        assert_eq!(decode(&chunk, 4, 3).unwrap(), (0..12u8).map(|v| v * 20).collect::<Vec<_>>());
        assert!(decode(&chunk[..5], 4, 3).is_err());
        assert!(decode(&[], 4, 3).is_err());
        assert!(decode(&[2], 4, 3).is_err());
    }
}
