//! Remove descriptive metadata from a file without touching its pixel data.
//!
//! When recompressing does not make a file smaller, `rc` keeps the original bytes. Those bytes still carry whatever
//! EXIF, text chunks or comments the file came with, although by default `rc` promises to keep only colour
//! information. This works on the container only (PNG chunks, JPEG segments, WebP chunks): the compressed image data
//! is copied as it is, so the pixels cannot change and the result is never larger than the input.
//!
//! What counts as descriptive matches the decoders (`Metadata::drop_descriptive`): EXIF, XMP, text, time and
//! pixel-density chunks, JPEG comments and Photoshop/other application blocks. ICC profiles, colour chunks and the
//! Adobe colour-transform marker stay. A file whose EXIF says the image must be rotated is left alone, since removing
//! the tag would change how it is displayed. GIF and AVIF are not handled.

use crate::collect::Format;
use rc_core::exif;
use rc_core::image::plausible_icc;

/// `data` without its descriptive metadata, or `None` when there is nothing to remove, the file is not laid out as
/// expected, or removing the metadata would change how the image is shown.
pub fn descriptive_metadata(data: &[u8], format: Format) -> Option<Vec<u8>> {
    let stripped = match format {
        Format::Png => png(data),
        Format::Jpeg => jpeg(data),
        Format::Webp => webp(data),
        Format::Gif | Format::Avif => None,
    }?;
    (stripped.len() < data.len()).then_some(stripped)
}

fn rotates(tiff: &[u8]) -> bool {
    exif::orientation(tiff).is_some_and(|o| o > 1)
}

fn png(data: &[u8]) -> Option<Vec<u8>> {
    if !data.starts_with(&rc_png::SIGNATURE) {
        return None;
    }
    let mut out = Vec::with_capacity(data.len());
    out.extend_from_slice(&data[..8]);
    let mut pos = 8;
    let mut saw_end = false;
    while pos + 12 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        // length + type + body + crc
        let end = pos.checked_add(12)?.checked_add(len)?;
        if end > data.len() {
            return None;
        }
        let ty = &data[pos + 4..pos + 8];
        match ty {
            b"eXIf" if rotates(&data[pos + 8..pos + 8 + len]) => return None,
            b"tEXt" | b"zTXt" | b"iTXt" | b"tIME" | b"pHYs" | b"eXIf" => {}
            _ => out.extend_from_slice(&data[pos..end]),
        }
        pos = end;
        if ty == b"IEND" {
            saw_end = true;
            break;
        }
    }
    // Bytes after IEND are not part of the image either.
    saw_end.then_some(out)
}

/// Whether a JPEG segment stays: everything that is not an application block or comment, plus the application
/// blocks that matter for display (JFIF, the ICC profile, Adobe's colour transform).
fn keep_segment(marker: u8, body: &[u8], icc_ok: bool) -> bool {
    match marker {
        0xE0 => body.starts_with(b"JFIF\0"),
        0xE2 => icc_ok && body.starts_with(b"ICC_PROFILE\0"),
        0xEE => body.starts_with(b"Adobe"),
        0xE1 | 0xE3..=0xED | 0xEF | 0xFE => false,
        _ => true,
    }
}

/// The ICC profile carried in a JPEG's `APP2` segments (all of them come before the first scan), reassembled in
/// sequence order, or nothing if there is none.
fn jpeg_icc_profile(data: &[u8]) -> Vec<u8> {
    let mut parts: Vec<(u8, &[u8])> = Vec::new();
    let mut pos = 2;
    while data.get(pos) == Some(&0xFF) {
        let Some(&marker) = data.get(pos + 1) else { break };
        if marker == 0xFF {
            pos += 1;
            continue;
        }
        if matches!(marker, 0xD9 | 0xDA) {
            break;
        }
        if matches!(marker, 0x01 | 0xD0..=0xD8) {
            pos += 2;
            continue;
        }
        let (Some(&hi), Some(&lo)) = (data.get(pos + 2), data.get(pos + 3)) else { break };
        let end = pos + 2 + u16::from_be_bytes([hi, lo]) as usize;
        if end > data.len() || end < pos + 4 {
            break;
        }
        let body = &data[pos + 4..end];
        if marker == 0xE2 && body.len() > 14 && body.starts_with(b"ICC_PROFILE\0") {
            parts.push((body[12], &body[14..]));
        }
        pos = end;
    }
    parts.sort_by_key(|p| p.0);
    parts.into_iter().flat_map(|p| p.1.iter().copied()).collect()
}

fn jpeg(data: &[u8]) -> Option<Vec<u8>> {
    if !data.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    // A profile that is not a plausible one is dropped like any other junk application block.
    let icc_ok = plausible_icc(&jpeg_icc_profile(data));
    let mut out = vec![0xFF, 0xD8];
    let mut pos = 2;
    loop {
        // Skip fill bytes in front of a marker.
        while data.get(pos) == Some(&0xFF) && data.get(pos + 1) == Some(&0xFF) {
            pos += 1;
        }
        if data.get(pos)? != &0xFF {
            return None;
        }
        let marker = *data.get(pos + 1)?;
        match marker {
            0xD9 => {
                out.extend_from_slice(&[0xFF, 0xD9]);
                return Some(out);
            }
            // Markers without a length (and restart markers, which only occur inside scans).
            0x01 | 0xD0..=0xD8 => {
                out.extend_from_slice(&data[pos..pos + 2]);
                pos += 2;
            }
            _ => {
                let len = u16::from_be_bytes([*data.get(pos + 2)?, *data.get(pos + 3)?]) as usize;
                let end = pos.checked_add(2)?.checked_add(len)?;
                if len < 2 || end > data.len() {
                    return None;
                }
                let body = &data[pos + 4..end];
                if marker == 0xE1 && body.starts_with(b"Exif\0\0") && rotates(&body[6..]) {
                    return None;
                }
                if keep_segment(marker, body, icc_ok) {
                    out.extend_from_slice(&data[pos..end]);
                }
                pos = end;
                if marker == 0xDA {
                    // Entropy-coded data runs up to the next marker that is not a stuffed zero or a restart.
                    let start = pos;
                    loop {
                        let b = *data.get(pos)?;
                        let next = *data.get(pos + 1)?;
                        if b == 0xFF && next != 0 && next != 0xFF && !(0xD0..=0xD7).contains(&next) {
                            break;
                        }
                        pos += 1;
                    }
                    out.extend_from_slice(&data[start..pos]);
                }
            }
        }
    }
}

fn webp(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return None;
    }
    let riff_end = (8 + u32::from_le_bytes(data[4..8].try_into().ok()?) as usize).min(data.len());
    let mut out = data[..12].to_vec();
    let mut pos = 12;
    while pos + 8 <= riff_end {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().ok()?) as usize;
        let end = pos.checked_add(8)?.checked_add(size)?.checked_add(size & 1)?;
        if end > riff_end {
            return None;
        }
        match id {
            b"EXIF" => {
                if rotates(strip_exif_header(&data[pos + 8..pos + 8 + size])) {
                    return None;
                }
            }
            b"XMP " => {}
            b"VP8X" if size >= 1 => {
                let mut chunk = data[pos..end].to_vec();
                chunk[8] &= !(0x08 | 0x04); // the EXIF and XMP flags
                out.extend_from_slice(&chunk);
            }
            _ => out.extend_from_slice(&data[pos..end]),
        }
        pos = end;
    }
    let riff_len = u32::try_from(out.len() - 8).ok()?;
    out[4..8].copy_from_slice(&riff_len.to_le_bytes());
    Some(out)
}

/// Some writers put the JPEG `Exif\0\0` prefix in front of the TIFF block of a WebP `EXIF` chunk.
fn strip_exif_header(body: &[u8]) -> &[u8] {
    body.strip_prefix(b"Exif\0\0").unwrap_or(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc_testutil::fixture;

    fn png_chunk(ty: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut c = (body.len() as u32).to_be_bytes().to_vec();
        let mut crc_in = ty.to_vec();
        crc_in.extend_from_slice(body);
        c.extend_from_slice(&crc_in);
        c.extend_from_slice(&rc_core::util::checksum::crc32(&crc_in).to_be_bytes());
        c
    }

    /// `png` with `chunk` inserted right after IHDR.
    fn png_with(png: &[u8], chunk: &[u8]) -> Vec<u8> {
        let ihdr_end = 8 + 12 + 13;
        [&png[..ihdr_end], chunk, &png[ihdr_end..]].concat()
    }

    #[test]
    fn png_text_time_density_and_exif_chunks_are_removed() {
        let original = fixture("rgb8.png");
        assert!(descriptive_metadata(&original, Format::Png).is_none(), "the fixture has nothing to remove");
        let mut dirty = original.clone();
        for (ty, body) in [
            (b"tEXt", &b"Author\0Jane Doe"[..]),
            (b"tIME", &[7, 232, 1, 2, 3, 4, 5][..]),
            (b"pHYs", &[0, 0, 11, 19, 0, 0, 11, 19, 1][..]),
            (b"eXIf", &b"II*\0\x08\0\0\0\0\0\0\0\0\0"[..]),
        ] {
            dirty = png_with(&dirty, &png_chunk(ty, body));
        }
        assert_eq!(descriptive_metadata(&dirty, Format::Png).unwrap(), original);
    }

    #[test]
    fn png_colour_chunks_and_pixels_stay_and_a_rotating_exif_blocks_stripping() {
        let original = fixture("rgb8.png");
        let with_gamma = png_with(&original, &png_chunk(b"gAMA", &[0, 1, 0x86, 0xA0]));
        assert!(descriptive_metadata(&with_gamma, Format::Png).is_none());
        // EXIF with orientation 6 (rotate): removing it would change the displayed image.
        let mut tiff = b"II*\0\x08\0\0\0\x01\0".to_vec();
        tiff.extend_from_slice(&[0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0]);
        let rotated = png_with(&original, &png_chunk(b"eXIf", &tiff));
        assert!(descriptive_metadata(&rotated, Format::Png).is_none());
    }

    #[test]
    fn jpeg_comments_exif_and_application_blocks_are_removed() {
        let original = fixture("rgb8.jpg");
        assert!(descriptive_metadata(&original, Format::Jpeg).is_none(), "the fixture has nothing to remove");
        let seg = |marker: u8, body: &[u8]| {
            let mut s = vec![0xFF, marker];
            s.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
            s.extend_from_slice(body);
            s
        };
        let mut dirty = vec![0xFF, 0xD8];
        dirty.extend(seg(0xFE, b"made by Jane Doe at home"));
        dirty.extend(seg(0xE1, b"Exif\0\0II*\0\x08\0\0\0\0\0\0\0\0\0"));
        dirty.extend(seg(0xED, b"Photoshop 3.0\0..."));
        dirty.extend_from_slice(&original[2..]);
        assert_eq!(descriptive_metadata(&dirty, Format::Jpeg).unwrap(), original);
        // A truncated file is left alone.
        assert!(descriptive_metadata(&dirty[..dirty.len() - 30], Format::Jpeg).is_none());
    }

    #[test]
    fn jpeg_keeps_a_real_icc_profile_and_drops_a_fake_one() {
        let original = fixture("rgb8.jpg");
        let seg = |body: &[u8]| {
            let mut s = vec![0xFF, 0xE2];
            s.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
            s.extend_from_slice(body);
            s
        };
        let with_icc = |profile: &[u8]| {
            // Split over two segments to cover reassembly.
            let (a, b) = profile.split_at(profile.len() / 2);
            let part = |seq: u8, bytes: &[u8]| seg(&[b"ICC_PROFILE\0".as_slice(), &[seq, 2], bytes].concat());
            [&original[..2], &part(1, a), &part(2, b), &original[2..]].concat()
        };
        let mut real = vec![0u8; 800];
        real[..4].copy_from_slice(&800u32.to_be_bytes());
        real[36..40].copy_from_slice(b"acsp");
        let good = with_icc(&real);
        assert!(descriptive_metadata(&good, Format::Jpeg).is_none(), "a plausible profile is colour information");
        let junk = with_icc(&vec![0x41; 3000]);
        assert_eq!(descriptive_metadata(&junk, Format::Jpeg).unwrap(), original);
    }

    #[test]
    fn webp_exif_and_xmp_chunks_are_removed_and_the_header_fixed() {
        let chunk = |id: &[u8; 4], body: &[u8]| {
            let mut c = id.to_vec();
            c.extend_from_slice(&(body.len() as u32).to_le_bytes());
            c.extend_from_slice(body);
            if body.len() % 2 == 1 {
                c.push(0);
            }
            c
        };
        let riff = |chunks: Vec<Vec<u8>>| {
            let body: Vec<u8> = chunks.concat();
            let mut f = b"RIFF".to_vec();
            f.extend_from_slice(&(body.len() as u32 + 4).to_le_bytes());
            f.extend_from_slice(b"WEBP");
            f.extend(body);
            f
        };
        let image = chunk(b"VP8L", &[0x2f, 1, 2, 3, 4, 5, 6]);
        let vp8x = |flags: u8| chunk(b"VP8X", &[flags, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let dirty = riff(vec![
            vp8x(0x0C),
            image.clone(),
            chunk(b"EXIF", b"II*\0\x08\0\0\0\0\0\0\0\0\0"),
            chunk(b"XMP ", b"<x:xmpmeta/>"),
        ]);
        assert_eq!(descriptive_metadata(&dirty, Format::Webp).unwrap(), riff(vec![vp8x(0), image.clone()]));
        // Flags other than EXIF/XMP survive (here: alpha).
        let alpha = riff(vec![vp8x(0x1C), image.clone(), chunk(b"EXIF", b"II*\0\x08\0\0\0\0\0\0\0\0\0")]);
        assert_eq!(descriptive_metadata(&alpha, Format::Webp).unwrap(), riff(vec![vp8x(0x10), image]));
        assert!(descriptive_metadata(&fixture("webp_rgb.webp"), Format::Webp).is_none());
    }

    #[test]
    fn unsupported_or_broken_input_is_never_touched() {
        for fmt in [Format::Png, Format::Jpeg, Format::Webp, Format::Gif, Format::Avif] {
            assert!(descriptive_metadata(b"", fmt).is_none());
            assert!(descriptive_metadata(b"not an image at all, just text", fmt).is_none());
        }
    }
}
