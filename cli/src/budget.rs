//! A soft memory budget for the images being processed at the same time.
//!
//! Every worker thread decodes a whole image, and a small file can expand to gigabytes (a 1 MB PNG of 150 megapixels at
//! 16 bits needs about 3 GB). On a machine with many cores a batch of such files used to be decoded all at once, and
//! running out of memory aborts the whole process without even a report. Here each image first reserves an estimate of
//! what it needs, read from its header, and waits while other images hold too much. An image that alone exceeds the
//! budget still runs, but by itself.

use std::sync::{Condvar, Mutex, MutexGuard};

pub struct Budget {
    limit: u64,
    used: Mutex<u64>,
    freed: Condvar,
}

/// A reservation; the memory is given back when it is dropped.
pub struct Permit<'a> {
    budget: &'a Budget,
    bytes: u64,
}

impl Budget {
    pub fn new(limit: u64) -> Self {
        Budget { limit, used: Mutex::new(0), freed: Condvar::new() }
    }

    fn lock(&self) -> MutexGuard<'_, u64> {
        self.used.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Reserve `bytes`, waiting until that fits next to what is reserved already. Nothing is reserved beyond the whole
    /// budget, and nothing waits when nothing else is running.
    pub fn acquire(&self, bytes: u64) -> Permit<'_> {
        let bytes = bytes.min(self.limit);
        let mut used = self.lock();
        while *used > 0 && used.saturating_add(bytes) > self.limit {
            used = self.freed.wait(used).unwrap_or_else(|e| e.into_inner());
        }
        *used += bytes;
        Permit { budget: self, bytes }
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        *self.budget.lock() -= self.bytes;
        self.budget.freed.notify_all();
    }
}

fn be16(d: &[u8], at: usize) -> Option<u64> {
    Some(u16::from_be_bytes(d.get(at..at + 2)?.try_into().ok()?) as u64)
}

fn le24(d: &[u8], at: usize) -> Option<u64> {
    let b = d.get(at..at + 3)?;
    Some(b[0] as u64 | (b[1] as u64) << 8 | (b[2] as u64) << 16)
}

/// Pixel count and an estimate of the working memory per pixel, read from the header only.
fn header_size(d: &[u8]) -> Option<(u64, u64)> {
    if d.starts_with(&rc_png::SIGNATURE) {
        let w = u32::from_be_bytes(d.get(16..20)?.try_into().ok()?) as u64;
        let h = u32::from_be_bytes(d.get(20..24)?.try_into().ok()?) as u64;
        let samples: u64 = match d.get(25)? {
            0 => 1,
            4 => 2,
            2 => 3,
            _ => 4, // palette (expanded to RGBA) and RGBA
        };
        let bytes = if *d.get(24)? == 16 { 2 } else { 1 };
        // Measured: about 2.5 times the decoded image (RGBA16 150 MP: 2.9 GB).
        return Some((w * h, (samples * bytes * 5).div_ceil(2)));
    }
    if d.starts_with(&[0xFF, 0xD8]) {
        let mut pos = 2;
        while d.get(pos) == Some(&0xFF) {
            let m = *d.get(pos + 1)?;
            match m {
                0xFF => pos += 1,
                0x01 | 0xD0..=0xD8 => pos += 2,
                0xD9 | 0xDA => break,
                0xC0..=0xCF if !matches!(m, 0xC4 | 0xC8 | 0xCC) => {
                    return Some((be16(d, pos + 7)? * be16(d, pos + 5)?, 16));
                }
                _ => pos += 2 + be16(d, pos + 2)? as usize,
            }
        }
        return None;
    }
    if d.starts_with(b"GIF8") {
        let w = u16::from_le_bytes(d.get(6..8)?.try_into().ok()?) as u64;
        let h = u16::from_le_bytes(d.get(8..10)?.try_into().ok()?) as u64;
        return Some((w * h, 16));
    }
    if d.len() >= 30 && &d[..4] == b"RIFF" && &d[8..12] == b"WEBP" {
        let (w, h) = match &d[12..16] {
            b"VP8X" => (le24(d, 24)? + 1, le24(d, 27)? + 1),
            b"VP8L" => {
                let v = u32::from_le_bytes(d.get(21..25)?.try_into().ok()?) as u64;
                ((v & 0x3FFF) + 1, ((v >> 14) & 0x3FFF) + 1)
            }
            b"VP8 " => {
                let w = u16::from_le_bytes(d.get(26..28)?.try_into().ok()?) as u64 & 0x3FFF;
                let h = u16::from_le_bytes(d.get(28..30)?.try_into().ok()?) as u64 & 0x3FFF;
                (w, h)
            }
            _ => return None,
        };
        return Some((w * h, 16));
    }
    None
}

/// Rough number of bytes needed to process this image; 0 when the header says nothing (AVIF, unknown formats).
pub fn estimate_bytes(data: &[u8]) -> u64 {
    header_size(data).map_or(0, |(pixels, per_pixel)| pixels.saturating_mul(per_pixel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc_testutil::fixture;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[test]
    fn a_second_big_image_waits_until_the_first_is_done() {
        let budget = Budget::new(100);
        let started = AtomicBool::new(false);
        let first = budget.acquire(60);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                let _second = budget.acquire(60);
                started.store(true, Ordering::SeqCst);
            });
            std::thread::sleep(Duration::from_millis(150));
            assert!(!started.load(Ordering::SeqCst), "60 + 60 does not fit in 100");
            drop(first);
            waiter.join().unwrap();
            assert!(started.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn small_images_run_together_and_an_oversized_one_runs_alone() {
        let budget = Budget::new(100);
        let _a = budget.acquire(30);
        let _b = budget.acquire(30);
        drop((_a, _b));
        // Larger than the whole budget, but nothing else is running: it must not wait forever.
        let big = budget.acquire(10_000);
        assert_eq!(*budget.lock(), 100, "the reservation is capped at the budget");
        drop(big);
        assert_eq!(*budget.lock(), 0);
    }

    #[test]
    fn estimates_come_from_the_header() {
        let png = fixture("rgb8.png"); // 67x45, RGB, 8-bit
        assert_eq!(estimate_bytes(&png), 67 * 45 * 8); // 3 samples * 2.5, rounded up
        assert_eq!(estimate_bytes(&fixture("rgb8.jpg")), 67 * 45 * 16);
        assert_eq!(estimate_bytes(&fixture("gif_plain.gif")), header_size(&fixture("gif_plain.gif")).unwrap().0 * 16);
        assert!(estimate_bytes(&fixture("webp_rgb.webp")) > 0);
        assert!(estimate_bytes(&fixture("webp_lossy.webp")) > 0);
        assert_eq!(estimate_bytes(b"not an image"), 0);
        assert_eq!(estimate_bytes(&[]), 0);
    }
}
