use std::path::PathBuf;

#[derive(Debug)]
pub enum Status {
    /// Output written (or would be, for --dry-run).
    Compressed,
    /// Result was not smaller; original kept / copied.
    NoGain,
    Failed(String),
}

#[derive(Debug)]
pub struct Report {
    pub input: PathBuf,
    pub in_size: u64,
    pub out_size: u64,
    pub status: Status,
    /// Extra detail shown after the size change (e.g. the chosen JPEG quality).
    pub note: Option<String>,
}

pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 { format!("{bytes} B") } else { format!("{v:.1} {}", UNITS[u]) }
}

pub fn percent(from: u64, to: u64) -> String {
    if from == 0 {
        return "n/a".into();
    }
    format!("{:+.1}%", (to as f64 - from as f64) * 100.0 / from as f64)
}

impl Report {
    pub fn line(&self, dry_run: bool) -> String {
        let name = self.input.display();
        match &self.status {
            Status::Compressed => format!(
                "{name}  {} -> {}  ({}){}",
                human(self.in_size),
                human(self.out_size),
                percent(self.in_size, self.out_size),
                match (&self.note, dry_run) {
                    (Some(n), true) => format!("  [{n}] [dry run]"),
                    (Some(n), false) => format!("  [{n}]"),
                    (None, true) => "  [dry run]".to_string(),
                    (None, false) => String::new(),
                }
            ),
            Status::NoGain => format!(
                "{name}  {}  (no gain, kept original size){}",
                human(self.in_size),
                self.note.as_ref().map_or(String::new(), |n| format!("  [{n}]"))
            ),
            Status::Failed(e) => format!("{name}  FAILED: {e}"),
        }
    }
}

#[derive(Default, Debug)]
pub struct Summary {
    pub files: usize,
    pub compressed: usize,
    pub no_gain: usize,
    pub failed: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

impl Summary {
    pub fn add(&mut self, r: &Report) {
        self.files += 1;
        match r.status {
            Status::Compressed => self.compressed += 1,
            Status::NoGain => self.no_gain += 1,
            Status::Failed(_) => {
                self.failed += 1;
                return;
            }
        }
        self.bytes_in += r.in_size;
        self.bytes_out += r.out_size;
    }

    pub fn line(&self, secs: f64) -> String {
        format!(
            "{} file(s): {} compressed, {} no gain, {} failed; {} -> {} ({}) in {:.2}s",
            self.files,
            self.compressed,
            self.no_gain,
            self.failed,
            human(self.bytes_in),
            human(self.bytes_out),
            percent(self.bytes_in, self.bytes_out),
            secs
        )
    }
}
