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
    pub output: PathBuf,
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

fn json_escape(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_str(out: &mut String, key: &str, value: &str) {
    out.push_str(&format!("\"{key}\":"));
    json_escape(out, value);
}

fn change_percent(from: u64, to: u64) -> String {
    if from == 0 { "null".into() } else { format!("{:.1}", (to as f64 - from as f64) * 100.0 / from as f64) }
}

fn status_name(status: &Status) -> &'static str {
    match status {
        Status::Compressed => "compressed",
        Status::NoGain => "no_gain",
        Status::Failed(_) => "failed",
    }
}

/// One JSON object describing the whole batch: per-file results, the summary and input-level problems.
pub fn json(files: &[Report], summary: &Summary, errors: &[String], secs: f64, dry_run: bool) -> String {
    let mut out = String::new();
    out.push('{');
    out.push_str(&format!("\"dry_run\":{dry_run},\"files\":["));
    for (i, r) in files.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        json_str(&mut out, "input", &r.input.to_string_lossy());
        out.push(',');
        json_str(&mut out, "output", &r.output.to_string_lossy());
        out.push(',');
        json_str(&mut out, "status", status_name(&r.status));
        out.push(',');
        out.push_str(&format!("\"bytes_in\":{},\"bytes_out\":{}", r.in_size, r.out_size));
        out.push(',');
        out.push_str(&format!("\"change_percent\":{}", change_percent(r.in_size, r.out_size)));
        if let Some(note) = &r.note {
            out.push(',');
            json_str(&mut out, "note", note);
        }
        if let Status::Failed(e) = &r.status {
            out.push(',');
            json_str(&mut out, "error", e);
        }
        out.push('}');
    }
    out.push_str("],\"summary\":{");
    out.push_str(&format!(
        "\"files\":{},\"compressed\":{},\"no_gain\":{},\"failed\":{},\"bytes_in\":{},\"bytes_out\":{},\"change_percent\":{},\"seconds\":{secs:.2}",
        summary.files,
        summary.compressed,
        summary.no_gain,
        summary.failed,
        summary.bytes_in,
        summary.bytes_out,
        change_percent(summary.bytes_in, summary.bytes_out),
    ));
    out.push_str("},\"errors\":[");
    for (i, e) in errors.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        json_escape(&mut out, e);
    }
    out.push_str("]}");
    out
}
