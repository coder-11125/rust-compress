//! Expand CLI inputs into (input, output) jobs.

use crate::cli::{Cli, FormatArg};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
    /// Input only: GIFs are converted to PNG or JPEG.
    Gif,
}

impl Format {
    pub fn from_extension(path: &Path) -> Option<Format> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "png" => Some(Format::Png),
            "jpg" | "jpeg" | "jpe" => Some(Format::Jpeg),
            "gif" => Some(Format::Gif),
            _ => None,
        }
    }

    pub fn sniff(data: &[u8]) -> Option<Format> {
        if data.starts_with(&crate::png::SIGNATURE) {
            Some(Format::Png)
        } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            Some(Format::Jpeg)
        } else if crate::gif::is_gif(data) {
            Some(Format::Gif)
        } else {
            None
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Format::Png => "png",
            Format::Jpeg => "jpg",
            Format::Gif => "gif",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Job {
    pub input: PathBuf,
    pub output: PathBuf,
}

/// The path spelling that means standard input (as an input) or standard output (as `-o`).
pub const STDIO: &str = "-";

pub fn is_stdio(p: &Path) -> bool {
    p.as_os_str() == STDIO
}

pub fn target_format(cli: &Cli, input: Format) -> Format {
    match cli.format {
        FormatArg::Keep => input,
        FormatArg::Png => Format::Png,
        FormatArg::Jpeg => Format::Jpeg,
    }
}

fn is_dir_output(cli: &Cli, n_file_inputs: usize, any_dir: bool) -> bool {
    match &cli.output {
        None => false,
        Some(p) => {
            p.is_dir() || n_file_inputs != 1 || any_dir || p.as_os_str().to_string_lossy().ends_with(['/', '\\'])
        }
    }
}

fn output_path(cli: &Cli, input: &Path, rel: &Path, dir_mode: bool) -> PathBuf {
    let fmt_in = Format::from_extension(input);
    let ext_in = input.extension().map(|e| e.to_string_lossy().into_owned());
    // Keep the original extension spelling when the format is unchanged.
    let ext = match (fmt_in, cli.format) {
        (Some(_), FormatArg::Keep) => ext_in.unwrap_or_default(),
        (Some(Format::Png), FormatArg::Png) | (Some(Format::Jpeg), FormatArg::Jpeg) => ext_in.unwrap_or_default(),
        (_, FormatArg::Png) => "png".to_string(),
        (_, FormatArg::Jpeg) => "jpg".to_string(),
        (None, FormatArg::Keep) => String::new(),
    };
    if cli.in_place {
        return input.to_path_buf();
    }
    match &cli.output {
        Some(out) if dir_mode => out.join(rel).with_extension(&ext),
        Some(out) => out.clone(),
        None => {
            let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            input.with_file_name(format!("{stem}.min.{ext}"))
        }
    }
}

fn walk(dir: &Path, recursive: bool, gifs: bool, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        let ty = e.file_type()?;
        if ty.is_dir() {
            if recursive {
                walk(&p, recursive, gifs, out)?;
            }
        } else if ty.is_file() && Format::from_extension(&p).is_some_and(|f| gifs || f != Format::Gif) {
            // Skip our own default outputs so reruns do not recompress them.
            let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            if !stem.ends_with(".min") {
                out.push(p);
            }
        }
    }
    Ok(())
}

/// Returns jobs plus human-readable problems with individual inputs.
pub fn collect(cli: &Cli) -> (Vec<Job>, Vec<String>) {
    let mut problems = Vec::new();
    let to_stdout = cli.output.as_deref().is_some_and(is_stdio);
    let from_stdin = cli.inputs.iter().filter(|p| is_stdio(p)).count();
    if from_stdin > 0 || to_stdout {
        // Streams carry exactly one image: no directory walking, no batches, no in-place edits.
        if cli.inputs.len() != 1 {
            problems.push("standard input/output can only be used with a single input".to_string());
        } else if cli.in_place {
            problems.push("--in-place cannot be combined with standard input/output".to_string());
        } else if cli.inputs[0].is_dir() {
            problems.push(format!("{}: cannot write a directory to standard output", cli.inputs[0].display()));
        } else if from_stdin == 0 && !cli.inputs[0].is_file() {
            problems.push(format!("{}: no such file or directory", cli.inputs[0].display()));
        }
        if !problems.is_empty() {
            return (Vec::new(), problems);
        }
        let input = cli.inputs[0].clone();
        // Reading a pipe with no -o writes to a pipe, the convention for filters.
        let output = if to_stdout || from_stdin > 0 && cli.output.is_none() {
            PathBuf::from(STDIO)
        } else {
            let rel = PathBuf::from(input.file_name().unwrap_or_default());
            output_path(cli, &input, &rel, false)
        };
        return (vec![Job { input, output }], problems);
    }
    let n_file_inputs = cli.inputs.iter().filter(|p| !p.is_dir()).count();
    let any_dir = cli.inputs.iter().any(|p| p.is_dir());
    let dir_mode = is_dir_output(cli, n_file_inputs, any_dir);
    let mut jobs = Vec::new();
    for input in &cli.inputs {
        if input.is_dir() {
            let mut files = Vec::new();
            // GIFs can only be converted, so directory walks skip them unless a target format was chosen.
            if let Err(e) = walk(input, cli.recursive, cli.format != FormatArg::Keep, &mut files) {
                problems.push(format!("{}: {e}", input.display()));
                continue;
            }
            for f in files {
                let rel = f.strip_prefix(input).unwrap_or(&f).to_path_buf();
                let output = output_path(cli, &f, &rel, dir_mode);
                jobs.push(Job { input: f, output });
            }
        } else if input.is_file() {
            let rel = PathBuf::from(input.file_name().unwrap_or_default());
            let output = output_path(cli, input, &rel, dir_mode);
            jobs.push(Job { input: input.clone(), output });
        } else {
            problems.push(format!("{}: no such file or directory", input.display()));
        }
    }
    // Two jobs writing the same file would race; reject the later ones.
    let mut seen = HashSet::new();
    jobs.retain(|j| {
        let fresh = seen.insert(j.output.clone());
        if !fresh {
            problems.push(format!(
                "{}: output {} already used by another input",
                j.input.display(),
                j.output.display()
            ));
        }
        fresh
    });
    (jobs, problems)
}
