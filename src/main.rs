use clap::Parser;
use rayon::prelude::*;
use rust_compress::cli::Cli;
use rust_compress::collect::{collect, is_stdio};
use rust_compress::pipeline::process;
use rust_compress::report::{Status, Summary};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(n) = cli.jobs {
        if n == 0 {
            eprintln!("error: --jobs must be at least 1");
            return ExitCode::from(2);
        }
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global() {
            eprintln!("error: could not configure thread pool: {e}");
            return ExitCode::from(2);
        }
    }
    let start = Instant::now();
    let (jobs, problems) = collect(&cli);
    for p in &problems {
        eprintln!("error: {p}");
    }
    if jobs.is_empty() {
        if problems.is_empty() {
            eprintln!("error: no PNG or JPEG files found (use -r to descend into subdirectories)");
        }
        return ExitCode::from(1);
    }
    // When the image itself goes to stdout, everything else must stay off it.
    let data_on_stdout = jobs.iter().any(|j| is_stdio(&j.output));
    let reports: Vec<_> = jobs
        .par_iter()
        .map(|job| {
            let r = process(job, &cli);
            let failed = matches!(r.status, Status::Failed(_));
            if failed {
                eprintln!("{}", r.line(cli.dry_run));
            } else if !cli.quiet {
                if data_on_stdout {
                    eprintln!("{}", r.line(cli.dry_run));
                } else {
                    println!("{}", r.line(cli.dry_run));
                }
            }
            r
        })
        .collect();
    let mut summary = Summary::default();
    for r in &reports {
        summary.add(r);
    }
    summary.failed += problems.len();
    let line = summary.line(start.elapsed().as_secs_f64());
    if data_on_stdout {
        eprintln!("{line}");
    } else {
        println!("{line}");
    }
    if summary.failed > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS }
}
