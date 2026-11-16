use clap::Parser;
use rayon::prelude::*;
mod budget;
mod cli;
mod collect;
mod pipeline;
mod report;
mod strip;
mod update;

use crate::cli::{Cli, Command};
use crate::collect::{collect, is_stdio};
use crate::pipeline::process;
use crate::report::{Status, Summary};
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(Command::Update(args)) = &cli.command {
        return update::run(args);
    }
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
    if let Some(n) = cli.max_pixels {
        rc_core::image::set_max_pixels(n);
    }
    let budget = budget::Budget::new(cli.max_memory << 20);
    let start = Instant::now();
    let (jobs, problems) = collect(&cli);
    if !cli.json {
        for p in &problems {
            eprintln!("error: {}", report::sanitize(p));
        }
    }
    if jobs.is_empty() {
        if cli.json {
            let summary = Summary { failed: problems.len(), ..Summary::default() };
            println!("{}", report::json(&[], &summary, &problems, start.elapsed().as_secs_f64(), cli.dry_run));
        } else if problems.is_empty() {
            eprintln!(
                "error: no supported image files found (PNG, JPEG, WebP, AVIF, GIF; use -r to descend into subdirectories)"
            );
        }
        return ExitCode::from(1);
    }
    // When the image itself goes to stdout, everything else must stay off it.
    let data_on_stdout = jobs.iter().any(|j| is_stdio(&j.output));
    let reports: Vec<_> = jobs
        .par_iter()
        .map(|job| {
            let r = process(job, &cli, &budget);
            if !cli.json {
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
            }
            r
        })
        .collect();
    let mut summary = Summary::default();
    for r in &reports {
        summary.add(r);
    }
    summary.failed += problems.len();
    if cli.json {
        let json = report::json(&reports, &summary, &problems, start.elapsed().as_secs_f64(), cli.dry_run);
        if data_on_stdout {
            eprintln!("{json}");
        } else {
            println!("{json}");
        }
    } else {
        let line = summary.line(start.elapsed().as_secs_f64());
        if data_on_stdout {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
    }
    if summary.failed > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS }
}
