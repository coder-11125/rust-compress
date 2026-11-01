//! `rc update` against a throwaway local server that plays GitHub's release endpoints. The tests that use a shell-script
//! stand-in for the new binary are Unix-only; the real-binary test also runs on Windows.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::tempdir;

const CURRENT: &str = env!("CARGO_PKG_VERSION");

/// Serve `/releases/latest` (redirecting to `tag`) and the release assets; returns the base URL.
fn serve(tag: &str, archive: Vec<u8>, checksum: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let tag = tag.to_string();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let mut parts = req.split_whitespace();
            let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
            let (status, extra, body): (&str, String, Vec<u8>) = if path == "/releases/latest" {
                ("302 Found", format!("Location: /releases/tag/{tag}\r\n"), Vec::new())
            } else if path == format!("/releases/tag/{tag}") {
                ("200 OK", String::new(), Vec::new())
            } else if path.starts_with(&format!("/releases/download/{tag}/")) && path.ends_with(".sha256") {
                ("200 OK", String::new(), checksum.clone().into_bytes())
            } else if path.starts_with(&format!("/releases/download/{tag}/")) {
                ("200 OK", String::new(), archive.clone())
            } else {
                ("404 Not Found", String::new(), Vec::new())
            };
            let head =
                format!("HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = s.write_all(head.as_bytes());
            if method != "HEAD" {
                let _ = s.write_all(&body);
            }
        }
    });
    base
}

/// A tar.gz holding a fake `rc` (a script that identifies itself as `version`) and its SHA-256 line.
#[cfg(unix)]
fn release_archive(dir: &Path, version: &str) -> (Vec<u8>, String) {
    archive_of_script(dir, &format!("#!/bin/sh\necho \"rc {version}\"\n"))
}

#[cfg(unix)]
fn archive_of_script(dir: &Path, script: &str) -> (Vec<u8>, String) {
    let src = dir.join("pkg");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("rc"), script).unwrap();
    let tgz = dir.join("release.tar.gz");
    let ok = Command::new("tar").arg("czf").arg(&tgz).arg("-C").arg(&src).arg("rc").status().unwrap().success();
    assert!(ok);
    let data = fs::read(&tgz).unwrap();
    let out = ["sha256sum", "shasum"]
        .iter()
        .find_map(|tool| {
            let mut c = Command::new(tool);
            if *tool == "shasum" {
                c.args(["-a", "256"]);
            }
            c.arg(&tgz).output().ok().filter(|o| o.status.success())
        })
        .expect("sha256sum or shasum");
    (data, String::from_utf8_lossy(&out.stdout).to_string())
}

/// A private copy of the real binary to update (never the one cargo built).
fn installed_copy(dir: &Path) -> PathBuf {
    let copy = dir.join(exe_name());
    fs::copy(assert_cmd::cargo::cargo_bin("rc"), &copy).unwrap();
    copy
}

fn run_update(rc: &Path, base: &str, args: &[&str]) -> Output {
    Command::new(rc).arg("update").args(args).env("RC_UPDATE_BASE_URL", base).output().unwrap()
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).to_string()
}

fn leftovers(dir: &Path) -> Vec<String> {
    fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with(".rc-update"))
        .collect()
}

#[cfg(unix)]
#[test]
fn replaces_the_binary_and_shows_both_versions() {
    let d = tempdir().unwrap();
    let (archive, sum) = release_archive(d.path(), "9.9.9");
    let base = serve("v9.9.9", archive, sum);
    let bin_dir = d.path().join("bin");
    fs::create_dir(&bin_dir).unwrap();
    let rc = installed_copy(&bin_dir);

    let out = run_update(&rc, &base, &[]);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains(&format!("Updating rc {CURRENT} -> 9.9.9")), "{stdout}");
    assert!(stdout.contains("Checksum verified."), "{stdout}");

    let after = Command::new(&rc).arg("--version").output().unwrap();
    assert_eq!(text(&after.stdout).trim(), "rc 9.9.9");
    assert!(leftovers(&bin_dir).is_empty(), "{:?}", leftovers(&bin_dir));
}

#[cfg(unix)]
#[test]
fn check_reports_without_touching_anything() {
    let d = tempdir().unwrap();
    let (archive, sum) = release_archive(d.path(), "9.9.9");
    let base = serve("v9.9.9", archive, sum);
    let bin_dir = d.path().join("bin");
    fs::create_dir(&bin_dir).unwrap();
    let rc = installed_copy(&bin_dir);
    let before = fs::read(&rc).unwrap();

    let out = run_update(&rc, &base, &["--check"]);
    let stdout = text(&out.stdout);
    assert!(out.status.success(), "{stdout}\n{}", text(&out.stderr));
    assert!(stdout.contains(&format!("{CURRENT} -> 9.9.9")), "{stdout}");
    assert_eq!(fs::read(&rc).unwrap(), before);
    assert!(leftovers(&bin_dir).is_empty());
}

#[cfg(unix)]
#[test]
fn already_current_does_nothing() {
    let d = tempdir().unwrap();
    let (archive, sum) = release_archive(d.path(), CURRENT);
    let base = serve(&format!("v{CURRENT}"), archive, sum);
    let bin_dir = d.path().join("bin");
    fs::create_dir(&bin_dir).unwrap();
    let rc = installed_copy(&bin_dir);
    let before = fs::read(&rc).unwrap();

    let out = run_update(&rc, &base, &[]);
    assert!(out.status.success());
    assert!(text(&out.stdout).contains("is up to date"), "{}", text(&out.stdout));
    assert_eq!(fs::read(&rc).unwrap(), before);
}

#[cfg(unix)]
#[test]
fn a_bad_checksum_keeps_the_installed_binary() {
    let d = tempdir().unwrap();
    let (archive, _) = release_archive(d.path(), "9.9.9");
    let wrong = format!("{}  release.tar.gz\n", "0".repeat(64));
    let base = serve("v9.9.9", archive, wrong);
    let bin_dir = d.path().join("bin");
    fs::create_dir(&bin_dir).unwrap();
    let rc = installed_copy(&bin_dir);
    let before = fs::read(&rc).unwrap();

    let out = run_update(&rc, &base, &[]);
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("checksum mismatch"), "{}", text(&out.stderr));
    assert_eq!(fs::read(&rc).unwrap(), before);
    assert!(leftovers(&bin_dir).is_empty());
}

#[cfg(unix)]
#[test]
fn a_broken_download_is_not_installed() {
    let d = tempdir().unwrap();
    // A valid archive (right checksum) whose `rc` fails its --version check.
    let (archive, sum) = archive_of_script(d.path(), "#!/bin/sh\nexit 3\n");
    let base = serve("v9.9.9", archive, sum);
    let bin_dir = d.path().join("bin");
    fs::create_dir(&bin_dir).unwrap();
    let rc = installed_copy(&bin_dir);
    let before = fs::read(&rc).unwrap();

    let out = run_update(&rc, &base, &[]);
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("--version check"), "{}", text(&out.stderr));
    assert_eq!(fs::read(&rc).unwrap(), before);
}

#[test]
fn an_unreachable_server_is_a_clean_error() {
    let d = tempdir().unwrap();
    let rc = installed_copy(d.path());
    let out = run_update(&rc, "http://127.0.0.1:1", &[]);
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("could not reach"), "{}", text(&out.stderr));
}

fn exe_name() -> &'static str {
    if cfg!(windows) { "rc.exe" } else { "rc" }
}

/// Pack `file` into this platform's release archive format (`.zip` with the system `tar` on Windows, `.tar.gz`
/// elsewhere); returns the archive and its checksum line.
fn archive_of_file(dir: &Path, file: &Path) -> (Vec<u8>, String) {
    let name = if cfg!(windows) { "release.zip" } else { "release.tar.gz" };
    let out = dir.join(name);
    let tar = if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        PathBuf::from(root).join("System32").join("tar.exe")
    } else {
        PathBuf::from("tar")
    };
    let mut c = Command::new(tar);
    // `-a` picks the format from the `.zip` name.
    c.arg(if cfg!(windows) { "-acf" } else { "-czf" });
    c.arg(&out).arg("-C").arg(file.parent().unwrap()).arg(file.file_name().unwrap());
    assert!(c.status().unwrap().success(), "could not build {name}");
    let data = fs::read(&out).unwrap();
    let sum = format!("{}  {name}\n", rc_core::util::sha256::hex(&rc_core::util::sha256::sha256(&data)));
    (data, sum)
}

/// The path the whole update is for, on every platform: a real executable, replaced while it runs.
#[test]
fn a_real_binary_is_swapped_in_and_keeps_working() {
    let d = tempdir().unwrap();
    let payload_dir = d.path().join("payload");
    fs::create_dir(&payload_dir).unwrap();
    let payload = payload_dir.join(exe_name());
    fs::copy(assert_cmd::cargo::cargo_bin("rc"), &payload).unwrap();
    let (archive, sum) = archive_of_file(d.path(), &payload);
    let base = serve("v9.9.9", archive, sum);

    let bin_dir = d.path().join("bin");
    fs::create_dir(&bin_dir).unwrap();
    let rc = installed_copy(&bin_dir);
    #[cfg(unix)]
    let before = std::os::unix::fs::MetadataExt::ino(&fs::metadata(&rc).unwrap());

    let out = run_update(&rc, &base, &[]);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains(&format!("Updating rc {CURRENT} -> 9.9.9")), "{stdout}");
    assert!(stdout.contains("Checksum verified."), "{stdout}");

    assert_eq!(fs::read(&rc).unwrap(), fs::read(&payload).unwrap(), "the executable is the archive's");
    #[cfg(unix)]
    assert_ne!(std::os::unix::fs::MetadataExt::ino(&fs::metadata(&rc).unwrap()), before, "replaced, not rewritten");
    // Windows cannot overwrite a running executable: the old one is renamed aside instead.
    #[cfg(windows)]
    assert!(bin_dir.join("rc.exe.old").exists(), "the previous executable should be parked as rc.exe.old");
    let version = Command::new(&rc).arg("--version").output().unwrap();
    assert!(version.status.success(), "the updated binary must run");
    assert!(leftovers(&bin_dir).is_empty(), "{:?}", leftovers(&bin_dir));

    // Updating again works over the parked file and leaves nothing behind either.
    let again = run_update(&rc, &base, &[]);
    assert!(again.status.success(), "{}\n{}", text(&again.stdout), text(&again.stderr));
    assert!(leftovers(&bin_dir).is_empty());
}
