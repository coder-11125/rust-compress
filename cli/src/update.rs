//! `rc update`: find the newest GitHub release, show `current -> newer`, and replace the running binary.
//!
//! There is no HTTP client in the tree, so downloads go through `curl` (as `install.sh` does) and archives are
//! unpacked with `tar` (which also reads the Windows `.zip` assets). The newest tag comes from the redirect of
//! `/releases/latest`, so no API token or rate limit is involved. The archive's SHA-256 is checked here, and the
//! new binary has to run before it replaces the old one.

use clap::Args;
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

const DEFAULT_REPO_URL: &str = "https://github.com/coder-11125/rust-compress";
const CURRENT: &str = env!("CARGO_PKG_VERSION");

#[derive(Args, Debug)]
pub struct UpdateArgs {
    /// Only report whether a newer release exists; do not download anything.
    #[arg(long)]
    pub check: bool,

    /// Install this release tag (e.g. v0.2.0) instead of the newest one; older tags are allowed.
    #[arg(long, value_name = "TAG")]
    pub to: Option<String>,

    /// Reinstall even when the installed version is already the newest.
    #[arg(long)]
    pub force: bool,
}

pub fn run(args: &UpdateArgs) -> ExitCode {
    match update(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn update(args: &UpdateArgs) -> Result<(), String> {
    // Overridable for mirrors and for the tests; releases live at `<base>/releases/...`.
    let base = std::env::var("RC_UPDATE_BASE_URL").unwrap_or_else(|_| DEFAULT_REPO_URL.to_string());
    let base = base.trim_end_matches('/');
    let target = target_triple()?;

    let tag = match &args.to {
        Some(t) => {
            if t.is_empty() || t.contains(['/', '\\', ' ']) || t.starts_with('-') {
                return Err(format!("invalid release tag {t:?}"));
            }
            t.clone()
        }
        None => latest_tag(base)?,
    };
    let current = Version::parse(CURRENT);
    let newer = Version::parse(&tag);
    println!("Installed version: {CURRENT}");
    println!("Release version:   {}", tag.trim_start_matches('v'));

    let order = match (&current, &newer) {
        (Some(c), Some(n)) => Some(n.cmp(c)),
        _ => None,
    };
    let explicit = args.to.is_some();
    match order {
        Some(Ordering::Greater) => {}
        Some(Ordering::Equal) if !args.force => {
            println!("rc {CURRENT} is up to date.");
            return Ok(());
        }
        Some(Ordering::Less) if !explicit && !args.force => {
            println!("rc {CURRENT} is newer than the latest release; nothing to do.");
            return Ok(());
        }
        None if !explicit => {
            return Err(format!("could not compare {CURRENT} with release tag {tag:?} (use --to {tag} to install it anyway)"));
        }
        _ => {}
    }
    let verb = match order {
        Some(Ordering::Greater) => "Updating",
        Some(Ordering::Less) => "Downgrading",
        _ => "Reinstalling",
    };
    if args.check {
        if order == Some(Ordering::Greater) {
            println!("A newer release is available: {CURRENT} -> {}. Run `rc update` to install it.", tag.trim_start_matches('v'));
        }
        return Ok(());
    }
    println!("{verb} rc {CURRENT} -> {}", tag.trim_start_matches('v'));

    let exe = std::env::current_exe()
        .and_then(fs::canonicalize)
        .map_err(|e| format!("could not locate the running executable: {e}"))?;
    let ext = if cfg!(windows) { "zip" } else { "tar.gz" };
    let stem = format!("rust-compress-{tag}-{target}");
    let asset = format!("{stem}.{ext}");
    let url = format!("{base}/releases/download/{tag}/{asset}");

    let work = Scratch::new(exe.parent().unwrap_or(Path::new(".")))?;
    let archive = work.path.join(&asset);
    println!("Downloading {asset} ...");
    curl_download(&url, &archive).map_err(|e| format!("download failed ({url}): {e}"))?;

    let sum_file = work.path.join(format!("{asset}.sha256"));
    match curl_download(&format!("{url}.sha256"), &sum_file) {
        Ok(()) => {
            let text = fs::read_to_string(&sum_file).map_err(|e| e.to_string())?;
            let expected = text.split_whitespace().next().unwrap_or("").to_ascii_lowercase();
            let data = fs::read(&archive).map_err(|e| e.to_string())?;
            let actual = hex(&sha256(&data));
            if expected != actual {
                return Err(format!("checksum mismatch for {asset}: expected {expected}, got {actual}"));
            }
            println!("Checksum verified.");
        }
        Err(_) => eprintln!("warning: no checksum published for {asset}; skipping verification"),
    }

    let unpack = work.path.join("unpacked");
    fs::create_dir(&unpack).map_err(|e| e.to_string())?;
    run_tool(Command::new("tar").arg(if cfg!(windows) { "-xf" } else { "-xzf" }).arg(&archive).arg("-C").arg(&unpack))
        .map_err(|e| format!("could not unpack {asset}: {e}"))?;
    let bin_name = if cfg!(windows) { "rc.exe" } else { "rc" };
    let new_bin = [unpack.join(&stem).join(bin_name), unpack.join(bin_name)]
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| format!("{asset} did not contain {bin_name}"))?;

    // Stage next to the target so the final rename stays on one filesystem.
    let staged = exe.with_file_name(format!(".rc-update-{}{}", std::process::id(), if cfg!(windows) { ".exe" } else { "" }));
    let result = stage_and_swap(&new_bin, &staged, &exe, &tag);
    if result.is_err() {
        let _ = fs::remove_file(&staged);
    }
    result?;
    println!("Updated {} to {}.", exe.display(), tag.trim_start_matches('v'));
    Ok(())
}

fn stage_and_swap(new_bin: &Path, staged: &Path, exe: &Path, tag: &str) -> Result<(), String> {
    let denied = |e: std::io::Error| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            format!("no permission to write {} ({e}); re-run with sufficient privileges", exe.display())
        } else {
            format!("could not write {}: {e}", staged.display())
        }
    };
    fs::copy(new_bin, staged).map_err(denied)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(staged, fs::Permissions::from_mode(0o755)).map_err(denied)?;
    }
    // The new binary must run before it replaces a working one.
    let out = Command::new(staged)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("the downloaded binary does not run on this system: {e}"))?;
    if !out.status.success() {
        return Err("the downloaded binary failed its --version check; keeping the installed one".into());
    }
    let reported = String::from_utf8_lossy(&out.stdout);
    if !reported.contains(tag.trim_start_matches('v')) {
        eprintln!("warning: release {tag} reports itself as {:?}", reported.trim());
    }
    swap_in(staged, exe).map_err(denied)
}

#[cfg(unix)]
fn swap_in(staged: &Path, exe: &Path) -> std::io::Result<()> {
    fs::rename(staged, exe)
}

/// A running `.exe` cannot be overwritten on Windows but can be renamed out of the way.
#[cfg(windows)]
fn swap_in(staged: &Path, exe: &Path) -> std::io::Result<()> {
    let old = exe.with_extension("exe.old");
    let _ = fs::remove_file(&old);
    fs::rename(exe, &old)?;
    if let Err(e) = fs::rename(staged, exe) {
        let _ = fs::rename(&old, exe);
        return Err(e);
    }
    Ok(())
}

/// The release asset suffix for this machine, matching `.github/workflows/release.yml`.
fn target_triple() -> Result<String, String> {
    let arch = match std::env::consts::ARCH {
        a @ ("x86_64" | "aarch64") => a,
        a => return Err(format!("no prebuilt releases for architecture {a}")),
    };
    let os = match std::env::consts::OS {
        "linux" => "unknown-linux-gnu",
        "macos" => "apple-darwin",
        "windows" if arch == "x86_64" => "pc-windows-msvc",
        os => return Err(format!("no prebuilt releases for {os}/{arch}")),
    };
    Ok(format!("{arch}-{os}"))
}

/// The tag `/releases/latest` redirects to (GitHub skips drafts and prereleases there).
fn latest_tag(base: &str) -> Result<String, String> {
    let null = if cfg!(windows) { "NUL" } else { "/dev/null" };
    let out = Command::new("curl")
        .args(["-fsSIL", "--retry", "3", "--connect-timeout", "20", "--max-time", "60", "-o", null, "-w", "%{url_effective}"])
        .arg(format!("{base}/releases/latest"))
        .stdin(Stdio::null())
        .output()
        .map_err(curl_missing)?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr);
        return Err(format!("could not reach {base} to find the latest release: {}", why.trim()));
    }
    let url = String::from_utf8_lossy(&out.stdout);
    let tag = url.trim().rsplit('/').next().unwrap_or("");
    if tag.is_empty() || tag == "latest" || tag == "releases" {
        return Err("the repository has no published release yet".into());
    }
    Ok(tag.to_string())
}

fn curl_download(url: &str, dest: &Path) -> Result<(), String> {
    run_tool(
        Command::new("curl")
            .args(["-fsSL", "--retry", "3", "--retry-delay", "2", "--connect-timeout", "20", "--max-time", "600", "-o"])
            .arg(dest)
            .arg(url),
    )
}

fn curl_missing(e: std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::NotFound {
        "`curl` is required for updates but was not found on PATH".into()
    } else {
        format!("could not run curl: {e}")
    }
}

fn run_tool(cmd: &mut Command) -> Result<(), String> {
    let out = cmd.stdin(Stdio::null()).output().map_err(curl_missing)?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// A scratch directory next to the executable, removed when dropped.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(near: &Path) -> Result<Self, String> {
        let path = near.join(format!(".rc-update-dir-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                format!("no permission to write in {} ({e}); re-run with sufficient privileges", near.display())
            } else {
                format!("could not create a scratch directory in {}: {e}", near.display())
            }
        })?;
        Ok(Scratch { path })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------------------------------------------

/// A semantic version: `[v]MAJOR.MINOR.PATCH[-prerelease][+build]`.
#[derive(Debug, PartialEq, Eq)]
struct Version {
    core: [u64; 3],
    pre: Vec<String>,
}

impl Version {
    fn parse(s: &str) -> Option<Version> {
        let s = s.strip_prefix('v').unwrap_or(s);
        let s = s.split('+').next()?;
        let (core, pre) = match s.split_once('-') {
            Some((c, p)) => (c, p.split('.').map(str::to_string).collect()),
            None => (s, Vec::new()),
        };
        let mut it = core.split('.');
        let mut nums = [0u64; 3];
        for n in &mut nums {
            *n = it.next()?.parse().ok()?;
        }
        if it.next().is_some() || pre.iter().any(|p: &String| p.is_empty()) {
            return None;
        }
        Some(Version { core: nums, pre })
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        self.core.cmp(&other.core).then_with(|| match (self.pre.is_empty(), other.pre.is_empty()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => {
                for (a, b) in self.pre.iter().zip(&other.pre) {
                    let o = match (a.parse::<u64>(), b.parse::<u64>()) {
                        (Ok(x), Ok(y)) => x.cmp(&y),
                        (Ok(_), Err(_)) => Ordering::Less,
                        (Err(_), Ok(_)) => Ordering::Greater,
                        (Err(_), Err(_)) => a.cmp(b),
                    };
                    if o != Ordering::Equal {
                        return o;
                    }
                }
                self.pre.len().cmp(&other.pre.len())
            }
        })
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------------------------------------------

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        // Two blocks: the padding spills into a second block.
        assert_eq!(
            hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex(&sha256(&vec![b'a'; 1_000_000])),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn versions_order_like_semver() {
        let v = |s: &str| Version::parse(s).unwrap();
        assert!(v("v0.2.0") > v("0.1.9"));
        assert!(v("1.0.0") > v("0.99.99"));
        assert!(v("0.10.0") > v("0.9.0"), "numeric, not lexical");
        assert!(v("1.0.0") > v("1.0.0-rc.1"));
        assert!(v("1.0.0-rc.2") > v("1.0.0-rc.1"));
        assert!(v("1.0.0-rc.10") > v("1.0.0-rc.9"));
        assert!(v("1.0.0-rc.1") > v("1.0.0-beta"));
        assert!(v("1.0.0-alpha.1") > v("1.0.0-alpha"));
        assert_eq!(v("v1.2.3+build5").cmp(&v("1.2.3")), Ordering::Equal);
        for bad in ["", "v", "1.2", "1.2.3.4", "a.b.c", "1.2.3-", "latest"] {
            assert!(Version::parse(bad).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn the_running_version_parses() {
        assert!(Version::parse(CURRENT).is_some());
    }
}
