#!/usr/bin/env bash
# The checks CI runs, one at a time, in order, stopping at the first failure.
#
#   tools/check.sh              fmt, clippy, build, tests, Windows type-check
#   tools/check.sh --quick      fmt and clippy only (a few minutes, no test run)
#   tools/check.sh --no-windows skip the Windows type-check
#   tools/check.sh --keep-going run every step even after a failure
#   tools/check.sh --help
#
# Heavy jobs are run strictly one after another, and a lock stops a second run from starting while one is in
# progress (two cargo runs on the same target directory only wait for each other, and overlapping them with other
# builds is what makes the machine crawl). Output of a passing step is summarized; a failing step prints its log.
#
# The CI job that builds the fuzz targets (`fuzz/`) is not run here; it needs `cargo-fuzz` and a nightly toolchain.

set -u
cd "$(dirname "$0")/.." || exit 2

quick=0; windows=1; keep_going=0
for arg in "$@"; do
  case "$arg" in
    --quick) quick=1 ;;
    --no-windows) windows=0 ;;
    --keep-going) keep_going=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# --- one run at a time ---------------------------------------------------------------------------------------------
mkdir -p target
lock=target/.check.lock
if ! mkdir "$lock" 2>/dev/null; then
  holder=$(cat "$lock/pid" 2>/dev/null || true)
  if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
    echo "another tools/check.sh is already running (pid $holder); wait for it to finish" >&2
    exit 3
  fi
  rm -rf "$lock" && mkdir "$lock"   # the previous run died without cleaning up
fi
echo $$ > "$lock/pid"
log_dir=$(mktemp -d "${TMPDIR:-/tmp}/rc-check.XXXXXX")
trap 'rm -rf "$lock" "$log_dir"' EXIT
trap 'echo; echo "interrupted"; exit 130' INT TERM

# --- steps ---------------------------------------------------------------------------------------------------------
results=()
failed=0

# run NAME CMD...   runs CMD, records PASS/FAIL with the time taken, and shows the log of a failure
run() {
  local name=$1; shift
  local log="$log_dir/${name// /_}.log"
  printf '%-22s ' "$name"
  local start=$SECONDS
  if "$@" >"$log" 2>&1; then
    local secs=$((SECONDS - start))
    printf 'PASS  (%ds)\n' "$secs"
    results+=("PASS  $name (${secs}s)")
    return 0
  fi
  local secs=$((SECONDS - start))
  printf 'FAIL  (%ds)\n' "$secs"
  results+=("FAIL  $name (${secs}s)")
  failed=1
  echo "----- last 60 lines of the $name log -----"
  tail -n 60 "$log"
  echo "------------------------------------------"
  [ "$keep_going" -eq 1 ] || return 1
  return 0
}

# The test step counts results instead of printing every line.
tests() {
  cargo test --workspace --locked --no-fail-fast 2>&1 | tee "$log_dir/test-full.log" |
    awk '/^test result/ {p += $4; f += $6; next} END {print p " passed, " f " failed"}'
  ! grep -q -E '^test result: FAILED|^error(\[|:)' "$log_dir/test-full.log"
}

summary() {
  echo
  echo "== summary =="
  printf '%s\n' "${results[@]}"
  [ "$failed" -eq 0 ] && echo "all checks passed" || echo "CHECKS FAILED"
}

run "fmt"    cargo fmt --all --check                                        || { summary; exit 1; }
run "clippy" cargo clippy --workspace --all-targets --locked -- -D warnings || { summary; exit 1; }
if [ "$quick" -eq 0 ]; then
  run "build" cargo build --workspace --locked                              || { summary; exit 1; }
  run "tests" tests                                                         || { summary; exit 1; }
  if [ "$windows" -eq 1 ]; then
    if rustup target list --installed 2>/dev/null | grep -q '^x86_64-pc-windows-msvc$'; then
      run "windows type-check" cargo check --workspace --all-targets --locked --target x86_64-pc-windows-msvc \
        || { summary; exit 1; }
    else
      echo "windows type-check     SKIP  (install it with: rustup target add x86_64-pc-windows-msvc)"
      results+=("SKIP  windows type-check (target not installed)")
    fi
  fi
fi

summary
exit "$failed"
