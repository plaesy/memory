//! Real crash/power-loss simulation: spawn the actual `memory` CLI binary as an OS
//! process, SIGKILL it mid-write, and verify the vault is left in a safe, consistent state —
//! no half-written note ever becomes visible, and the orphaned temp file left behind by the
//! kill is reclaimed on the next `index_directory` (startup reconciliation), not left to
//! accumulate forever.
//!
//! This exercises the real binary under `target/debug/memory`, not just the library's
//! `write_note` function in-process, so it also covers process exit/signal handling that a
//! unit test calling the function directly cannot.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

fn memory_bin() -> PathBuf {
    // Built by `cargo test`'s own build step (the bin target is part of this crate).
    let mut path = std::env::current_exe().expect("current exe");
    path.pop(); // drop test binary file name
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("memory");
    assert!(
        path.exists(),
        "expected memory binary at {path:?}; run `cargo build --bin memory` first"
    );
    path
}

fn write_minimal_config(dir: &Path, watch_dir: &Path, db: &Path) -> PathBuf {
    let cfg = dir.join("memory.toml");
    std::fs::write(
        &cfg,
        format!(
            r#"
watch_dir = {watch_dir:?}

[database]
path = {db:?}

[embeddings]
backend = "hashing"
dimension = 64

[search]
default_limit = 10
max_limit = 100
timeout_secs = 5
enable_vector_search = true
enable_keyword_search = true

[logging]
level = "error"
"#,
        ),
    )
    .expect("write config");
    cfg
}

/// Recursively find every leftover note-publication temp file (dot-prefixed, containing
/// `.tmp-`) under `root`. The note lands in the vault's default inbox subfolder (no `--folder`
/// is passed), not directly in `root`, so this must walk the tree rather than just `read_dir`.
fn find_tmp_files(root: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with('.') && name.contains(".tmp-")
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn find_md_files(root: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
        .map(|e| e.path().to_path_buf())
        .collect()
}

/// Kill the `memory remember` process mid-write (the write path is slowed by a test-only env
/// var so the publish window is wide enough to hit deterministically) and assert the vault
/// never contains a half-written `.md` note, and that a subsequent `memory index`
/// (`index_directory`, i.e. startup/crash reconciliation) call reclaims the leftover temp file
/// rather than leaving it behind forever.
#[test]
fn killing_remember_mid_write_leaves_no_partial_note_and_is_reconciled() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let watch_dir = tmp.path().join("vault");
    std::fs::create_dir_all(&watch_dir).expect("mkdir vault");
    let db = tmp.path().join("memory.sqlite");
    let cfg = write_minimal_config(tmp.path(), &watch_dir, &db);
    let bin = memory_bin();

    let body = "Crash test note body.\n".to_string();

    let mut child = Command::new(&bin)
        .arg("--config")
        .arg(&cfg)
        .arg("remember")
        .arg("Crash Test Note")
        // Test-only fault injection (see `write_note` in notes/mod.rs): sleeps for 2s between
        // the temp file becoming visible on disk and its publication, widening the race window
        // enough that polling below reliably observes it before the kill.
        .env("PLAESY_MEMORY_TEST_SLOW_PUBLISH", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn memory remember");

    // Feed the body from a second thread so we can poll/kill the process while it is still
    // running, instead of blocking this thread on a `write_all` that might finish first.
    let mut stdin = child.stdin.take().expect("stdin");
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(body.as_bytes());
    });

    // Poll for the temp file to actually appear rather than guessing a fixed sleep.
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut seen_tmp = false;
    while std::time::Instant::now() < deadline {
        if !find_tmp_files(&watch_dir).is_empty() {
            seen_tmp = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        seen_tmp,
        "never observed the temp file appear within the deadline; \
         the write path or polling loop may need adjustment"
    );

    #[cfg(unix)]
    {
        // SAFETY: libc::kill with SIGKILL on our own freshly-spawned child pid; no aliasing,
        // no reentrancy, and the process is reaped via child.wait() immediately after.
        unsafe {
            libc::kill(child.id() as i32, libc::SIGKILL);
        }
    }
    let _ = child.wait();
    let _ = writer.join();

    // The kill landed while the temp file existed but before publish (hard_link + unlink), per
    // the fault-injection sleep above, so exactly one leftover temp file must still be present.
    let leftover_tmp = find_tmp_files(&watch_dir);
    assert!(
        !leftover_tmp.is_empty(),
        "expected the SIGKILL to land mid-write and leave a .tmp- file behind"
    );
    for path in &leftover_tmp {
        // Back-date it so startup reconciliation (which only reaps temp files older than its
        // staleness margin, to avoid racing a concurrent in-flight writer) treats it as
        // abandoned, exactly as it would after any real crash once enough time has passed.
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        filetime::set_file_mtime(path, filetime::FileTime::from_system_time(old))
            .expect("backdate tmp file mtime");
    }
    // SIGKILL also leaves the directory lock behind. Back-date that lock past its stale-lock
    // threshold so the next real CLI invocation exercises stale-lock recovery before it reaches
    // temp-file reconciliation (this is the actual power-loss sequence).
    let lock_path = watch_dir.join(".plaesy-memory-writer.lock");
    assert!(
        lock_path.is_dir(),
        "crashed writer must leave its lock directory"
    );
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
    filetime::set_file_mtime(&lock_path, filetime::FileTime::from_system_time(old))
        .expect("backdate stale lock");

    // 1. No fully-named note must exist with partial content.
    for path in find_md_files(&watch_dir) {
        let content = std::fs::read_to_string(&path).expect("read note");
        assert!(
            content.ends_with('\n') && content.contains("Crash test note body"),
            "a published note must never be truncated/partial: {path:?} ({} bytes)",
            content.len()
        );
    }

    // 2. `memory index` (the real crash-recovery entry point) must actually reclaim the
    //    leftover temp file, proving reconciliation is wired up end-to-end, not just a passing
    //    no-op.
    let index_status = Command::new(&bin)
        .arg("--config")
        .arg(&cfg)
        .arg("--db")
        .arg(&db)
        .arg("index")
        .arg(&watch_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("run memory index");
    assert!(
        index_status.success(),
        "memory index must succeed after a crash, not just leave the index broken"
    );
    for path in &leftover_tmp {
        assert!(
            !path.exists(),
            "startup reconciliation must remove the stale temp file left by the SIGKILL: {path:?}"
        );
    }
}
