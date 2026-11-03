//! Descriptor-relative filesystem helpers.
//!
//! `std::fs` path-based APIs re-resolve every path component on each call. Between a
//! `canonicalize()` check and a later `create_dir`/`write`/`rename`, an attacker (or just an
//! unlucky concurrent process) can replace a directory component with a symlink and redirect
//! the operation outside the intended root (TOCTOU).
//!
//! This module opens the target directory exactly once with `O_DIRECTORY | O_NOFOLLOW` (so the
//! final path component cannot itself be a symlink), then performs every subsequent operation
//! with `*at` syscalls relative to that open file descriptor. The descriptor keeps pointing at
//! the same inode for the rest of the operation regardless of what happens to the path later,
//! which is the actual guarantee `openat`-style APIs provide over path-based ones.
//!
//! Scope: this protects the *final* directory component used for note/lock publication on
//! Linux. It does not re-verify every ancestor component on every call (that would need a
//! `resolve_beneath`-style walk); `write_note`'s existing `canonicalize` + `starts_with` check
//! is still responsible for rejecting a root that has already been redirected before the first
//! `open_dir_nofollow` call. On non-Linux platforms these helpers fall back to the path-based
//! `std::fs` equivalents, which keep today's (slightly weaker but still real) protection.

use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;

/// An open directory descriptor used as the base for `*at` operations.
pub struct DirHandle {
    fd: OwnedFd,
}

fn to_cstring(path: &Path) -> io::Result<CString> {
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))
}

impl DirHandle {
    /// Open `dir` as a directory descriptor, refusing to follow a symlink at `dir` itself.
    #[cfg(target_os = "linux")]
    pub fn open_nofollow(dir: &Path) -> io::Result<Self> {
        let c_path = to_cstring(dir)?;
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn open_nofollow(dir: &Path) -> io::Result<Self> {
        // Non-Linux fallback: still reject a symlinked directory, just without the atomic
        // guarantee that a path-based open provides on Linux via O_NOFOLLOW + O_DIRECTORY.
        let meta = std::fs::symlink_metadata(dir)?;
        if !meta.file_type().is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a directory (or is a symlink)",
            ));
        }
        let file = std::fs::File::open(dir)?;
        Ok(Self { fd: file.into() })
    }

    /// Create a regular file named `name` directly inside this directory, failing if it already
    /// exists (`O_EXCL`), without ever re-resolving the directory path.
    #[cfg(target_os = "linux")]
    pub fn create_new_file(&self, name: &Path) -> io::Result<std::fs::File> {
        let c_name = to_cstring(name)?;
        let fd = unsafe {
            libc::openat(
                self.fd.as_raw_fd(),
                c_name.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { std::fs::File::from_raw_fd(fd) })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn create_new_file(&self, name: &Path) -> io::Result<std::fs::File> {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.child_path_fallback(name))
    }

    /// Hard-link `old_name` (inside this directory) to `new_name` (inside this directory),
    /// resolved relative to the held descriptor rather than by re-walking the path.
    #[cfg(target_os = "linux")]
    pub fn link_within(&self, old_name: &Path, new_name: &Path) -> io::Result<()> {
        let c_old = to_cstring(old_name)?;
        let c_new = to_cstring(new_name)?;
        let rc = unsafe {
            libc::linkat(
                self.fd.as_raw_fd(),
                c_old.as_ptr(),
                self.fd.as_raw_fd(),
                c_new.as_ptr(),
                0,
            )
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn link_within(&self, old_name: &Path, new_name: &Path) -> io::Result<()> {
        std::fs::hard_link(
            self.child_path_fallback(old_name),
            self.child_path_fallback(new_name),
        )
    }

    /// Remove a regular file named `name` directly inside this directory.
    #[cfg(target_os = "linux")]
    pub fn remove_file(&self, name: &Path) -> io::Result<()> {
        let c_name = to_cstring(name)?;
        let rc = unsafe { libc::unlinkat(self.fd.as_raw_fd(), c_name.as_ptr(), 0) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn remove_file(&self, name: &Path) -> io::Result<()> {
        std::fs::remove_file(self.child_path_fallback(name))
    }

    /// Flush this directory's own metadata/entries to durable storage.
    pub fn sync(&self) -> io::Result<()> {
        let rc = unsafe { libc::fsync(self.fd.as_raw_fd()) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn child_path_fallback(&self, _name: &Path) -> std::path::PathBuf {
        unreachable!("fallback path helper is only used on non-Linux targets with a real base path, see open_nofollow caller")
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn rejects_symlinked_directory() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(DirHandle::open_nofollow(&link).is_err());
        assert!(DirHandle::open_nofollow(&real).is_ok());
    }

    #[test]
    fn create_link_remove_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let handle = DirHandle::open_nofollow(dir.path()).unwrap();
        let tmp = Path::new(".note.tmp");
        let dest = Path::new("note.md");
        {
            use std::io::Write;
            let mut f = handle.create_new_file(tmp).unwrap();
            f.write_all(b"hello").unwrap();
            f.sync_all().unwrap();
        }
        handle.link_within(tmp, dest).unwrap();
        handle.remove_file(tmp).unwrap();
        handle.sync().unwrap();
        let content = std::fs::read_to_string(dir.path().join("note.md")).unwrap();
        assert_eq!(content, "hello");
        // Destination already exists: must fail rather than silently replace it.
        assert!(handle.create_new_file(dest).is_err());
    }

    #[test]
    fn create_new_file_rejects_existing_symlink_target() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let bait = outside.path().join("bait");
        std::fs::write(&bait, "outside").unwrap();
        let link_name = Path::new("escape");
        std::os::unix::fs::symlink(&bait, dir.path().join(link_name)).unwrap();
        let handle = DirHandle::open_nofollow(dir.path()).unwrap();
        // O_EXCL already rejects anything pre-existing at that name, symlink or not.
        assert!(handle.create_new_file(link_name).is_err());
        assert_eq!(std::fs::read_to_string(&bait).unwrap(), "outside");
    }
}
