//! User settings written by `memory onboard`: `<root>/Plaesy/settings.json`.
//!
//! The root defaults to the platform data-local directory (`%LOCALAPPDATA%` on Windows,
//! `~/.local/share` on Linux, `~/Library/Application Support` on macOS). Set `PLAESY_ROOT` to use
//! another root. Settings sit below `.plaesy/memory.toml` and `MEMORY_*` variables in the precedence
//! order of [`crate::memory::config::MemoryConfig::load`].

use crate::memory::config::MemoryConfig;
use crate::memory::error::MemoryError;
use crate::memory::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Current settings format version.
pub const SETTINGS_VERSION: u32 = 1;

/// Folder name under the root that holds `settings.json`.
pub const SETTINGS_DIR: &str = "Plaesy";

/// The file written by onboarding. Every field is optional; absent means "use the default".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_version")]
    pub version: u32,
    /// Folder of notes to index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes_dir: Option<PathBuf>,
    /// SQLite database file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub db_path: Option<PathBuf>,
    /// Embedding model folder name (for example `all-MiniLM-L6-v2`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Cross-encoder reranking on or off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank: Option<bool>,
    /// Folder or file names kept out of the index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude_dirs: Option<Vec<String>>,
}

fn default_version() -> u32 {
    SETTINGS_VERSION
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            version: SETTINGS_VERSION,
            notes_dir: None,
            db_path: None,
            model_name: None,
            rerank: None,
            exclude_dirs: None,
        }
    }
}

impl Settings {
    /// Copy the fields that are set onto `cfg`. Unset fields leave `cfg` unchanged.
    pub fn apply_to(&self, cfg: &mut MemoryConfig) {
        if let Some(v) = &self.notes_dir {
            cfg.watch_dir = v.clone();
        }
        if let Some(v) = &self.db_path {
            cfg.database.db_path = v.clone();
        }
        if let Some(v) = &self.model_name {
            cfg.embeddings.model_name = v.clone();
        }
        if let Some(v) = self.rerank {
            cfg.search.rerank = v;
        }
        if let Some(v) = &self.exclude_dirs {
            cfg.search.exclude_dirs = v.clone();
        }
    }
}

/// Settings file location: `<root>/Plaesy/settings.json`, where `<root>` is `PLAESY_ROOT` or the platform default.
pub fn default_path() -> Option<PathBuf> {
    root_from(
        std::env::var_os("PLAESY_ROOT").map(PathBuf::from),
        dirs::data_local_dir(),
    )
    .map(|root| file_in_root(&root))
}

/// Settings file location for an explicit root.
pub fn file_in_root(root: &Path) -> PathBuf {
    root.join(SETTINGS_DIR).join("settings.json")
}

fn root_from(env_root: Option<PathBuf>, platform: Option<PathBuf>) -> Option<PathBuf> {
    env_root.filter(|p| !p.as_os_str().is_empty()).or(platform)
}

/// Read settings from `path`. Returns `Ok(None)` when the file does not exist.
///
/// # Errors
///
/// - `ConfigError` if the file exists but is not valid settings JSON
pub fn load(path: &Path) -> Result<Option<Settings>> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(MemoryError::ConfigError(format!(
                "cannot inspect {}: {error}",
                path.display()
            )))
        }
    };
    if !metadata.file_type().is_file() {
        return Err(MemoryError::ConfigError(format!(
            "settings path is not a regular file: {}",
            path.display()
        )));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| MemoryError::ConfigError(format!("cannot read {}: {e}", path.display())))?;
    let settings: Settings = serde_json::from_str(&text).map_err(|e| {
        MemoryError::ConfigError(format!("invalid settings {}: {e}", path.display()))
    })?;
    Ok(Some(settings))
}

/// Write settings to `path`, creating parent folders. Overwrites an existing file.
///
/// # Errors
///
/// - `ConfigError` if the folder cannot be created or the file cannot be written
pub fn save(path: &Path, settings: &Settings) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            MemoryError::ConfigError(format!("cannot create {}: {e}", parent.display()))
        })?;
    }
    let text = serde_json::to_string_pretty(settings)
        .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            MemoryError::ConfigError(format!("invalid settings path: {}", path.display()))
        })?;
    let tmp = parent.join(format!(".{file_name}.tmp-{}", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        file.write_all((text + "\n").as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)?;
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.map_err(|e| MemoryError::ConfigError(format!("cannot write {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_root_wins_over_platform_default() {
        let got = root_from(
            Some(PathBuf::from("/custom")),
            Some(PathBuf::from("/platform")),
        );
        assert_eq!(got, Some(PathBuf::from("/custom")));
    }

    #[test]
    fn empty_env_root_falls_back_to_platform() {
        let got = root_from(Some(PathBuf::new()), Some(PathBuf::from("/platform")));
        assert_eq!(got, Some(PathBuf::from("/platform")));
    }

    #[test]
    fn file_lives_under_plaesy_folder() {
        let p = file_in_root(Path::new("/r"));
        assert!(p.ends_with(Path::new("Plaesy").join("settings.json")));
    }

    #[test]
    fn round_trip_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = file_in_root(dir.path());
        assert_eq!(load(&path).unwrap(), None);
        let s = Settings {
            notes_dir: Some(PathBuf::from("/notes")),
            rerank: Some(false),
            ..Settings::default()
        };
        save(&path, &s).unwrap();
        assert_eq!(load(&path).unwrap(), Some(s));
    }

    #[test]
    fn unset_fields_are_not_written() {
        let text = serde_json::to_string(&Settings::default()).unwrap();
        assert_eq!(text, format!("{{\"version\":{SETTINGS_VERSION}}}"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_settings_path_is_rejected_on_load_and_replaced_on_save() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("outside.json");
        std::fs::write(&target, "{\"version\":1}\n").unwrap();
        let path = dir.path().join("settings.json");
        symlink(&target, &path).unwrap();

        assert!(load(&path).is_err());
        save(&path, &Settings::default()).unwrap();
        assert!(!std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "{\"version\":1}\n"
        );
        assert_eq!(load(&path).unwrap(), Some(Settings::default()));
    }
}
