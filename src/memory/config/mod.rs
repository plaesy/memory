//! Configuration management for the memory system.
//!
//! Handles:
//! - Loading configuration from files (.env, config.toml, etc.)
//! - Validating configuration
//! - Providing sensible defaults
//! - Exposing settings to all subsystems
//!
//! # Configuration sources (in order of precedence)
//!
//! 1. Environment variables (MEMORY_*)
//! 2. Configuration file (.plaesy/memory.toml or ~/.hermes/memory.toml)
//! 3. Default values
//!
//! # Example
//!
//! ```ignore
//! use plaesy::memory::config::MemoryConfig;
//!
//! let config = MemoryConfig::load().await?;
//! println!("Watch dir: {}", config.watch_dir);
//! println!("DB path: {}", config.database.db_path.display());
//! ```

use crate::memory::embeddings::EmbeddingConfig;
use crate::memory::error::Result;
use crate::memory::storage::DatabaseConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod mcp;

/// Top-level configuration for the memory system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Directory to watch for changes (typically .plaesy/).
    pub watch_dir: PathBuf,

    /// Database configuration.
    pub database: DatabaseConfig,

    /// Embedding engine configuration.
    pub embeddings: EmbeddingConfig,

    /// Search configuration.
    pub search: SearchConfig,

    /// Logging configuration.
    pub logging: LoggingConfig,

    /// Scheduled jobs (`memory schedule`). Empty means the default: `index` every hour.
    #[serde(default)]
    pub schedule: ScheduleConfig,
}

/// Scheduled jobs. Each entry is validated by `memory::schedule::jobs_from_config`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScheduleConfig {
    /// Jobs in the order they run.
    #[serde(default)]
    pub jobs: Vec<JobSpec>,
}

/// One scheduled job as written in the config: `every` is `30m`, `6h` or `1d`; `action` is
/// `index`, `hygiene` or `missing`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobSpec {
    /// Unique name; also the key of the job's last-run record.
    pub name: String,
    /// Interval between runs.
    pub every: String,
    /// What the job does.
    pub action: String,
}

/// Search subsystem configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default number of results to return.
    pub default_limit: usize,

    /// Maximum number of results.
    pub max_limit: usize,

    /// Timeout for search operations (seconds).
    pub timeout_secs: u64,

    /// Enable vector search (can be disabled for keyword-only mode).
    pub enable_vector_search: bool,

    /// Enable full-text search.
    pub enable_keyword_search: bool,

    /// Search result confidence threshold (0-1).
    pub confidence_threshold: f32,

    /// Days after `last_verified` at which a source counts as stale (0 disables staleness).
    #[serde(default = "default_stale_after_days")]
    pub stale_after_days: u32,

    /// Score multiplier applied to results from stale sources (0-1).
    #[serde(default = "default_stale_penalty")]
    pub stale_penalty: f32,

    /// Rerank the top hybrid results with a cross-encoder (needs its ONNX model; off by default).
    #[serde(default)]
    pub rerank: bool,

    /// How many fused candidates the cross-encoder rescoring covers.
    #[serde(default = "default_rerank_top_n")]
    pub rerank_top_n: usize,

    /// Weight of the cross-encoder score in the final score (0-1); the rest is the fused score.
    #[serde(default = "default_rerank_weight")]
    pub rerank_weight: f32,

    /// Folder paths never indexed, matched anywhere in the tree (e.g. `.plaesy/templates`
    /// excludes every `.plaesy/templates` folder under the watch root). Empty = index everything.
    #[serde(default)]
    pub exclude_dirs: Vec<String>,

    /// Directory with the cross-encoder `model.onnx` + `vocab.txt`
    /// (else `$PLAESY_RERANK_MODEL_DIR`, else `~/.plaesy-models/ms-marco-MiniLM-L-6-v2`).
    #[serde(default)]
    pub rerank_model_dir: Option<std::path::PathBuf>,
}

fn default_rerank_top_n() -> usize {
    30
}

fn default_rerank_weight() -> f32 {
    0.5
}

fn default_stale_after_days() -> u32 {
    180
}

fn default_stale_penalty() -> f32 {
    0.85
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            default_limit: 10,
            max_limit: 100,
            timeout_secs: 30,
            enable_vector_search: true,
            enable_keyword_search: true,
            confidence_threshold: 0.0,
            stale_after_days: default_stale_after_days(),
            stale_penalty: default_stale_penalty(),
            rerank: false,
            rerank_top_n: default_rerank_top_n(),
            rerank_weight: default_rerank_weight(),
            exclude_dirs: Vec::new(),
            rerank_model_dir: None,
        }
    }
}

/// Logging configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (error, warn, info, debug, trace).
    pub level: String,

    /// Enable structured JSON logging.
    pub json_format: bool,

    /// Log file path (optional).
    pub log_file: Option<PathBuf>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            level: "info".to_string(),
            json_format: false,
            log_file: None,
        }
    }
}

/// Default database file: `<data root>/memory.db` (see [`crate::memory::paths`]).
///
/// Falls back to `./memory.db` only when the platform has no data directory.
fn default_db_path() -> PathBuf {
    crate::memory::paths::data_root()
        .map(|d| d.join("memory.db"))
        .unwrap_or_else(|| PathBuf::from("memory.db"))
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            watch_dir: PathBuf::from(".plaesy"),
            database: DatabaseConfig {
                db_path: default_db_path(),
                ..DatabaseConfig::default()
            },
            embeddings: EmbeddingConfig::default(),
            search: SearchConfig::default(),
            logging: LoggingConfig::default(),
            schedule: ScheduleConfig::default(),
        }
    }
}

impl MemoryConfig {
    /// Load configuration from environment, config files, and defaults.
    ///
    /// Configuration is loaded in this order (later sources override):
    /// 1. Defaults (hardcoded)
    /// 2. `<root>/Plaesy/settings.json` written by `memory onboard` (see [`crate::memory::settings`])
    /// 3. ~/.hermes/memory.toml (user config)
    /// 4. .plaesy/memory.toml (project config)
    /// 5. Environment variables (MEMORY_*)
    ///
    /// # Errors
    ///
    /// - `ConfigError` if config file is invalid or unreadable
    /// - `ConfigError` if environment variable has invalid value
    pub async fn load() -> Result<Self> {
        let mut base = MemoryConfig::default();
        if let Some(path) = crate::memory::settings::default_path() {
            if let Some(settings) = crate::memory::settings::load(&path)? {
                settings.apply_to(&mut base);
            }
        }
        let mut value = toml::Value::try_from(base)
            .map_err(|e| crate::memory::error::MemoryError::ConfigError(e.to_string()))?;
        let mut paths = Vec::new();
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            paths.push(PathBuf::from(home).join(".hermes").join("memory.toml"));
        }
        paths.push(PathBuf::from(".plaesy").join("memory.toml"));
        for p in paths.iter().filter(|p| p.is_file()) {
            merge_toml(&mut value, read_toml(p)?);
        }
        let mut cfg = Self::from_toml(value)?;
        cfg.apply_env(|k| std::env::var(k).ok())?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load configuration from a specific file, merged over the defaults.
    ///
    /// Only the keys present in the file are overridden. The result is validated.
    ///
    /// # Errors
    ///
    /// - `ConfigError` if the file is unreadable, invalid TOML, has mistyped
    ///   values, or fails validation (unknown keys are ignored)
    pub async fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let mut value = Self::defaults_as_toml()?;
        merge_toml(&mut value, read_toml(path)?);
        let cfg = Self::from_toml(value)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn defaults_as_toml() -> Result<toml::Value> {
        toml::Value::try_from(MemoryConfig::default())
            .map_err(|e| crate::memory::error::MemoryError::ConfigError(e.to_string()))
    }

    fn from_toml(value: toml::Value) -> Result<Self> {
        value
            .try_into()
            .map_err(|e| crate::memory::error::MemoryError::ConfigError(e.to_string()))
    }

    /// Apply `MEMORY_*` overrides using `get` to look up variables.
    ///
    /// Supported: `MEMORY_WATCH_DIR`, `MEMORY_DB_PATH`, `MEMORY_MODEL_DIR`,
    /// `MEMORY_SEARCH_DEFAULT_LIMIT`, `MEMORY_SEARCH_MAX_LIMIT`,
    /// `MEMORY_SEARCH_TIMEOUT_SECS`, `MEMORY_ENABLE_VECTOR_SEARCH`,
    /// `MEMORY_ENABLE_KEYWORD_SEARCH`, `MEMORY_LOG_LEVEL`, `MEMORY_LOG_JSON`,
    /// `MEMORY_LOG_FILE`.
    ///
    /// # Errors
    ///
    /// - `ConfigError` if a variable is set to an unparsable value
    pub fn apply_env(&mut self, get: impl Fn(&str) -> Option<String>) -> Result<()> {
        use crate::memory::error::MemoryError::ConfigError;
        fn parse<T: std::str::FromStr>(key: &str, v: &str) -> Result<T> {
            v.trim()
                .parse()
                .map_err(|_| ConfigError(format!("invalid value '{v}' for {key}")))
        }
        if let Some(v) = get("MEMORY_WATCH_DIR") {
            self.watch_dir = PathBuf::from(v);
        }
        if let Some(v) = get("MEMORY_DB_PATH") {
            self.database.db_path = PathBuf::from(v);
        }
        if let Some(v) = get("MEMORY_MODEL_DIR") {
            self.embeddings.model_dir = Some(PathBuf::from(v));
        }
        if let Some(v) = get("MEMORY_MODEL_NAME") {
            self.embeddings.model_name = v.trim().to_string();
        }
        if let Some(v) = get("MEMORY_SEARCH_DEFAULT_LIMIT") {
            self.search.default_limit = parse("MEMORY_SEARCH_DEFAULT_LIMIT", &v)?;
        }
        if let Some(v) = get("MEMORY_SEARCH_MAX_LIMIT") {
            self.search.max_limit = parse("MEMORY_SEARCH_MAX_LIMIT", &v)?;
        }
        if let Some(v) = get("MEMORY_SEARCH_TIMEOUT_SECS") {
            self.search.timeout_secs = parse("MEMORY_SEARCH_TIMEOUT_SECS", &v)?;
        }
        if let Some(v) = get("MEMORY_EXCLUDE_DIRS") {
            self.search.exclude_dirs = v
                .split(',')
                .map(|d| d.trim().trim_matches('/').to_string())
                .filter(|d| !d.is_empty())
                .collect();
        }
        if let Some(v) = get("MEMORY_ENABLE_VECTOR_SEARCH") {
            self.search.enable_vector_search = parse("MEMORY_ENABLE_VECTOR_SEARCH", &v)?;
        }
        if let Some(v) = get("MEMORY_ENABLE_KEYWORD_SEARCH") {
            self.search.enable_keyword_search = parse("MEMORY_ENABLE_KEYWORD_SEARCH", &v)?;
        }
        if let Some(v) = get("MEMORY_LOG_LEVEL") {
            self.logging.level = v;
        }
        if let Some(v) = get("MEMORY_LOG_JSON") {
            self.logging.json_format = parse("MEMORY_LOG_JSON", &v)?;
        }
        if let Some(v) = get("MEMORY_LOG_FILE") {
            self.logging.log_file = Some(PathBuf::from(v));
        }
        Ok(())
    }

    /// Validate configuration (check required fields, ranges, dependencies).
    pub fn validate(&self) -> Result<()> {
        // Scheduled jobs: unknown actions, bad intervals and duplicate names fail at load.
        crate::memory::schedule::jobs_from_config(&self.schedule)?;
        // Validate search config
        if self.search.default_limit > self.search.max_limit {
            return Err(crate::memory::error::MemoryError::ConfigError(
                "default_limit cannot exceed max_limit".to_string(),
            ));
        }

        if self.search.confidence_threshold < 0.0 || self.search.confidence_threshold > 1.0 {
            return Err(crate::memory::error::MemoryError::ConfigError(
                "confidence_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate embedding config
        if self.embeddings.dimension == 0 {
            return Err(crate::memory::error::MemoryError::ConfigError(
                "embedding dimension must be > 0".to_string(),
            ));
        }

        if self.embeddings.max_batch_size == 0 {
            return Err(crate::memory::error::MemoryError::ConfigError(
                "max_batch_size must be > 0".to_string(),
            ));
        }

        // Validate watch directory
        if !self.watch_dir.to_string_lossy().is_empty() {
            // Directory existence check would happen at runtime
        }

        Ok(())
    }

    /// Get configuration as a formatted string (for diagnostics).
    pub fn to_diagnostic_string(&self) -> String {
        format!(
            "Memory Config:\n  Watch Dir: {}\n  DB Path: {}\n  Embedding Model: {}\n  Search Timeout: {}s",
            self.watch_dir.display(),
            self.database.db_path.display(),
            self.embeddings.model_name,
            self.search.timeout_secs
        )
    }
}

fn read_toml(path: &std::path::Path) -> Result<toml::Value> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        crate::memory::error::MemoryError::ConfigError(format!(
            "cannot read {}: {e}",
            path.display()
        ))
    })?;
    raw.parse::<toml::Value>().map_err(|e| {
        crate::memory::error::MemoryError::ConfigError(format!(
            "invalid TOML in {}: {e}",
            path.display()
        ))
    })
}

/// Recursively merge `over` into `base` (tables merge key-by-key, other values replace).
fn merge_toml(base: &mut toml::Value, over: toml::Value) {
    match (base, over) {
        (toml::Value::Table(b), toml::Value::Table(o)) => {
            for (k, v) in o {
                match b.get_mut(&k) {
                    Some(existing) => merge_toml(existing, v),
                    None => {
                        b.insert(k, v);
                    }
                }
            }
        }
        (slot, v) => *slot = v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_from_file_merges_over_defaults() {
        let dir = tempfile::tempdir().expect("tmp");
        let p = dir.path().join("memory.toml");
        std::fs::write(
            &p,
            "[search]\ndefault_limit = 5\n\n[embeddings]\nmodel_dir = \"/models/x\"\n",
        )
        .expect("write");
        let cfg = MemoryConfig::load_from_file(&p).await.expect("load");
        assert_eq!(cfg.search.default_limit, 5);
        assert_eq!(cfg.search.max_limit, 100); // untouched default
        assert_eq!(cfg.embeddings.model_dir, Some(PathBuf::from("/models/x")));
    }

    #[tokio::test]
    async fn test_load_from_file_errors() {
        let dir = tempfile::tempdir().expect("tmp");
        let missing = dir.path().join("nope.toml");
        assert!(MemoryConfig::load_from_file(&missing).await.is_err());
        let bad = dir.path().join("bad.toml");
        std::fs::write(&bad, "[search\n").expect("write");
        assert!(MemoryConfig::load_from_file(&bad).await.is_err());
        let wrong_type = dir.path().join("t.toml");
        std::fs::write(&wrong_type, "[search]\ndefault_limit = \"ten\"\n").expect("write");
        assert!(MemoryConfig::load_from_file(&wrong_type).await.is_err());
        let invalid = dir.path().join("v.toml");
        std::fs::write(&invalid, "[search]\ndefault_limit = 500\n").expect("write");
        assert!(MemoryConfig::load_from_file(&invalid).await.is_err());
    }

    #[test]
    fn test_apply_env_overrides_and_rejects_garbage() {
        let mut cfg = MemoryConfig::default();
        let env: std::collections::HashMap<&str, &str> = [
            ("MEMORY_SEARCH_MAX_LIMIT", "50"),
            ("MEMORY_LOG_JSON", "true"),
            ("MEMORY_DB_PATH", "x/state.db"),
        ]
        .into_iter()
        .collect();
        cfg.apply_env(|k| env.get(k).map(|s| s.to_string()))
            .expect("env");
        assert_eq!(cfg.search.max_limit, 50);
        assert!(cfg.logging.json_format);
        assert_eq!(cfg.database.db_path, PathBuf::from("x/state.db"));

        let mut cfg = MemoryConfig::default();
        let r = cfg.apply_env(|k| (k == "MEMORY_SEARCH_TIMEOUT_SECS").then(|| "soon".to_string()));
        assert!(r.is_err());
    }

    #[test]
    fn test_default_config() {
        let config = MemoryConfig::default();
        assert_eq!(config.watch_dir, PathBuf::from(".plaesy"));
        assert_eq!(config.search.default_limit, 10);
    }

    #[test]
    fn test_validate_config_valid() {
        let config = MemoryConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_config_invalid_limits() {
        let mut config = MemoryConfig::default();
        config.search.default_limit = 200;
        config.search.max_limit = 100;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_config_invalid_confidence() {
        let mut config = MemoryConfig::default();
        config.search.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_config_invalid_dimension() {
        let mut config = MemoryConfig::default();
        config.embeddings.dimension = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_diagnostic_string() {
        let config = MemoryConfig::default();
        let diag = config.to_diagnostic_string();
        assert!(diag.contains("Watch Dir"));
        assert!(diag.contains("DB Path"));
        assert!(diag.contains("Memory Config"));
    }
}
