//! Scheduled memory jobs (`memory schedule once|run`).
//!
//! A job runs one existing engine method (`index`, `hygiene` or `missing`) on the vault at a fixed
//! interval. Last-run times live in the `schedule_runs` table of the same database, so no second
//! store exists. Each run appends one line to `<root>\Plaesy\schedule.log`. A failing job is
//! recorded and the next job still runs.
//!
//! Overlap: a job's `last_run` is written before it starts. A second `once` that begins meanwhile
//! sees the job as not due, so Task Scheduler firing during a long index does not run it twice.
//! If the process dies mid-run the job waits one full interval, then runs again.

use crate::memory::config::ScheduleConfig;
use crate::memory::error::{MemoryError, Result};
use crate::memory::hygiene::HygieneOptions;
use crate::memory::search::SearchEngine;
use crate::memory::settings;
use chrono::{DateTime, Duration, Utc};
use rusqlite::{params, OptionalExtension};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Longest sleep in `run` mode, so edits to the config are picked up within a minute.
const MAX_SLEEP: std::time::Duration = std::time::Duration::from_secs(60);

const STATE_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS schedule_runs (
    job      TEXT PRIMARY KEY,
    last_run TEXT NOT NULL,
    outcome  TEXT NOT NULL
);";

/// What a job does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Sync the vault into the index.
    Index,
    /// Audit the memory (`memory hygiene`).
    Hygiene,
    /// Report what the memory is missing (`memory missing`).
    Missing,
}

impl Action {
    /// Parse an action name from the config.
    ///
    /// # Errors
    ///
    /// - `ConfigError` for any name other than `index`, `hygiene` or `missing`
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "index" => Ok(Action::Index),
            "hygiene" => Ok(Action::Hygiene),
            "missing" => Ok(Action::Missing),
            other => Err(MemoryError::ConfigError(format!(
                "unknown schedule action '{other}' (use index, hygiene or missing)"
            ))),
        }
    }
}

/// A validated job.
#[derive(Clone, Debug, PartialEq)]
pub struct Job {
    /// Unique name.
    pub name: String,
    /// Interval between runs.
    pub every: Duration,
    /// What the job does.
    pub action: Action,
}

/// Parse an interval: a positive whole number followed by `m` (minutes), `h` (hours) or `d` (days).
///
/// # Errors
///
/// - `ConfigError` for anything else, including zero and negative numbers
pub fn parse_every(s: &str) -> Result<Duration> {
    let bad =
        || MemoryError::ConfigError(format!("interval must look like 30m, 6h or 1d (got '{s}')"));
    let unit = s.trim().chars().last().ok_or_else(bad)?;
    let text = s.trim();
    let num: i64 = text[..text.len() - unit.len_utf8()]
        .parse()
        .map_err(|_| bad())?;
    if num <= 0 {
        return Err(bad());
    }
    match unit {
        'm' => Ok(Duration::minutes(num)),
        'h' => Ok(Duration::hours(num)),
        'd' => Ok(Duration::days(num)),
        _ => Err(bad()),
    }
}

/// Validate the configured jobs. With none configured, the default is `index` every hour.
///
/// # Errors
///
/// - `ConfigError` for a bad interval, an unknown action, or a duplicate name
pub fn jobs_from_config(cfg: &ScheduleConfig) -> Result<Vec<Job>> {
    if cfg.jobs.is_empty() {
        return Ok(vec![Job {
            name: "index".into(),
            every: Duration::hours(1),
            action: Action::Index,
        }]);
    }
    let mut seen = std::collections::HashSet::new();
    let mut jobs = Vec::with_capacity(cfg.jobs.len());
    for spec in &cfg.jobs {
        if spec.name.trim().is_empty() {
            return Err(MemoryError::ConfigError(
                "a schedule job needs a name".into(),
            ));
        }
        if !seen.insert(spec.name.clone()) {
            return Err(MemoryError::ConfigError(format!(
                "duplicate schedule job name '{}'",
                spec.name
            )));
        }
        jobs.push(Job {
            name: spec.name.clone(),
            every: parse_every(&spec.every)?,
            action: Action::parse(&spec.action)?,
        });
    }
    Ok(jobs)
}

/// A job is due when it never ran or its interval has passed since the last run.
pub fn is_due(last_run: Option<DateTime<Utc>>, every: Duration, now: DateTime<Utc>) -> bool {
    last_run.is_none_or(|t| t + every <= now)
}

/// One finished job, as written to the log.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Job name.
    pub job: String,
    /// Whether the action succeeded.
    pub ok: bool,
    /// One-line summary of what happened.
    pub summary: String,
}

impl Outcome {
    /// `<time> job=<name> outcome=<ok|failed> <summary>`, always one line.
    pub fn line(&self, now: DateTime<Utc>) -> String {
        let summary = self
            .summary
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "{} job={} outcome={} {}",
            now.to_rfc3339(),
            self.job,
            if self.ok { "ok" } else { "failed" },
            summary
        )
    }
}

/// `<root>\Plaesy\schedule.log`, beside `settings.json`. `None` when no per-user root exists.
pub fn log_path() -> Option<PathBuf> {
    settings::default_path().and_then(|p| p.parent().map(|d| d.join("schedule.log")))
}

fn db_err(e: rusqlite::Error) -> MemoryError {
    MemoryError::DatabaseError(e.to_string())
}

fn last_run(engine: &SearchEngine, job: &str) -> Result<Option<DateTime<Utc>>> {
    let db = engine.database();
    let conn = db.conn();
    conn.execute_batch(STATE_SCHEMA).map_err(db_err)?;
    let stored: Option<String> = conn
        .query_row(
            "SELECT last_run FROM schedule_runs WHERE job = ?1",
            params![job],
            |r| r.get(0),
        )
        .optional()
        .map_err(db_err)?;
    Ok(stored.and_then(|s| {
        DateTime::parse_from_rfc3339(&s)
            .ok()
            .map(|t| t.with_timezone(&Utc))
    }))
}

fn record(engine: &SearchEngine, job: &str, at: DateTime<Utc>, outcome: &str) -> Result<()> {
    let db = engine.database();
    let conn = db.conn();
    conn.execute_batch(STATE_SCHEMA).map_err(db_err)?;
    conn.execute(
        "INSERT INTO schedule_runs (job, last_run, outcome) VALUES (?1, ?2, ?3)
         ON CONFLICT(job) DO UPDATE SET last_run = excluded.last_run, outcome = excluded.outcome",
        params![job, at.to_rfc3339(), outcome],
    )
    .map_err(db_err)?;
    Ok(())
}

async fn run_action(engine: &SearchEngine, root: &Path, action: Action) -> Result<String> {
    // A missing vault must fail the job; indexing a non-existent folder would report success.
    if !root.is_dir() {
        return Err(MemoryError::ConfigError(format!(
            "not a directory: {}",
            root.display()
        )));
    }
    match action {
        Action::Index => {
            let s = engine.index_directory(root).await?;
            Ok(format!(
                "indexed={} unchanged={} removed={} failed={}",
                s.indexed,
                s.unchanged,
                s.removed,
                s.failed.len()
            ))
        }
        Action::Hygiene => {
            let r = engine
                .hygiene(&HygieneOptions {
                    root: Some(root.to_path_buf()),
                    ..HygieneOptions::default()
                })
                .await?;
            Ok(format!(
                "checked={} findings={}",
                r.checked,
                r.findings.len()
            ))
        }
        Action::Missing => {
            let g = engine.missing(Some(root.to_path_buf()), 20).await?;
            Ok(format!(
                "query_gaps={} structural={}",
                g.query_gaps.len(),
                g.structural.len()
            ))
        }
    }
}

fn append_log(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")
}

/// Run every job that is due at `now`, in config order. Each outcome is logged when `log` is set.
/// A failing job does not stop the others.
///
/// # Errors
///
/// - `DatabaseError` when the run state cannot be read or written
pub async fn run_due(
    engine: &SearchEngine,
    root: &Path,
    jobs: &[Job],
    now: DateTime<Utc>,
    log: Option<&Path>,
) -> Result<Vec<Outcome>> {
    let mut outcomes = Vec::new();
    for job in jobs {
        if !is_due(last_run(engine, &job.name)?, job.every, now) {
            continue;
        }
        // Claim the job before running it, so an overlapping run sees it as not due.
        record(engine, &job.name, now, "running")?;
        let (ok, summary) = match run_action(engine, root, job.action).await {
            Ok(s) => (true, s),
            Err(e) => (false, e.to_string()),
        };
        record(engine, &job.name, now, if ok { "ok" } else { "failed" })?;
        let outcome = Outcome {
            job: job.name.clone(),
            ok,
            summary,
        };
        if let Some(path) = log {
            // A log that cannot be written must not fail the job.
            if let Err(e) = append_log(path, &outcome.line(now)) {
                eprintln!("warning: cannot write {}: {e}", path.display());
            }
        }
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

/// Time until the next job is due, clamped to `[1 s, MAX_SLEEP]`.
fn sleep_for(
    engine: &SearchEngine,
    jobs: &[Job],
    now: DateTime<Utc>,
) -> Result<std::time::Duration> {
    let mut wait = MAX_SLEEP;
    for job in jobs {
        let due_at = last_run(engine, &job.name)?.map_or(now, |t| t + job.every);
        if let Ok(d) = (due_at - now).to_std() {
            wait = wait.min(d);
        } else {
            wait = std::time::Duration::from_secs(1);
        }
    }
    Ok(wait.max(std::time::Duration::from_secs(1)))
}

/// Keep running due jobs until Ctrl-C. Errors are printed and the loop continues.
///
/// # Errors
///
/// - `IoError` only if the terminal cannot be written
pub async fn run_loop(
    engine: &SearchEngine,
    root: &Path,
    jobs: &[Job],
    log: Option<&Path>,
) -> Result<()> {
    loop {
        let now = Utc::now();
        match run_due(engine, root, jobs, now, log).await {
            Ok(outcomes) => {
                for o in outcomes {
                    println!("{}", o.line(now));
                }
            }
            Err(e) => eprintln!("schedule: {e}"),
        }
        let wait = sleep_for(engine, jobs, Utc::now()).unwrap_or(MAX_SLEEP);
        tokio::select! {
            _ = tokio::time::sleep(wait) => {}
            _ = tokio::signal::ctrl_c() => return Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::config::{JobSpec, ScheduleConfig};
    use crate::memory::storage::{Database, DatabaseConfig};
    use std::sync::Arc;

    async fn engine(root: &Path) -> SearchEngine {
        let db = Arc::new(
            Database::new(DatabaseConfig {
                db_path: ":memory:".into(),
                ..DatabaseConfig::default()
            })
            .await
            .expect("db"),
        );
        let e = SearchEngine::new(db, None, crate::memory::config::SearchConfig::default())
            .await
            .expect("engine");
        e.set_watch_root(root.to_path_buf());
        e
    }

    fn spec(name: &str, every: &str, action: &str) -> JobSpec {
        JobSpec {
            name: name.into(),
            every: every.into(),
            action: action.into(),
        }
    }

    #[test]
    fn intervals_parse_and_bad_ones_are_rejected() {
        assert_eq!(parse_every("30m").unwrap(), Duration::minutes(30));
        assert_eq!(parse_every(" 6h ").unwrap(), Duration::hours(6));
        assert_eq!(parse_every("1d").unwrap(), Duration::days(1));
        for bad in ["", "h", "0h", "-1h", "5x", "1.5h", "ééh"] {
            assert!(parse_every(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn unknown_action_and_duplicate_names_fail_at_config_time() {
        assert!(Action::parse("reindex").is_err());
        let dup = ScheduleConfig {
            jobs: vec![spec("a", "1h", "index"), spec("a", "2h", "hygiene")],
        };
        assert!(jobs_from_config(&dup).is_err());
        let ok = ScheduleConfig {
            jobs: vec![spec("a", "1h", "index"), spec("b", "1d", "missing")],
        };
        assert_eq!(jobs_from_config(&ok).unwrap().len(), 2);
        assert_eq!(
            jobs_from_config(&ScheduleConfig::default()).unwrap()[0].action,
            Action::Index
        );
    }

    #[test]
    fn due_only_after_the_interval() {
        let now = Utc::now();
        assert!(is_due(None, Duration::hours(1), now));
        assert!(!is_due(Some(now), Duration::hours(1), now));
        assert!(is_due(
            Some(now - Duration::hours(1)),
            Duration::hours(1),
            now
        ));
    }

    #[tokio::test]
    async fn once_twice_second_run_is_not_due() {
        let dir = tempfile::tempdir().expect("tmp");
        std::fs::write(dir.path().join("a.md"), "# A\n\nAlpha note.\n").expect("w");
        let e = engine(dir.path()).await;
        let jobs = jobs_from_config(&ScheduleConfig::default()).unwrap();
        let now = Utc::now();
        assert_eq!(
            run_due(&e, dir.path(), &jobs, now, None)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(run_due(&e, dir.path(), &jobs, now, None)
            .await
            .unwrap()
            .is_empty());
        let later = now + Duration::hours(1) + Duration::seconds(1);
        assert_eq!(
            run_due(&e, dir.path(), &jobs, later, None)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn failed_job_does_not_stop_others() {
        let dir = tempfile::tempdir().expect("tmp");
        let missing_root = dir.path().join("no-such-vault");
        let e = engine(dir.path()).await;
        let jobs = vec![
            Job {
                name: "broken".into(),
                every: Duration::hours(1),
                action: Action::Index,
            },
            Job {
                name: "audit".into(),
                every: Duration::hours(1),
                action: Action::Hygiene,
            },
        ];
        let log = dir.path().join("logs").join("schedule.log");
        let now = Utc::now();
        let out = run_due(&e, &missing_root, &jobs, now, Some(&log))
            .await
            .unwrap();
        // Both jobs share the root, so both fail; the point is that the second still runs and logs.
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|o| !o.ok));
        let text = std::fs::read_to_string(&log).unwrap();
        assert_eq!(text.lines().count(), 2);
        assert!(text.contains("job=broken outcome=failed"));
        assert!(text.contains("job=audit outcome=failed"));
        assert!(text.contains("not a directory"));
    }

    #[test]
    fn schedule_log_line_is_one_line() {
        let o = Outcome {
            job: "index".into(),
            ok: true,
            summary: "indexed=1\nunchanged=2".into(),
        };
        let line = o.line(Utc::now());
        assert!(!line.contains('\n'));
        assert!(line.ends_with("indexed=1 unchanged=2"));
    }

    #[test]
    fn schedule_log_appends_and_creates_its_folder() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("sub").join("schedule.log");
        append_log(&path, "first").unwrap();
        append_log(&path, "second").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first\nsecond\n");
    }

    #[tokio::test]
    async fn one_line_per_job_per_run() {
        let dir = tempfile::tempdir().expect("tmp");
        std::fs::write(dir.path().join("a.md"), "# A\n\nAlpha.\n").expect("w");
        let e = engine(dir.path()).await;
        let jobs = vec![
            Job {
                name: "index".into(),
                every: Duration::hours(1),
                action: Action::Index,
            },
            Job {
                name: "audit".into(),
                every: Duration::hours(1),
                action: Action::Hygiene,
            },
            Job {
                name: "gaps".into(),
                every: Duration::hours(1),
                action: Action::Missing,
            },
        ];
        let log = dir.path().join("schedule.log");
        run_due(&e, dir.path(), &jobs, Utc::now(), Some(&log))
            .await
            .unwrap();
        let text = std::fs::read_to_string(&log).unwrap();
        assert_eq!(text.lines().count(), jobs.len());
        assert!(text.lines().all(|l| l.contains("outcome=ok")));
    }
}
