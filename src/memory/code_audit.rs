//! Conservative, offline-first duplicate API audit for Dart source trees.
use serde::Serialize;
use std::{collections::HashMap, fs, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Dart,
}
#[derive(Clone, Debug, Serialize)]
pub struct Declaration {
    pub name: String,
    pub signature: String,
    pub fingerprint: String,
    pub file: String,
    pub line: usize,
    pub container: Option<String>,
    pub documentation: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct DuplicateGroup {
    pub key: String,
    pub declarations: Vec<Declaration>,
    pub severity: Severity,
    pub classification: String,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}
#[derive(Clone, Debug, Serialize)]
pub struct AuditReport {
    pub root: String,
    pub language: String,
    pub files_scanned: usize,
    pub declarations: Vec<Declaration>,
    pub exact_duplicates: Vec<DuplicateGroup>,
    pub semantic_duplicates: Vec<DuplicateGroup>,
    pub canonical_records: Vec<CanonicalRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FindCandidate {
    pub name: String,
    pub container: Option<String>,
    pub file: String,
    pub line: usize,
    pub signature: String,
    pub match_kind: String,
    pub score: f32,
    pub reason: String,
    pub documentation: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FindReport {
    pub decision: String,
    pub query: String,
    pub candidates: Vec<FindCandidate>,
    pub canonical_records: Vec<CanonicalRecord>,
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|x| x.len() > 2)
        .map(|x| x.to_ascii_lowercase())
        .collect()
}

pub fn find(
    declarations: &[Declaration],
    query: &str,
    purpose: Option<&str>,
    canonical_records: Vec<CanonicalRecord>,
) -> FindReport {
    let q = query.trim();
    let query_name = q.split('(').next().unwrap_or(q).trim();
    let exact_sig = q.contains('(');
    let purpose_words = words(purpose.unwrap_or(""));
    let mut candidates = Vec::new();
    for d in declarations {
        let exact_name = d.name == query_name;
        let exact_signature = exact_sig && d.signature == q;
        let name_words = words(&d.name);
        let doc_words = words(d.documentation.as_deref().unwrap_or(""));
        let overlap = purpose_words
            .iter()
            .filter(|w| name_words.contains(w) || doc_words.contains(w))
            .count();
        let lifecycle = matches!(
            d.name.as_str(),
            "build" | "dispose" | "initState" | "didUpdateWidget"
        );
        let semantic = !lifecycle && !purpose_words.is_empty() && overlap > 0;
        if exact_signature || exact_name || semantic {
            let (kind, score, reason) = if exact_signature {
                (
                    "exact_signature",
                    1.0,
                    "signature matches exactly".to_string(),
                )
            } else if exact_name {
                (
                    "exact_name",
                    0.98,
                    "symbol name matches exactly".to_string(),
                )
            } else {
                (
                    "semantic_candidate",
                    0.55 + 0.1 * overlap as f32 / purpose_words.len() as f32,
                    format!(
                        "purpose overlaps declaration name ({overlap} term{})",
                        if overlap == 1 { "" } else { "s" }
                    ),
                )
            };
            candidates.push(FindCandidate {
                name: d.name.clone(),
                container: d.container.clone(),
                file: d.file.clone(),
                line: d.line,
                signature: d.signature.clone(),
                match_kind: kind.into(),
                score,
                reason,
                documentation: d.documentation.clone(),
            });
        }
    }
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.file.cmp(&b.file))
            .then(a.line.cmp(&b.line))
    });
    let decision = if candidates.iter().any(|c| c.match_kind == "exact_signature") {
        "reuse_exact"
    } else if candidates.iter().any(|c| c.match_kind == "exact_name") {
        "reuse_exact"
    } else if !candidates.is_empty() || !canonical_records.is_empty() {
        "reuse_candidate"
    } else {
        "create_new"
    };
    FindReport {
        decision: decision.into(),
        query: q.into(),
        candidates,
        canonical_records,
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct CanonicalRecord {
    pub id: String,
    pub title: String,
    pub path: String,
    pub status: String,
    pub project: Option<String>,
    pub scope: Vec<String>,
}

#[derive(Clone, Debug)]
struct Tok {
    text: String,
    line: usize,
    kind: u8,
}
fn lex(s: &str) -> Vec<Tok> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut line = 1;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            if c == '\n' {
                line += 1
            };
            i += 1;
            continue;
        }
        if c == '/' && i + 1 < b.len() && b[i + 1] as char == '/' {
            i += 2;
            while i < b.len() && b[i] as char != '\n' {
                i += 1
            }
            continue;
        }
        if c == '/' && i + 1 < b.len() && b[i + 1] as char == '*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] as char == '*' && b[i + 1] as char == '/') {
                if b[i] as char == '\n' {
                    line += 1
                };
                i += 1
            }
            i = (i + 2).min(b.len());
            continue;
        }
        if c == '\'' || c == '"' {
            let q = c;
            let start = line;
            i += 1;
            while i < b.len() {
                let x = b[i] as char;
                if x == '\\' {
                    i += 2;
                    continue;
                }
                if x == q {
                    i += 1;
                    break;
                }
                if x == '\n' {
                    line += 1
                };
                i += 1
            }
            out.push(Tok {
                text: "<string>".into(),
                line: start,
                kind: 2,
            });
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let st = i;
            while i < b.len() && ((b[i] as char).is_ascii_alphanumeric() || b[i] as char == '_') {
                i += 1
            }
            out.push(Tok {
                text: s[st..i].to_string(),
                line,
                kind: 1,
            });
            continue;
        }
        out.push(Tok {
            text: c.to_string(),
            line,
            kind: 0,
        });
        i += 1;
    }
    out
}
fn norm_params(ts: &[Tok]) -> String {
    ts.iter()
        .filter(|t| t.text != "{" && t.text != "}")
        .map(|t| {
            if t.kind == 1 && !matches!(t.text.as_str(), "this" | "required" | "covariant") {
                "id"
            } else {
                t.text.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn norm_body(ts: &[Tok]) -> String {
    ts.iter()
        .map(|t| {
            if t.kind == 1 {
                match t.text.as_str() {
                    "await" | "return" | "if" | "else" | "for" | "while" | "throw" | "try"
                    | "catch" | "finally" | "true" | "false" | "null" => t.text.as_str(),
                    _ => "id",
                }
            } else if t.kind == 2 {
                "<string>"
            } else {
                t.text.as_str()
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

fn classify(ds: &[Declaration]) -> String {
    let files: Vec<_> = ds.iter().map(|d| d.file.as_str()).collect();
    let names: Vec<_> = ds.iter().map(|d| d.name.as_str()).collect();
    if names
        .iter()
        .all(|n| matches!(*n, "build" | "dispose" | "initState" | "didUpdateWidget"))
    {
        return "framework_lifecycle".into();
    }
    if files
        .iter()
        .all(|f| f.contains("l10n") || f.contains("localization"))
    {
        return "localization_variant".into();
    }
    if files
        .iter()
        .all(|f| f.starts_with("test/") || f.contains("/test/"))
        || ds.iter().all(|d| {
            d.container
                .as_deref()
                .is_some_and(|c| c.starts_with("_Fake") || c.starts_with("_No"))
        })
    {
        return "test_double".into();
    }
    if files
        .iter()
        .all(|f| f.contains("platform/") || f.contains("_stub.dart"))
    {
        return "platform_variant".into();
    }
    if ds.iter().all(|d| d.container.is_some()) {
        let containers: std::collections::HashSet<_> =
            ds.iter().filter_map(|d| d.container.as_deref()).collect();
        if containers.len() > 1 {
            return "cross_class_collision".into();
        }
    }
    "review".into()
}

fn documentation_before(text: &str, line: usize) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    if line == 0 || line > lines.len() {
        return None;
    }
    let mut docs = Vec::new();
    let mut i = line.saturating_sub(1);
    while i > 0 {
        let raw = lines[i - 1].trim();
        if let Some(doc) = raw.strip_prefix("///") {
            docs.push(doc.trim().to_string());
            i -= 1;
        } else if raw.is_empty() && !docs.is_empty() {
            i -= 1;
        } else {
            break;
        }
    }
    if docs.is_empty() {
        None
    } else {
        docs.reverse();
        Some(docs.join(" "))
    }
}

fn parse(path: &Path, root: &Path) -> std::io::Result<Vec<Declaration>> {
    let text = fs::read_to_string(path)?;
    let t = lex(&text);
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut container: Option<String> = None;
    let mut i = 0;
    while i < t.len() {
        if t[i].text == "class" {
            if let Some(n) = t.get(i + 1) {
                container = Some(n.text.clone())
            }
        }
        if t[i].text == "{" {
            depth += 1
        } else if t[i].text == "}" {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                container = None
            }
        }
        if t[i].text == "(" && depth <= 1 {
            let mut close = i + 1;
            let mut p = 1;
            while close < t.len() && p > 0 {
                if t[close].text == "(" {
                    p += 1
                } else if t[close].text == ")" {
                    p -= 1
                };
                close += 1
            }
            if p == 0 && close < t.len() {
                // Dart permits `async`, `async*`, and `sync*` between the parameter
                // list and the body. Skip those modifiers before classifying the declaration.
                let mut after_index = close;
                if t.get(after_index)
                    .is_some_and(|x| x.text == "async" || x.text == "sync")
                {
                    after_index += 1;
                    if t.get(after_index).is_some_and(|x| x.text == "*") {
                        after_index += 1;
                    }
                }
                let after = t.get(after_index).map(|x| x.text.as_str()).unwrap_or("");
                if after == "{" || after == "=>" || after == ";" {
                    let name = t.get(i.wrapping_sub(1));
                    if let Some(n) = name.filter(|x| x.kind == 1) {
                        let params = norm_params(&t[i + 1..close - 1]);
                        let sig = format!("{}({})", n.text, params);
                        let body_end = if after == "{" {
                            let mut j = close;
                            let mut d = 0;
                            while j < t.len() {
                                if t[j].text == "{" {
                                    d += 1
                                } else if t[j].text == "}" {
                                    d -= 1;
                                    if d == 0 {
                                        break;
                                    }
                                };
                                j += 1
                            }
                            j
                        } else {
                            close
                        };
                        let fingerprint = norm_body(&t[close..body_end.min(t.len())]);
                        out.push(Declaration {
                            name: n.text.clone(),
                            signature: sig,
                            fingerprint,
                            file: path
                                .strip_prefix(root)
                                .unwrap_or(path)
                                .display()
                                .to_string(),
                            line: n.line,
                            container: container.clone(),
                            documentation: documentation_before(&text, n.line),
                        });
                        i = body_end;
                    }
                }
            }
        }
        i += 1;
    }
    Ok(out)
}

pub fn audit(root: &Path, language: Language) -> std::io::Result<AuditReport> {
    audit_with_options(root, language, false)
}

pub fn audit_with_options(
    root: &Path,
    language: Language,
    include_tests: bool,
) -> std::io::Result<AuditReport> {
    let mut decls = Vec::new();
    let mut files = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        for e in fs::read_dir(&p)? {
            let p = e?.path();
            if p.is_dir() {
                if !include_tests && p.file_name().and_then(|x| x.to_str()) == Some("test") {
                    continue;
                }
                if p.file_name().and_then(|x| x.to_str()).is_some_and(|x| {
                    matches!(
                        x,
                        ".git"
                            | "build"
                            | ".dart_tool"
                            | "ephemeral"
                            | ".plugin_symlinks"
                            | ".symlinks"
                            | "third_party"
                            | "node_modules"
                            | ".pub-cache"
                            | "Pods"
                            | "vendor"
                            | "generated_plugin_registrant"
                    )
                }) {
                    continue;
                }
                stack.push(p)
            } else if p.extension().and_then(|x| x.to_str()) == Some("dart")
                && !p.file_name().and_then(|x| x.to_str()).is_some_and(|name| {
                    name.ends_with(".g.dart")
                        || name.ends_with(".freezed.dart")
                        || name.ends_with(".gr.dart")
                        || name.ends_with(".pb.dart")
                        || name.ends_with(".pbenum.dart")
                        || name.ends_with(".pbjson.dart")
                        || name.ends_with(".mocks.dart")
                        || name.starts_with("generated_")
                })
            {
                files += 1;
                decls.extend(parse(&p, root)?);
            }
        }
    }
    let groups = |key: fn(&Declaration) -> String, sev: Severity, min_fp_len: usize| {
        let mut m: HashMap<String, Vec<Declaration>> = HashMap::new();
        for d in &decls {
            if d.fingerprint.len() < min_fp_len {
                continue;
            }
            m.entry(key(d)).or_default().push(d.clone())
        }
        m.into_iter()
            .filter(|(_, v)| v.len() > 1)
            .map(|(k, v)| DuplicateGroup {
                key: k,
                classification: classify(&v),
                declarations: v,
                severity: sev.clone(),
            })
            .collect::<Vec<_>>()
    };
    let exact = groups(
        |d| format!("{}\u{1}{}", d.signature, d.fingerprint),
        Severity::Error,
        1,
    );
    let exact_keys: std::collections::HashSet<String> = exact
        .iter()
        .flat_map(|g| g.declarations.iter().map(|d| d.fingerprint.clone()))
        .collect();
    let semantic = groups(|d| d.fingerprint.clone(), Severity::Warning, 24)
        .into_iter()
        .filter(|g| !exact_keys.contains(&g.key))
        .collect();
    Ok(AuditReport {
        root: root.display().to_string(),
        language: match language {
            Language::Dart => "dart",
        }
        .into(),
        files_scanned: files,
        declarations: decls,
        exact_duplicates: exact,
        semantic_duplicates: semantic,
        canonical_records: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;
    #[test]
    fn extracts_duplicates() {
        let d = tempdir().unwrap();
        fs::write(
            d.path().join("a.dart"),
            "class A { String greet(String name) { return name.trim(); } }",
        )
        .unwrap();
        fs::write(
            d.path().join("b.dart"),
            "class B { String greet( String name ) { return name.trim(); } }",
        )
        .unwrap();
        let r = audit(d.path(), Language::Dart).unwrap();
        assert_eq!(r.declarations.len(), 2);
        assert_eq!(r.exact_duplicates.len(), 1);
        assert_eq!(
            r.exact_duplicates[0].classification,
            "cross_class_collision"
        );
        // Exact duplicates are not duplicated into the semantic report.
        assert_eq!(r.semantic_duplicates.len(), 0);
    }
    #[test]
    fn find_returns_exact_reuse_and_cross_class_candidates() {
        let d = tempdir().unwrap();
        fs::write(
            d.path().join("a.dart"),
            "class A { String greet(String name) { return name.trim(); } }",
        )
        .unwrap();
        fs::write(
            d.path().join("b.dart"),
            "class B { String greet(String name) { return name.trim(); } }",
        )
        .unwrap();
        let r = audit(d.path(), Language::Dart).unwrap();
        let found = find(&r.declarations, "greet", None, Vec::new());
        assert_eq!(found.decision, "reuse_exact");
        assert_eq!(found.candidates.len(), 2);
        assert!(found
            .candidates
            .iter()
            .all(|c| c.match_kind == "exact_name"));
    }

    #[test]
    fn find_semantic_candidate_and_create_new() {
        let d = tempdir().unwrap();
        fs::write(
            d.path().join("a.dart"),
            "class A { String authenticate(String token) { return token; } }",
        )
        .unwrap();
        let r = audit(d.path(), Language::Dart).unwrap();
        assert_eq!(
            find(
                &r.declarations,
                "login",
                Some("authenticate a user"),
                Vec::new()
            )
            .decision,
            "reuse_candidate"
        );
        assert_eq!(
            find(&r.declarations, "missing", Some("encrypt data"), Vec::new()).decision,
            "create_new"
        );
    }

    #[test]
    fn generated_files_are_excluded() {
        let d = tempdir().unwrap();
        fs::write(
            d.path().join("generated.g.dart"),
            "class A { void greet() {} }",
        )
        .unwrap();
        let r = audit(d.path(), Language::Dart).unwrap();
        assert!(r.declarations.is_empty());
    }

    #[test]
    fn documentation_is_used_for_reuse_search() {
        let d = tempdir().unwrap();
        fs::write(
            d.path().join("auth.dart"),
            "/// Authenticates a user with email and password.\nFuture<User> login(String email, String password) async { return client.login(); }\n",
        )
        .unwrap();
        let report = audit(d.path(), Language::Dart).unwrap();
        assert_eq!(
            report.declarations[0].documentation.as_deref(),
            Some("Authenticates a user with email and password.")
        );
        let found = find(
            &report.declarations,
            "signIn",
            Some("authenticate user with email and password"),
            vec![],
        );
        assert_eq!(found.decision, "reuse_candidate");
        assert_eq!(found.candidates[0].name, "login");
    }

    #[test]
    fn extracts_async_generic_methods_and_ignores_locals() {
        let d = tempdir().unwrap();
        fs::write(
            d.path().join("a.dart"),
            r#"// void fake(int x) {}
class A {
  Future<void> login(String email, String password) async {
    void local(int x) {}
    print("void fake(int x) {}");
  }
}"#,
        )
        .unwrap();
        let r = audit(d.path(), Language::Dart).unwrap();
        assert_eq!(r.declarations.len(), 1);
        assert_eq!(r.declarations[0].name, "login");
        assert_eq!(r.declarations[0].signature, "login(idid,idid)");
    }
}
