//! Remembering the last request across restarts.
//!
//! Only a convenience: while trying things out you retype the same ticket text far too often.
//! Everything here therefore fails soft — a missing, unreadable or malformed file starts an
//! empty form. Refusing to launch over a convenience file would be the worse trade.

use crate::draft::QuestionDraft;
use cerno_sdk::{Client, SystemOne};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    #[serde(default)]
    pub state: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub calibration: Option<f64>,
    #[serde(default)]
    pub questions: Vec<QuestionDraft>,
}

/// `$XDG_CONFIG_HOME/cerno/last-session.json`, falling back to `$HOME/.config/…`.
///
/// Resolved by hand rather than through a crate: it is two environment variables, and the TUI
/// already earns its dependencies elsewhere.
pub fn default_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(std::env::var_os("HOME")?).join(".config"),
    };
    Some(base.join("cerno").join("last-session.json"))
}

impl Session {
    /// Read a session, or return an empty one for any reason at all.
    pub fn load_from(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn load() -> Self {
        default_path()
            .map(|p| Self::load_from(&p))
            .unwrap_or_default()
    }

    pub fn save_to(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_private(path, &serde_json::to_string_pretty(self)?)
    }

    /// Save, reporting failure to the caller rather than swallowing it — the caller decides
    /// whether a failed save is worth a message on the way out.
    pub fn save(&self) -> std::io::Result<()> {
        match default_path() {
            Some(path) => self.save_to(&path),
            None => Ok(()),
        }
    }

    /// Assemble the request this form describes, through the SDK's own builder.
    ///
    /// The request is built from the session rather than from the live widgets for two reasons.
    /// It makes what gets saved and what gets sent the same thing by construction — a reloaded
    /// form cannot send something different from the one that was saved. And it hands the
    /// background task an owned snapshot, so the request in flight is the form as it was when
    /// you pressed send, not as you have edited it since.
    ///
    /// Nothing is validated here. `cerno-core` owns those rules, the OpenAPI document publishes
    /// them, and the service names both the offending question and what to do instead. A fourth
    /// copy in the client would be the one that drifts.
    pub fn build<'a>(&self, client: &'a Client) -> SystemOne<'a> {
        let mut builder = client.systemone(self.state.clone());

        if let Some(model) = &self.model {
            builder = builder.model(model);
        }
        if let Some(temperature) = self.calibration {
            builder = builder.calibration(temperature);
        }

        for draft in &self.questions {
            builder = draft.apply(builder);
        }
        builder
    }

    /// Whether there is anything worth writing. An empty form should not leave a file behind.
    pub fn is_empty(&self) -> bool {
        self.state.trim().is_empty() && self.questions.is_empty()
    }
}

/// Write `text` readable by its owner only. The state is whatever ticket or message was being
/// tried out, which is nobody else's business on a shared machine.
#[cfg(unix)]
fn write_private(path: &Path, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    // `mode` only applies to a file this call creates; one saved by an older version keeps its
    // permissions unless they are tightened here, before anything new is written into it.
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(text.as_bytes())
}

#[cfg(not(unix))]
fn write_private(path: &Path, text: &str) -> std::io::Result<()> {
    std::fs::write(path, text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::Kind;

    fn sample() -> Session {
        Session {
            state: "Ticket: server room at 31C.".into(),
            model: Some("small".into()),
            calibration: Some(2.5),
            questions: vec![QuestionDraft {
                id: "urgent".into(),
                kind: Kind::Noul,
                question: "Is this urgent?".into(),
                options: "IT, Facility".into(),
                levels: "5".into(),
            }],
        }
    }

    #[test]
    fn a_session_survives_a_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("last-session.json");

        sample().save_to(&path).unwrap();

        assert_eq!(Session::load_from(&path), sample());
    }

    #[cfg(unix)]
    #[test]
    fn a_saved_session_is_readable_by_its_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("last-session.json");
        // As an older version left it.
        std::fs::write(&path, "{}").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        sample().save_to(&path).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{mode:o}");
        assert_eq!(Session::load_from(&path), sample());
    }

    /// The first ever launch has no file, and that is the normal case, not an error.
    #[test]
    fn a_missing_file_loads_as_an_empty_session() {
        let dir = tempfile::tempdir().unwrap();

        let loaded = Session::load_from(&dir.path().join("absent.json"));

        assert_eq!(loaded, Session::default());
        assert!(loaded.is_empty());
    }

    /// A half-written or hand-edited file must not stop the program starting.
    #[test]
    fn a_corrupt_file_loads_as_an_empty_session() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.json");
        std::fs::write(&path, "{ this is not json").unwrap();

        assert_eq!(Session::load_from(&path), Session::default());
    }

    /// A file from an older version missing fields should keep what it does have.
    #[test]
    fn unknown_and_missing_fields_are_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("partial.json");
        std::fs::write(&path, r#"{"state":"kept","future_field":42}"#).unwrap();

        let loaded = Session::load_from(&path);

        assert_eq!(loaded.state, "kept");
        assert!(loaded.questions.is_empty());
        assert_eq!(loaded.model, None);
    }

    #[test]
    fn an_untouched_form_is_empty() {
        assert!(Session::default().is_empty());
        assert!(!sample().is_empty());

        let only_state = Session {
            state: "something".into(),
            ..Default::default()
        };
        assert!(!only_state.is_empty());
    }

    #[test]
    fn the_default_path_follows_xdg_then_home() {
        // Only the shape is asserted; the environment is process-wide and shared with other
        // tests, so it is read here rather than set.
        if let Some(path) = default_path() {
            assert!(path.ends_with("cerno/last-session.json"), "{path:?}");
        }
    }
}
