//! Configuration, read once at startup.
//!
//! Environment variables carry the deployment knobs; an optional TOML file carries the model
//! table, because aliases and per-model calibration are structured data that does not fit an
//! environment variable well. Env wins over file for the values both can set.

use cerno_types::Calibration;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

const DEFAULT_BIND: &str = "0.0.0.0:3000";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "gemma4:e2b-it-qat";
const DEFAULT_KEEP_ALIVE: &str = "5m";
const DEFAULT_CONCURRENCY: usize = 4;
const DEFAULT_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{var} is not valid: {source}")]
    Var {
        var: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("could not read config file {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("could not parse config file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },

    #[error(
        "default_model {model:?} is not in the model table, and strict_models is on; \
         configure it or pick one of: {known:?}"
    )]
    UnknownDefault { model: String, known: Vec<String> },
}

/// One entry in the model table.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    /// The name passed to the host.
    pub model: String,
    #[serde(default = "default_calibration")]
    pub calibration: CalibrationEntry,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CalibrationEntry {
    pub temperature: f64,
}

fn default_calibration() -> CalibrationEntry {
    CalibrationEntry { temperature: 1.0 }
}

impl From<CalibrationEntry> for Calibration {
    fn from(entry: CalibrationEntry) -> Self {
        Calibration {
            temperature: entry.temperature,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    #[serde(default)]
    default_model: Option<String>,
    #[serde(default)]
    strict_models: Option<bool>,
    #[serde(default)]
    models: BTreeMap<String, ModelEntry>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub bind: SocketAddr,
    pub ollama_url: String,
    /// The alias or model name used when a request names none.
    pub default_model: String,
    /// Alias to model, with the calibration that alias implies.
    pub models: BTreeMap<String, ModelEntry>,
    /// When set, a request may only name a configured alias or one of their target models.
    /// Off by default, so a fresh install can point at any model Ollama has pulled.
    pub strict_models: bool,
    pub max_concurrent_questions: usize,
    pub keep_alive: Option<String>,
    pub host_timeout: Duration,
}

fn var<T>(name: &'static str, fallback: T) -> Result<T, ConfigError>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match std::env::var(name) {
        Err(_) => Ok(fallback),
        Ok(raw) => raw.trim().parse().map_err(|e: T::Err| ConfigError::Var {
            var: name,
            source: Box::new(e),
        }),
    }
}

impl Config {
    /// Read configuration from the environment and, if `CERNO_CONFIG` names one, a TOML file.
    pub fn from_env() -> Result<Self, ConfigError> {
        let file = match std::env::var("CERNO_CONFIG") {
            Err(_) => FileConfig::default(),
            Ok(path) => {
                let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
                    path: path.clone(),
                    source,
                })?;
                toml::from_str(&text).map_err(|source| ConfigError::Parse { path, source })?
            }
        };

        let default_model = std::env::var("CERNO_DEFAULT_MODEL")
            .ok()
            .or(file.default_model)
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());

        let strict_models = match std::env::var("CERNO_STRICT_MODELS") {
            Ok(raw) => {
                raw.trim()
                    .parse()
                    .map_err(|e: std::str::ParseBoolError| ConfigError::Var {
                        var: "CERNO_STRICT_MODELS",
                        source: Box::new(e),
                    })?
            }
            Err(_) => file.strict_models.unwrap_or(false),
        };

        let config = Self {
            bind: var(
                "CERNO_BIND",
                DEFAULT_BIND.parse().expect("valid default bind"),
            )?,
            ollama_url: std::env::var("CERNO_OLLAMA_URL")
                .unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            default_model,
            models: file.models,
            strict_models,
            max_concurrent_questions: var("CERNO_MAX_CONCURRENT_QUESTIONS", DEFAULT_CONCURRENCY)?
                .max(1),
            keep_alive: match std::env::var("CERNO_KEEP_ALIVE") {
                // An empty value means "do not send keep_alive at all", which is distinct from
                // the variable being absent.
                Ok(v) if v.trim().is_empty() => None,
                Ok(v) => Some(v),
                Err(_) => Some(DEFAULT_KEEP_ALIVE.to_string()),
            },
            host_timeout: Duration::from_secs(var(
                "CERNO_HOST_TIMEOUT_SECS",
                DEFAULT_TIMEOUT_SECS,
            )?),
        };

        // A default nobody can reach is a startup fault, not a runtime surprise.
        if config.strict_models && config.resolve(&config.default_model).is_none() {
            return Err(ConfigError::UnknownDefault {
                model: config.default_model.clone(),
                known: config.models.keys().cloned().collect(),
            });
        }

        Ok(config)
    }

    /// Resolve a caller-supplied name to a host model and the calibration it implies.
    ///
    /// An alias resolves to its entry. A raw model name that some alias points at resolves to
    /// that alias's calibration, so naming the model directly behaves the same as naming the
    /// alias. Anything else passes through untouched unless `strict_models` forbids it.
    pub fn resolve(&self, name: &str) -> Option<(String, Calibration)> {
        if let Some(entry) = self.models.get(name) {
            return Some((entry.model.clone(), entry.calibration.into()));
        }
        if let Some(entry) = self.models.values().find(|e| e.model == name) {
            return Some((entry.model.clone(), entry.calibration.into()));
        }
        if self.strict_models {
            return None;
        }
        Some((name.to_string(), Calibration::default()))
    }

    pub fn known_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(model: &str, temperature: f64) -> ModelEntry {
        ModelEntry {
            model: model.into(),
            calibration: CalibrationEntry { temperature },
        }
    }

    fn config(strict: bool) -> Config {
        Config {
            bind: DEFAULT_BIND.parse().unwrap(),
            ollama_url: DEFAULT_OLLAMA_URL.into(),
            default_model: "small".into(),
            models: BTreeMap::from([("small".to_string(), entry("gemma4:e2b-it-qat", 2.5))]),
            strict_models: strict,
            max_concurrent_questions: 4,
            keep_alive: Some("5m".into()),
            host_timeout: Duration::from_secs(30),
        }
    }

    #[test]
    fn an_alias_resolves_to_its_model_and_calibration() {
        let (model, calibration) = config(false).resolve("small").unwrap();

        assert_eq!(model, "gemma4:e2b-it-qat");
        assert_eq!(calibration.temperature, 2.5);
    }

    /// Naming the model directly must behave exactly like naming its alias — otherwise the same
    /// model would be calibrated two different ways depending on how it was spelled.
    #[test]
    fn a_models_own_name_inherits_the_alias_calibration() {
        let (model, calibration) = config(false).resolve("gemma4:e2b-it-qat").unwrap();

        assert_eq!(model, "gemma4:e2b-it-qat");
        assert_eq!(calibration.temperature, 2.5);
    }

    #[test]
    fn an_unconfigured_model_passes_through_uncalibrated() {
        let (model, calibration) = config(false).resolve("granite4:3b").unwrap();

        assert_eq!(model, "granite4:3b");
        assert_eq!(calibration.temperature, 1.0);
    }

    #[test]
    fn strict_mode_rejects_anything_not_configured() {
        assert!(config(true).resolve("granite4:3b").is_none());
        assert!(config(true).resolve("small").is_some());
        assert!(config(true).resolve("gemma4:e2b-it-qat").is_some());
    }

    #[test]
    fn a_toml_table_parses_into_the_model_map() {
        let file: FileConfig = toml::from_str(
            r#"
            default_model = "small"
            strict_models = true

            [models.small]
            model = "gemma4:e2b-it-qat"
            calibration = { temperature = 2.5 }

            [models.big]
            model = "gemma4:26b-a4b-it-q4_K_M"
            "#,
        )
        .unwrap();

        assert_eq!(file.default_model.as_deref(), Some("small"));
        assert_eq!(file.strict_models, Some(true));
        assert_eq!(file.models["small"].calibration.temperature, 2.5);
        // Calibration is optional and falls back to the identity.
        assert_eq!(file.models["big"].calibration.temperature, 1.0);
    }
}
