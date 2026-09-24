//! Configuration, read once at startup.
//!
//! Environment variables carry the deployment knobs; an optional TOML file carries the model
//! table, because aliases and per-model calibration are structured data that does not fit an
//! environment variable well. Env wins over file for the values both can set.

use cerno_host::HostKind;
use cerno_types::Calibration;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

/// Loopback only. The service has no authentication of its own, so listening on every interface
/// by default would hand the model — and, with an OpenAI key configured, the bill — to anyone on
/// the network. A container or a shared box opts in with `CERNO_BIND=0.0.0.0:3000`.
const DEFAULT_BIND: &str = "127.0.0.1:3000";
const DEFAULT_MODEL: &str = "gemma4:e2b-it-qat";
const DEFAULT_KEEP_ALIVE: &str = "5m";
const DEFAULT_CONCURRENCY: usize = 4;
const DEFAULT_TIMEOUT_SECS: u64 = 30;
/// Below the SDKs' 60 s, so a request that runs long comes back as the service's own
/// `host_timeout` rather than as a client-side transport error that says nothing about why.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 50;

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

    /// A value that parses but cannot work: an empty model name or URL, a zero timeout.
    #[error("{setting} {reason}")]
    Unusable {
        setting: &'static str,
        reason: &'static str,
    },

    #[error(
        "default_model {model:?} is not in the model table, and strict_models is on; \
         configure it or pick one of: {known:?}"
    )]
    UnknownDefault { model: String, known: Vec<String> },

    /// A temperature that is zero, negative or not finite. Zero turns every answer uniform and a
    /// negative value inverts the ranking, both without a single error at request time.
    #[error(
        "model {alias:?} has calibration temperature {temperature}; it must be finite and above zero"
    )]
    InvalidCalibration { alias: String, temperature: f64 },
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
    /// Which runtime answers. One per process; a second runtime is a second instance.
    pub host: HostKind,
    /// Defaults to where `host` listens out of the box.
    pub host_url: String,
    /// Sent as a bearer token to the OpenAI-compatible hosts. Never logged.
    pub host_api_key: Option<String>,
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
    /// How long one request may take in total, waiting for a free slot included. Without it, 32
    /// questions at a concurrency of 4 could take eight host timeouts back to back.
    pub request_timeout: Duration,
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
            .map(|model| model.trim().to_string())
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

        let host: HostKind = var("CERNO_HOST", HostKind::Ollama)?;

        let config = Self {
            bind: var(
                "CERNO_BIND",
                DEFAULT_BIND.parse().expect("valid default bind"),
            )?,
            host,
            host_url: std::env::var("CERNO_HOST_URL")
                .unwrap_or_else(|_| host.default_url().to_string())
                .trim()
                .trim_end_matches('/')
                .to_string(),
            host_api_key: std::env::var("CERNO_HOST_API_KEY")
                .ok()
                .filter(|key| !key.trim().is_empty()),
            default_model,
            models: file.models,
            strict_models,
            max_concurrent_questions: var("CERNO_MAX_CONCURRENT_QUESTIONS", DEFAULT_CONCURRENCY)?,
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
            request_timeout: Duration::from_secs(var(
                "CERNO_REQUEST_TIMEOUT_SECS",
                DEFAULT_REQUEST_TIMEOUT_SECS,
            )?),
        };

        config.validate()?;

        // Not an error, since every request is still bounded, but the host timeout can then never
        // be the one that fires: the request's deadline always arrives first.
        if config.host_timeout >= config.request_timeout {
            tracing::warn!(
                host_timeout_secs = config.host_timeout.as_secs(),
                request_timeout_secs = config.request_timeout.as_secs(),
                "CERNO_HOST_TIMEOUT_SECS is not below CERNO_REQUEST_TIMEOUT_SECS, so it never \
                 takes effect"
            );
        }

        // Not an error: both aliases work, only naming the model directly is ambiguous.
        for (model, aliases) in config.conflicting_aliases() {
            tracing::warn!(
                model,
                ?aliases,
                "aliases point at the same model with different calibrations; a request naming \
                 the model itself gets the first alias's"
            );
        }

        Ok(config)
    }

    /// Reject a configuration that would start but answer wrongly.
    fn validate(&self) -> Result<(), ConfigError> {
        // Each of these starts a server that fails every request, and says why only then.
        if self.default_model.trim().is_empty() {
            return Err(ConfigError::Unusable {
                setting: "default_model",
                reason: "must not be empty",
            });
        }
        if self.host_url.trim().is_empty() {
            return Err(ConfigError::Unusable {
                setting: "CERNO_HOST_URL",
                reason: "must not be empty; unset it to use the host's default",
            });
        }
        if self.host_timeout.is_zero() {
            return Err(ConfigError::Unusable {
                setting: "CERNO_HOST_TIMEOUT_SECS",
                reason: "must be above zero",
            });
        }
        if self.request_timeout.is_zero() {
            return Err(ConfigError::Unusable {
                setting: "CERNO_REQUEST_TIMEOUT_SECS",
                reason: "must be above zero",
            });
        }
        // Refused like a zero timeout rather than quietly raised to one: an operator who wrote
        // 0 meant something, and it was not "one".
        if self.max_concurrent_questions == 0 {
            return Err(ConfigError::Unusable {
                setting: "CERNO_MAX_CONCURRENT_QUESTIONS",
                reason: "must be above zero",
            });
        }

        for (alias, entry) in &self.models {
            let temperature = entry.calibration.temperature;
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(ConfigError::InvalidCalibration {
                    alias: alias.clone(),
                    temperature,
                });
            }
        }

        // A default nobody can reach is a startup fault, not a runtime surprise.
        if self.strict_models && self.resolve(&self.default_model).is_none() {
            return Err(ConfigError::UnknownDefault {
                model: self.default_model.clone(),
                known: self.models.keys().cloned().collect(),
            });
        }

        Ok(())
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

    /// Models that several aliases point at with different temperatures, with those aliases.
    ///
    /// [`Config::resolve`] gives a request naming such a model the calibration of the first
    /// alias in order, which is a guess the operator should know about.
    pub fn conflicting_aliases(&self) -> Vec<(String, Vec<String>)> {
        let mut by_model: BTreeMap<&str, Vec<(&str, f64)>> = BTreeMap::new();
        for (alias, entry) in &self.models {
            by_model
                .entry(&entry.model)
                .or_default()
                .push((alias, entry.calibration.temperature));
        }

        by_model
            .into_iter()
            .filter(|(_, aliases)| aliases.iter().any(|(_, t)| *t != aliases[0].1))
            .map(|(model, aliases)| {
                (
                    model.to_string(),
                    aliases.iter().map(|(a, _)| a.to_string()).collect(),
                )
            })
            .collect()
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
            host: HostKind::Ollama,
            host_url: HostKind::Ollama.default_url().into(),
            host_api_key: None,
            default_model: "small".into(),
            models: BTreeMap::from([("small".to_string(), entry("gemma4:e2b-it-qat", 2.5))]),
            strict_models: strict,
            max_concurrent_questions: 4,
            keep_alive: Some("5m".into()),
            host_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(50),
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

    /// A negative temperature inverts every ranking and zero flattens every answer to uniform;
    /// neither may reach a request.
    #[test]
    fn a_non_positive_or_non_finite_temperature_is_refused_at_startup() {
        for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut config = config(false);
            config
                .models
                .insert("bad".into(), entry("gemma4:e2b-it-qat", temperature));

            assert!(
                matches!(
                    config.validate(),
                    Err(ConfigError::InvalidCalibration { ref alias, .. }) if alias == "bad"
                ),
                "temperature {temperature} was accepted"
            );
        }

        assert!(config(false).validate().is_ok());
    }

    /// Each of these parses, and each would fail every request rather than the startup — or,
    /// for the concurrency, be quietly replaced with a value nobody wrote.
    #[test]
    fn an_empty_model_or_url_and_a_zero_timeout_are_refused_at_startup() {
        let mut empty_model = config(false);
        empty_model.default_model = " ".into();
        let mut empty_url = config(false);
        empty_url.host_url = String::new();
        let mut no_time = config(false);
        no_time.host_timeout = Duration::ZERO;
        let mut no_request_time = config(false);
        no_request_time.request_timeout = Duration::ZERO;
        let mut no_concurrency = config(false);
        no_concurrency.max_concurrent_questions = 0;

        for (config, setting) in [
            (empty_model, "default_model"),
            (empty_url, "CERNO_HOST_URL"),
            (no_time, "CERNO_HOST_TIMEOUT_SECS"),
            (no_request_time, "CERNO_REQUEST_TIMEOUT_SECS"),
            (no_concurrency, "CERNO_MAX_CONCURRENT_QUESTIONS"),
        ] {
            assert!(
                matches!(
                    config.validate(),
                    Err(ConfigError::Unusable { setting: s, .. }) if s == setting
                ),
                "{setting} was accepted"
            );
        }
    }

    #[test]
    fn aliases_disagreeing_about_one_model_are_found() {
        let mut config = config(false);
        config
            .models
            .insert("same".into(), entry("gemma4:e2b-it-qat", 2.5));
        assert!(config.conflicting_aliases().is_empty(), "same temperature");

        config
            .models
            .insert("warm".into(), entry("gemma4:e2b-it-qat", 4.0));
        assert_eq!(
            config.conflicting_aliases(),
            vec![(
                "gemma4:e2b-it-qat".to_string(),
                vec!["same".to_string(), "small".to_string(), "warm".to_string()]
            )]
        );
    }

    #[test]
    fn strict_mode_refuses_an_unreachable_default() {
        let mut config = config(true);
        config.default_model = "missing".into();

        assert!(matches!(
            config.validate(),
            Err(ConfigError::UnknownDefault { .. })
        ));
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
