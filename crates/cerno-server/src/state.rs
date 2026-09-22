//! What every handler shares.

use crate::config::Config;
use cerno_core::Engine;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub config: Arc<Config>,
    /// Caps how many questions are in flight against the host at once.
    ///
    /// Ollama serialises beyond its own parallelism setting, so firing an unbounded number of
    /// questions at it turns a burst into a queue with no backpressure. A bound keeps the
    /// latency of any single request predictable.
    pub semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn new(engine: Engine, config: Config) -> Self {
        let permits = config.max_concurrent_questions;
        Self {
            engine: Arc::new(engine),
            config: Arc::new(config),
            semaphore: Arc::new(Semaphore::new(permits)),
        }
    }
}
