//! Entry point.
//!
//! Every startup fault takes the same shape — log it through tracing, exit 1 — rather than
//! panicking through an `expect`, so an operator reads a sentence instead of a backtrace.

use cerno_core::Engine;
use cerno_server::{AppState, Config, build_router};
use std::process::ExitCode;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("cerno_server=info,cerno_core=info")),
        )
        .init();

    let config = match Config::from_env() {
        Ok(config) => config,
        Err(err) => {
            tracing::error!("configuration is not usable: {err}");
            return ExitCode::FAILURE;
        }
    };

    let host = match cerno_host::connect(
        config.host,
        &config.host_url,
        config.host_api_key.clone(),
        config.host_timeout,
    ) {
        Ok(host) => host,
        Err(err) => {
            tracing::error!("could not build the {} client: {err}", config.host);
            return ExitCode::FAILURE;
        }
    };

    let capabilities = host.capabilities();
    let engine = Engine::new(host, config.keep_alive.clone());

    tracing::info!(
        host = %config.host,
        host_url = %config.host_url,
        default_model = %config.default_model,
        aliases = config.models.len(),
        strict_models = config.strict_models,
        max_options = engine.max_options(),
        max_top_logprobs = capabilities.max_top_logprobs,
        concurrency = config.max_concurrent_questions,
        "cerno starting"
    );

    let bind = config.bind;
    let app = build_router(AppState::new(engine, config));

    let listener = match tokio::net::TcpListener::bind(bind).await {
        Ok(listener) => listener,
        Err(err) => {
            tracing::error!("could not bind {bind}: {err}");
            return ExitCode::FAILURE;
        }
    };

    tracing::info!("listening on {bind}");

    if let Err(err) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await
    {
        tracing::error!("server stopped: {err}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

/// Resolve on Ctrl+C or, on Unix, SIGTERM — which is what `docker stop` and systemd send. Without
/// the second, in-flight requests would be cut off at the end of the grace period instead of
/// being allowed to finish.
async fn shutdown() {
    let interrupt = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(err) => {
                tracing::warn!("could not listen for SIGTERM: {err}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => {}
        () = terminate => {}
    }
    tracing::info!("shutting down");
}
