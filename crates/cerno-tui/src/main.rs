//! cerno-tui — fill in a form, fire it at the service, read the distribution.
//!
//!     cerno-tui [--url http://localhost:3000]
//!
//! The URL also comes from `CERNO_URL`, and defaults to `http://localhost:3000`.

use cerno_sdk::{Answers, Client, Error};
use cerno_tui::app::App;
use cerno_tui::keys::{self, Action};
use cerno_tui::render;
use cerno_tui::session::Session;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event};
use std::process::ExitCode;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const DEFAULT_URL: &str = "http://localhost:3000";

/// How often the spinner advances. Fast enough to look alive, slow enough that an idle TUI is
/// not a busy loop — the terminal is only redrawn when something actually changed.
const TICK: Duration = Duration::from_millis(120);

/// What the background probe found out about the service.
struct Probe {
    healthy: bool,
    models: Vec<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let url = std::env::args()
        .position(|arg| arg == "--url")
        .and_then(|i| std::env::args().nth(i + 1))
        .or_else(|| std::env::var("CERNO_URL").ok())
        .unwrap_or_else(|| DEFAULT_URL.to_string());

    let client = match Client::new(&url) {
        Ok(client) => client,
        Err(err) => {
            eprintln!("could not build a client for {url}: {err}");
            return ExitCode::FAILURE;
        }
    };

    let mut app = App::new(Session::load(), url);

    let terminal = ratatui::init();
    let outcome = run(terminal, &mut app, client).await;
    ratatui::restore();

    // Saving after the terminal is restored, so a failure can actually be read.
    let session = app.to_session();
    if !session.is_empty()
        && let Err(err) = session.save()
    {
        eprintln!("could not save the session: {err}");
    }

    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("cerno-tui stopped: {err}");
            ExitCode::FAILURE
        }
    }
}

async fn run(
    mut terminal: DefaultTerminal,
    app: &mut App,
    client: Client,
) -> Result<(), std::io::Error> {
    let mut events = spawn_event_reader();
    let (answer_tx, mut answer_rx) = mpsc::unbounded_channel::<Result<Answers, Error>>();
    let mut probe = spawn_probe(client.clone());

    let mut ticker = tokio::time::interval(TICK);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // The request currently in flight, kept so Esc can abort it.
    let mut inflight: Option<JoinHandle<()>> = None;

    terminal.draw(|frame| render::draw(frame, app))?;

    loop {
        let mut dirty = false;

        tokio::select! {
            event = events.recv() => match event {
                // The reader thread ended, which means stdin is gone. Leaving the loop is the
                // only sensible response; staying would spin on a closed channel.
                None => break,
                Some(Event::Key(key)) => {
                    match keys::handle(app, key) {
                        Action::Quit => break,
                        Action::Send => {
                            let form = app.to_session();
                            let client = client.clone();
                            let tx = answer_tx.clone();

                            app.begin_send();
                            inflight = Some(tokio::spawn(async move {
                                let _ = tx.send(form.build(&client).send().await);
                            }));
                        }
                        Action::None => {}
                    }

                    // Esc during a request clears the sending state; drop the task with it.
                    if !app.is_sending()
                        && let Some(handle) = inflight.take() {
                            handle.abort();
                        }

                    dirty = true;
                }
                Some(Event::Resize(_, _)) => dirty = true,
                Some(_) => {}
            },

            Some(result) = answer_rx.recv() => {
                app.finish_send(result);
                inflight = None;
                dirty = true;
            }

            Some(found) = probe.recv() => {
                app.healthy = Some(found.healthy);
                app.models = found.models;
                dirty = true;
            }

            _ = ticker.tick(), if app.is_sending() => {
                app.tick = app.tick.wrapping_add(1);
                dirty = true;
            }
        }

        if dirty {
            terminal.draw(|frame| render::draw(frame, app))?;
        }
    }

    if let Some(handle) = inflight {
        handle.abort();
    }
    Ok(())
}

/// Read terminal events on their own thread.
///
/// `event::read` blocks, and blocking the runtime would freeze the spinner and the in-flight
/// request along with it. A thread plus a channel keeps the loop free without pulling in
/// crossterm's async feature and a second crossterm version with it.
fn spawn_event_reader() -> mpsc::UnboundedReceiver<Event> {
    let (tx, rx) = mpsc::unbounded_channel();

    std::thread::spawn(move || {
        while let Ok(event) = event::read() {
            if tx.send(event).is_err() {
                break;
            }
        }
    });

    rx
}

/// Ask the service, once, whether it is up and which models it offers.
///
/// The answer comes back over a channel rather than by polling the task's `JoinHandle`. A
/// handle guarded with `is_finished()` has a race that a live run walks straight into: a probe
/// against a closed port fails so fast that it is already finished before the first `select!`,
/// the guard then disables the branch, and the result is never collected — the status bar sat
/// on "?" instead of saying the service was unreachable. A channel keeps the message whether or
/// not anyone was looking when it was sent.
fn spawn_probe(client: Client) -> mpsc::UnboundedReceiver<Probe> {
    let (tx, rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        let healthy = client.health().await.unwrap_or(false);

        // Aliases are a convenience for the `m` key; a service without them is not a problem.
        let models = match client.models().await {
            Ok(response) => response.models.into_iter().map(|m| m.alias).collect(),
            Err(_) => Vec::new(),
        };

        let _ = tx.send(Probe { healthy, models });
    });

    rx
}
