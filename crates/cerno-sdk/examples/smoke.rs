//! Live check against a running cerno service.
//!
//!     cargo run -p cerno-sdk --example smoke [base-url]

use cerno_sdk::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:3000".to_string());
    let client = Client::new(&url)?;

    println!("health: {}", client.health().await);

    let answers = client
        .systemone("Ticket: Server room cooling failed, 31 degrees and rising.")
        .noul("urgent", "Is this urgent?")
        .choice("team", "Which team?", ["IT", "Facility", "HR"])
        .score(
            "sev",
            "How severe?",
            ["negligible", "minor", "moderate", "major", "critical"],
        )
        .send()
        .await?;

    println!("  noul(urgent)  = {:.4}", answers.noul("urgent")?);
    println!(
        "  choice(team)  = {} (conf {:.3})",
        answers.choice("team")?,
        answers.confidence("team")?
    );
    println!(
        "  score(sev)    = {} {:?}",
        answers.score("sev")?,
        answers.legend("sev")?
    );
    println!(
        "  model         = {} {}ms",
        answers.model(),
        answers.timing_ms()
    );

    Ok(())
}
