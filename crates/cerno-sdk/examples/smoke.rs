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

    println!("health: {}", client.health().await?);

    let answers = client
        .systemone("Ticket: Serverraum-Klima ausgefallen, 31 Grad und steigend.")
        .noul("urgent", "Ist das dringend?")
        .choice("team", "Welches Team?", ["IT", "Facility", "Personal"])
        .score(
            "sev",
            "Wie schwer?",
            ["unkritisch", "gering", "mittel", "hoch", "kritisch"],
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
