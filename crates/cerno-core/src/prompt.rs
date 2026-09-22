//! Prompt construction.
//!
//! The instructions are in English even though the state and the question routinely are not.
//! That is deliberate: the answer is a single letter either way, so the instruction language
//! never reaches the output, and small instruct-tuned models follow English formatting
//! instructions more reliably than translated ones. The content stays in whatever language the
//! caller wrote it in.

/// Told to the model once, ahead of the question.
pub const SYSTEM: &str = "You answer exactly one question about the context you are given. \
Reply with a single capital letter naming your answer, and nothing else. \
No explanation, no punctuation, no other words.";

/// Build the user turn: context, question, lettered options, and a closing reminder.
///
/// The reminder after the options repeats the instruction because it is the text nearest the
/// generation point, which is where a small model is most likely to still be following it.
pub fn user_turn(state: &str, question: Option<&str>, options: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(state.len() + 256);

    out.push_str("CONTEXT:\n");
    out.push_str(state.trim());
    out.push_str("\n\n");

    if let Some(question) = question {
        out.push_str("QUESTION: ");
        out.push_str(question.trim());
        out.push_str("\n\n");
    }

    for (label, text) in options {
        out.push_str(label);
        out.push_str(") ");
        out.push_str(text);
        out.push('\n');
    }

    out.push_str("\nAnswer with one letter only.");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_context_question_and_lettered_options() {
        let prompt = user_turn(
            "Printer is jammed.",
            Some("Which team?"),
            &[("A", "IT"), ("B", "Facility")],
        );

        assert_eq!(
            prompt,
            "CONTEXT:\nPrinter is jammed.\n\n\
             QUESTION: Which team?\n\n\
             A) IT\nB) Facility\n\n\
             Answer with one letter only."
        );
    }

    /// A Choice whose options speak for themselves needs no question line.
    #[test]
    fn omits_the_question_line_when_there_is_no_question() {
        let prompt = user_turn("Ticket text.", None, &[("A", "Yes"), ("B", "No")]);

        assert!(!prompt.contains("QUESTION:"));
        assert!(prompt.contains("CONTEXT:\nTicket text."));
    }

    /// Stray whitespace in caller input must not shift the layout the model was tuned against.
    #[test]
    fn trims_caller_whitespace() {
        let prompt = user_turn("  padded  \n", Some("  Urgent?  "), &[("A", "Yes")]);

        assert!(prompt.starts_with("CONTEXT:\npadded\n\n"));
        assert!(prompt.contains("QUESTION: Urgent?\n"));
    }

    #[test]
    fn the_last_line_is_the_reminder() {
        let prompt = user_turn("s", Some("q"), &[("A", "x"), ("B", "y")]);

        assert!(prompt.ends_with("Answer with one letter only."));
    }
}
