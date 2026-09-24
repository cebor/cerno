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
        push_one_line(&mut out, question);
        out.push_str("\n\n");
    }

    for (label, text) in options {
        out.push_str(label);
        out.push_str(") ");
        push_one_line(&mut out, text);
        out.push('\n');
    }

    out.push_str("\nAnswer with one letter only.");
    out
}

/// Append `text` with every run of whitespace, line breaks included, collapsed to one space.
///
/// The question and each option are one line, always. A line break inside either would start
/// what looks to the model like an option: `"x\nB) y"` offered an option B that was never asked.
/// The state keeps its lines; it sits above the question, where nothing reads it as a label.
fn push_one_line(out: &mut String, text: &str) {
    for (i, word) in text.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(word);
    }
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

    /// An option containing a line break must not forge the next label.
    #[test]
    fn an_option_is_always_one_line() {
        let prompt = user_turn("s", None, &[("A", "x\nB) y"), ("B", "  z  ")]);

        assert!(prompt.contains("A) x B) y\nB) z\n"), "{prompt}");
        assert_eq!(prompt.matches("\nB) ").count(), 1, "{prompt}");
    }

    /// The question sits right above the options, so a line break in it can forge one too.
    #[test]
    fn the_question_is_always_one_line() {
        let prompt = user_turn("s", Some("Which?\nC) extra"), &[("A", "x"), ("B", "y")]);

        assert!(
            prompt.contains("QUESTION: Which? C) extra\n\nA) x\n"),
            "{prompt}"
        );
        assert!(!prompt.contains("\nC) "), "{prompt}");
    }

    /// The state is free text and keeps its layout; only the lines near the labels are flattened.
    #[test]
    fn the_state_keeps_its_lines() {
        let prompt = user_turn("line one\nline two", None, &[("A", "x")]);

        assert!(
            prompt.starts_with("CONTEXT:\nline one\nline two\n\n"),
            "{prompt}"
        );
    }

    #[test]
    fn the_last_line_is_the_reminder() {
        let prompt = user_turn("s", Some("q"), &[("A", "x"), ("B", "y")]);

        assert!(prompt.ends_with("Answer with one letter only."));
    }
}
