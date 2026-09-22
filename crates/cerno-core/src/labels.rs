//! The answer alphabet.

use cerno_types::MAX_OPTIONS;

/// Single capital letters, in order.
///
/// Every entry is one token in every tokeniser we have measured, which is the whole point:
/// multi-token answers ("Nein" arrives as `Ne` + `in`) cannot be read off a single-token
/// distribution at all, and near-variants ("Ja", "JA", " Ja", "ja") split the mass across
/// entries. A capital letter has neither problem.
///
/// The list stops at 20 because that is Ollama's `top_logprobs` ceiling — a 21st label could
/// never be observed.
const ALPHABET: [&str; MAX_OPTIONS] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T",
];

/// The label for a 0-based option index, or `None` past the end of the alphabet.
pub fn label(index: usize) -> Option<&'static str> {
    ALPHABET.get(index).copied()
}

/// The first `count` labels.
pub fn labels(count: usize) -> Vec<&'static str> {
    ALPHABET.iter().take(count).copied().collect()
}

/// Whether `token` is a spelling of `label`.
///
/// Matching ignores surrounding whitespace and letter case, because a tokeniser may hand back
/// `" A"` where another hands back `"A"`, and a model may answer in lower case. Callers fold
/// every matching token together rather than picking one.
pub fn matches(token: &str, label: &str) -> bool {
    let token = token.trim();
    token.len() == label.len() && token.eq_ignore_ascii_case(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_start_at_a_and_run_in_order() {
        assert_eq!(labels(3), vec!["A", "B", "C"]);
        assert_eq!(label(0), Some("A"));
        assert_eq!(label(19), Some("T"));
    }

    /// The alphabet must not outrun what the host can report.
    #[test]
    fn the_alphabet_stops_at_the_host_ceiling() {
        assert_eq!(ALPHABET.len(), MAX_OPTIONS);
        assert_eq!(label(MAX_OPTIONS), None);
        assert_eq!(labels(99).len(), MAX_OPTIONS);
    }

    #[test]
    fn matching_tolerates_leading_space_and_lower_case() {
        assert!(matches("A", "A"));
        assert!(matches(" A", "A"));
        assert!(matches("a", "A"));
        assert!(matches("  a  ", "A"));
    }

    /// `"AB"` is a different answer, not a sloppy `"A"`.
    #[test]
    fn matching_rejects_longer_tokens() {
        assert!(!matches("AB", "A"));
        assert!(!matches("A)", "A"));
        assert!(!matches("B", "A"));
        assert!(!matches("", "A"));
    }
}
