//! Turning label logprobs into probabilities and a confidence.

/// Softmax over `logprobs`, with `temperature` scaling applied first.
///
/// Scaling before the exponential is plain temperature scaling: `T = 1` reproduces the model's
/// own distribution, `T > 1` flattens it. Small instruct-tuned models answer clear cases at a
/// probability of 1.0000, so flattening is usually what a caller wants — but it is their call,
/// which is why the raw logprobs travel back in every answer.
///
/// Subtracting the maximum before exponentiating keeps the sum finite for the very negative
/// logprobs a truncated distribution produces (values around `-25` are routine).
pub fn softmax(logprobs: &[f64], temperature: f64) -> Vec<f64> {
    if logprobs.is_empty() {
        return Vec::new();
    }

    let scaled: Vec<f64> = logprobs.iter().map(|lp| lp / temperature).collect();
    let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentiated: Vec<f64> = scaled.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exponentiated.iter().sum();

    if sum <= 0.0 || !sum.is_finite() {
        // Every candidate underflowed. Uniform is the only honest answer.
        let uniform = 1.0 / logprobs.len() as f64;
        return vec![uniform; logprobs.len()];
    }

    exponentiated.iter().map(|e| e / sum).collect()
}

/// How peaked a distribution is, in `0.0..=1.0`.
///
/// This is `1 - H(p)/log(n)`: one when all mass sits on a single label, zero when the labels are
/// indistinguishable. Dividing by `log(n)` puts every primitive on the same `0..=1` scale, so a
/// two-option Noul and a ten-level Score can be thresholded by the same rule — one definition for
/// all three beats three special cases no caller can hold in their head at once.
///
/// It is a scale, not an invariant. At a fixed top probability the value *rises* with the number
/// of labels: 0.9 on one of ten options scores 0.76, 0.9 on one of two scores 0.53. That is the
/// measure working as intended — narrowing ten candidates down to one is the stronger statement —
/// but it does mean confidence values are only directly comparable between questions of the same
/// shape.
pub fn confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    if n < 2 {
        return 1.0;
    }

    let entropy: f64 = probabilities
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum();

    (1.0 - entropy / (n as f64).ln()).clamp(0.0, 1.0)
}

/// The probability-weighted mean of 1-based level indices.
pub fn expected_level(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| (i + 1) as f64 * p)
        .sum()
}

/// The index of the largest probability. Ties go to the lower index.
pub fn argmax(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, v)| {
            if *v > bv { (i, *v) } else { (bi, bv) }
        })
        .0
}

/// Combine logprobs of several tokens that mean the same answer, in probability space.
///
/// Tokenisers spell one answer many ways. Measured on `gemma4:26b`, asking for a yes/no in
/// German surfaced `Ja`, `JA`, ` Ja` and `ja` as four separate entries, each holding part of the
/// mass — and `Nein` was not a token at all, arriving as `Ne` + `in`. Single-letter labels avoid
/// the split entirely, but a leading-space variant such as `" A"` still shows up on some
/// tokenisers, so every variant of a label is folded together here rather than one of them being
/// picked and the rest discarded.
pub fn logsumexp(logprobs: &[f64]) -> f64 {
    if logprobs.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }

    max + logprobs.iter().map(|lp| (lp - max).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn softmax_of_equal_logprobs_is_uniform() {
        let p = softmax(&[-1.0, -1.0, -1.0, -1.0], 1.0);

        assert!(p.iter().all(|v| close(*v, 0.25)));
    }

    /// Hand-checked against the measured `gemma4:26b` distribution: with logprobs -0.005 and
    /// -5.246 the difference is 5.241 nats, so the odds are e^5.241 : 1.
    #[test]
    fn softmax_matches_a_hand_computed_pair() {
        let p = softmax(&[-0.005, -5.246], 1.0);

        let expected_top = 1.0 / (1.0 + (-5.241f64).exp());
        assert!(close(p[0], expected_top), "got {}", p[0]);
        assert!(close(p[0] + p[1], 1.0));
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[-0.005, -5.246, -11.515, -13.662], 1.0);

        assert!(close(p.iter().sum::<f64>(), 1.0));
    }

    /// The point of calibration: same ranking, less certainty.
    #[test]
    fn higher_temperature_flattens_without_reordering() {
        let raw = softmax(&[-0.005, -5.246, -11.515], 1.0);
        let warm = softmax(&[-0.005, -5.246, -11.515], 3.0);

        assert!(warm[0] < raw[0], "{} should be below {}", warm[0], raw[0]);
        assert!(warm[1] > raw[1]);
        assert_eq!(argmax(&raw), argmax(&warm));
        assert!(close(warm.iter().sum::<f64>(), 1.0));
    }

    /// Extremely negative logprobs are the normal case once a label falls back to the floor.
    #[test]
    fn extreme_logprobs_do_not_produce_nan() {
        let p = softmax(&[0.0, -800.0, -1500.0], 1.0);

        assert!(p.iter().all(|v| v.is_finite()));
        assert!(close(p.iter().sum::<f64>(), 1.0));
        assert!(close(p[0], 1.0));
    }

    #[test]
    fn confidence_is_one_when_all_mass_is_on_one_label() {
        assert!(close(confidence(&[1.0, 0.0, 0.0]), 1.0));
    }

    #[test]
    fn confidence_is_zero_for_a_uniform_distribution() {
        assert!(close(confidence(&[0.25; 4]), 0.0));
        assert!(close(confidence(&[0.5, 0.5]), 0.0));
    }

    /// Confidence rises as mass concentrates, for a fixed number of labels.
    #[test]
    fn confidence_rises_with_concentration() {
        let flat = confidence(&[0.4, 0.3, 0.3]);
        let peaked = confidence(&[0.8, 0.1, 0.1]);
        let certain = confidence(&[0.98, 0.01, 0.01]);

        assert!(flat < peaked, "{flat} should be below {peaked}");
        assert!(peaked < certain, "{peaked} should be below {certain}");
    }

    /// Every primitive lands on the same 0..=1 scale, whatever its label count.
    #[test]
    fn confidence_stays_within_the_unit_range() {
        for n in 2..=10 {
            let mut p = vec![0.02 / (n - 1) as f64; n];
            p[0] = 0.98;

            let c = confidence(&p);
            assert!((0.0..=1.0).contains(&c), "n={n} gave {c}");
        }
    }

    /// The same top probability scores *higher* with more labels, because picking one of ten is
    /// the stronger statement. Pinned with hand-computed values so the semantics stay documented.
    #[test]
    fn confidence_is_not_invariant_to_label_count() {
        let two = confidence(&[0.9, 0.1]);
        let ten = {
            let mut p = vec![0.1 / 9.0; 10];
            p[0] = 0.9;
            confidence(&p)
        };

        assert!(close(two, 0.531_004_406_410_718_9), "got {two}");
        assert!(close(ten, 0.763_394_007_551_459_9), "got {ten}");
        assert!(two < ten);
    }

    #[test]
    fn expected_level_weights_by_probability() {
        // Split evenly between level 1 and level 5.
        assert!(close(expected_level(&[0.5, 0.0, 0.0, 0.0, 0.5]), 3.0));
        // All mass on level 4.
        assert!(close(expected_level(&[0.0, 0.0, 0.0, 1.0, 0.0]), 4.0));
    }

    #[test]
    fn argmax_picks_the_largest_and_breaks_ties_low() {
        assert_eq!(argmax(&[0.1, 0.7, 0.2]), 1);
        assert_eq!(argmax(&[0.5, 0.5]), 0);
    }

    /// Two tokens at equal probability must fold to exactly twice the mass.
    #[test]
    fn logsumexp_adds_probability_mass() {
        let combined = logsumexp(&[-1.0, -1.0]);

        assert!(
            close(combined.exp(), 2.0 * (-1.0f64).exp()),
            "got {}",
            combined.exp()
        );
    }

    #[test]
    fn logsumexp_of_a_single_value_is_that_value() {
        assert!(close(logsumexp(&[-3.25]), -3.25));
    }

    #[test]
    fn logsumexp_of_nothing_is_negative_infinity() {
        assert_eq!(logsumexp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn logsumexp_is_dominated_by_the_largest_term() {
        let combined = logsumexp(&[-0.001, -30.0]);

        assert!((combined - -0.001).abs() < 1e-6, "got {combined}");
    }
}
