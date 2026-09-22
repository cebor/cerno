//! Drawing.

use crate::app::{App, Focus, Status};
use crate::draft::Kind;
use crate::editor::{Editor, Field};
use cerno_types::Answer;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, List, ListItem, Paragraph, Wrap};

/// Eighth-width blocks, so a probability of one percent is still visible as something.
const EIGHTHS: [char; 8] = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
const EMPTY: char = '░';

/// A horizontal bar `width` cells wide showing `probability`.
///
/// Sub-cell resolution matters here: a choice between twenty options routinely puts one percent
/// on a loser, and rounding that to an empty bar would show "impossible" where the model
/// actually said "unlikely". Eighth-blocks keep the distinction at any sane width.
pub fn bar(probability: f64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let clamped = probability.clamp(0.0, 1.0);
    let eighths = (clamped * width as f64 * 8.0).round() as usize;
    let full = (eighths / 8).min(width);
    let remainder = eighths % 8;

    let mut out = String::with_capacity(width * 3);
    for _ in 0..full {
        out.push(EIGHTHS[7]);
    }
    if full < width && remainder > 0 {
        out.push(EIGHTHS[remainder - 1]);
    }

    let drawn = full + usize::from(full < width && remainder > 0);
    for _ in drawn..width {
        out.push(EMPTY);
    }
    out
}

pub fn draw(frame: &mut Frame, app: &App) {
    let [body, status] =
        Layout::vertical([Constraint::Min(6), Constraint::Length(1)]).areas(frame.area());
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(body);
    let [state_area, questions_area] =
        Layout::vertical([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(left);

    draw_state(frame, app, state_area);
    draw_questions(frame, app, questions_area);
    draw_answers(frame, app, right);
    draw_status(frame, app, status);

    if let Some(editor) = &app.editor {
        draw_editor(frame, editor, frame.area());
    }
    if app.show_help {
        draw_help(frame, frame.area());
    }
}

fn border(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn draw_state(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::bordered()
        .title(" State (1) ")
        .border_style(border(app.focus == Focus::State));

    frame.render_widget(&block, area);
    frame.render_widget(&app.state, block.inner(area));
}

fn draw_questions(frame: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == Focus::Questions;
    let block = Block::bordered()
        .title(" Questions (2) ")
        .border_style(border(focused));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut items: Vec<ListItem> = Vec::with_capacity(app.questions.len() + 1);

    for (index, draft) in app.questions.iter().enumerate() {
        let selected = focused && index == app.selected;
        let blamed = app.is_blamed(index);

        let marker = if selected { "▸ " } else { "  " };
        let style = if blamed {
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
        } else if selected {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        items.push(ListItem::new(Line::from(vec![
            Span::styled(marker, style),
            Span::styled(format!("{:<10}", truncate(&draft.id, 10)), style),
            Span::styled(
                format!("{:<7}", draft.kind.label()),
                Style::default().fg(kind_colour(draft.kind)),
            ),
            Span::styled(draft.summary(), Style::default().fg(Color::Gray)),
        ])));
    }

    items.push(ListItem::new(Span::styled(
        "  + add question  (a)",
        Style::default().fg(Color::DarkGray),
    )));

    frame.render_widget(List::new(items), inner);
}

fn kind_colour(kind: Kind) -> Color {
    match kind {
        Kind::Noul => Color::Green,
        Kind::Choice => Color::Blue,
        Kind::Score => Color::Magenta,
    }
}

fn draw_answers(frame: &mut Frame, app: &App, area: Rect) {
    let title = match (&app.answers, app.stale) {
        (Some(_), true) => " Answers (stale) ",
        _ => " Answers ",
    };
    let block = Block::bordered()
        .title(title)
        .border_style(border(app.focus == Focus::Answers));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(answers) = &app.answers else {
        let hint = if app.questions.is_empty() {
            "Add a question with `a`, then send with Ctrl+S."
        } else {
            "Ctrl+S to send."
        };
        frame.render_widget(
            Paragraph::new(hint).style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    };

    // Answers from a previous request are dimmed rather than cleared, so a run at a different
    // calibration can be compared against the one before it.
    let base = if app.stale {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };

    let bar_width = (inner.width as usize).saturating_sub(26).clamp(4, 24);
    let mut lines: Vec<Line> = Vec::new();

    for id in answers.ids() {
        let Ok(answer) = answers.get(id) else {
            continue;
        };
        lines.extend(answer_lines(id, answer, bar_width, base));
        lines.push(Line::default());
    }

    lines.push(Line::from(Span::styled(
        format!(
            "{} · {} ms · {} tokens",
            answers.model(),
            answers.timing_ms(),
            answers.usage().input_tokens
        ),
        Style::default().fg(Color::DarkGray),
    )));

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

/// One answer, headline plus a bar per option.
fn answer_lines<'a>(id: &'a str, answer: &'a Answer, width: usize, base: Style) -> Vec<Line<'a>> {
    let bold = base.add_modifier(Modifier::BOLD);
    let dim = base.fg(Color::DarkGray);
    let mut lines = Vec::new();

    let headline = |value: String, confidence: f64| {
        Line::from(vec![
            Span::styled(format!("{:<9}", truncate(id, 9)), bold),
            Span::styled(value, bold),
            Span::styled(format!("   conf {confidence:.3}"), dim),
        ])
    };

    match answer {
        Answer::Noul {
            noul, confidence, ..
        } => {
            lines.push(headline(format!("noul {noul:.4}"), *confidence));
            lines.push(option_line("Yes", *noul, width, base));
            lines.push(option_line("No", 1.0 - *noul, width, base));
        }

        Answer::Choice {
            choice,
            confidence,
            probabilities,
            ..
        } => {
            lines.push(headline(format!("choice → {choice}"), *confidence));
            for entry in probabilities {
                lines.push(option_line(&entry.option, entry.probability, width, base));
            }
        }

        Answer::Score {
            score,
            expected_score,
            legend,
            confidence,
            probabilities,
            ..
        } => {
            lines.push(headline(
                format!("score → {score} \"{legend}\""),
                *confidence,
            ));
            lines.push(Line::from(Span::styled(
                format!("          expected {expected_score:.2}"),
                dim,
            )));
            for entry in probabilities {
                lines.push(option_line(
                    &format!("{} {}", entry.level, entry.legend),
                    entry.probability,
                    width,
                    base,
                ));
            }
        }
    }

    // A truncated answer is an upper bound, not an observation, and saying so is the whole
    // reason the flag travels back with every answer.
    if answer.truncated() {
        lines.push(Line::from(Span::styled(
            "  ⚠ truncated — a label fell outside the host's window; its value is an upper bound",
            base.fg(Color::Yellow),
        )));
    }

    lines
}

fn option_line<'a>(label: &str, probability: f64, width: usize, base: Style) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("  {:<12}", truncate(label, 12)), base),
        Span::styled(bar(probability, width), base.fg(Color::Cyan)),
        Span::styled(format!(" {:>5.1}%", probability * 100.0), base),
    ])
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let health = match app.healthy {
        Some(true) => Span::styled(" ●", Style::default().fg(Color::Green)),
        Some(false) => Span::styled(" ○ unreachable", Style::default().fg(Color::Red)),
        None => Span::styled(" ?", Style::default().fg(Color::DarkGray)),
    };

    let mut spans = vec![
        Span::styled(app.url.clone(), Style::default().fg(Color::DarkGray)),
        health,
        Span::styled(
            format!(" · {}", app.model.as_deref().unwrap_or("default model")),
            Style::default().fg(Color::DarkGray),
        ),
    ];

    if let Some(temperature) = app.calibration {
        spans.push(Span::styled(
            format!(" · T {temperature:.2}"),
            Style::default().fg(Color::Magenta),
        ));
    }

    match &app.status {
        Status::Sending { started } => {
            const SPINNER: [char; 4] = ['|', '/', '-', '\\'];
            spans.push(Span::styled(
                format!(
                    " · {} sending {:.1}s  (Esc cancels)",
                    SPINNER[(app.tick as usize) % SPINNER.len()],
                    started.elapsed().as_secs_f32()
                ),
                Style::default().fg(Color::Yellow),
            ));
        }
        Status::Failed(failure) => {
            let code = failure
                .code
                .map(|c| format!("{c:?}"))
                .unwrap_or_else(|| "error".into());
            spans.push(Span::styled(
                format!(" · {code}: {}", failure.message),
                Style::default().fg(Color::Red),
            ));
        }
        Status::Idle => spans.push(Span::styled(
            " · ^S send · ? help",
            Style::default().fg(Color::DarkGray),
        )),
    }

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_editor(frame: &mut Frame, editor: &Editor, area: Rect) {
    let fields = editor.fields();
    let height = fields.len() as u16 * 3 + 2;
    let area = centred(area, 64, height.min(area.height));

    frame.render_widget(Clear, area);
    let block = Block::bordered()
        .title(if editor.is_editing() {
            " Edit question "
        } else {
            " New question "
        })
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let constraints: Vec<Constraint> = fields.iter().map(|_| Constraint::Length(3)).collect();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    for (field, row) in fields.iter().zip(rows.iter()) {
        let focused = *field == editor.focused();

        // A default belongs beside its field, not inside it: text in the field joins whatever
        // gets typed, which is what produced "urgentq1" and "5unkritisch".
        let title = match field {
            Field::Id if editor.is_blank(Field::Id) => {
                format!(" id  (blank → {}) ", editor.fallback_id())
            }
            Field::Levels if editor.is_blank(Field::Levels) => {
                format!(" levels  (blank → {}) ", Editor::DEFAULT_LEVELS)
            }
            _ => format!(" {} ", field.label()),
        };

        let field_block = Block::bordered().title(title).border_style(border(focused));
        let field_inner = field_block.inner(*row);
        frame.render_widget(field_block, *row);

        if *field == Field::Kind {
            let spans: Vec<Span> = Kind::ALL
                .iter()
                .flat_map(|kind| {
                    let selected = *kind == editor.kind();
                    let style = if selected {
                        Style::default()
                            .fg(kind_colour(*kind))
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    [
                        Span::styled(
                            if selected {
                                format!("[{}]", kind.label())
                            } else {
                                format!(" {} ", kind.label())
                            },
                            style,
                        ),
                        Span::raw(" "),
                    ]
                })
                .collect();
            frame.render_widget(Paragraph::new(Line::from(spans)), field_inner);
        } else {
            frame.render_widget(editor.field(*field), field_inner);
        }
    }
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let lines = vec![
        "Tab / Shift+Tab   move between panes",
        "↑ ↓               select a question",
        "a / e / d         add · edit · delete",
        "Ctrl+S            send",
        "Esc               close a dialog, or cancel a request in flight",
        "m                 next model",
        "t                 calibration temperature",
        "?                 this help",
        "Ctrl+C            quit  (q also quits outside the state box)",
    ];

    let area = centred(area, 70, lines.len() as u16 + 2);
    frame.render_widget(Clear, area);

    let block = Block::bordered()
        .title(" Keys ")
        .border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(lines.join("\n")), inner);
}

fn centred(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

fn truncate(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        return value.to_string();
    }
    value
        .chars()
        .take(width.saturating_sub(1))
        .collect::<String>()
        + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(bar: &str) -> usize {
        bar.chars().filter(|c| *c != EMPTY).count()
    }

    #[test]
    fn a_bar_is_always_exactly_its_width() {
        for probability in [0.0, 0.001, 0.25, 0.5, 0.987, 1.0] {
            for width in [1usize, 4, 16, 24] {
                let rendered = bar(probability, width);
                assert_eq!(
                    rendered.chars().count(),
                    width,
                    "p={probability} width={width} gave {rendered:?}"
                );
            }
        }
    }

    #[test]
    fn a_full_bar_is_all_blocks_and_an_empty_one_is_none() {
        assert_eq!(bar(1.0, 4), "████");
        assert_eq!(bar(0.0, 4), "░░░░");
    }

    /// The point of eighth-blocks: one percent must not render as nothing.
    #[test]
    fn a_small_probability_still_shows_something() {
        let rendered = bar(0.011, 16);

        assert!(
            filled(&rendered) > 0,
            "1.1% rendered as empty: {rendered:?}"
        );
        assert!(
            filled(&rendered) < 2,
            "1.1% should be a sliver: {rendered:?}"
        );
    }

    #[test]
    fn bars_grow_monotonically_with_probability() {
        let mut previous = 0usize;
        for step in 0..=20 {
            let rendered = bar(step as f64 / 20.0, 20);
            let ink = rendered.chars().filter(|c| *c == EIGHTHS[7]).count();
            assert!(ink >= previous, "went backwards at {step}: {rendered:?}");
            previous = ink;
        }
    }

    /// Probabilities arrive from a softmax and should already be in range, but a bar is not the
    /// place to find out otherwise.
    #[test]
    fn out_of_range_values_are_clamped_rather_than_panicking() {
        assert_eq!(bar(-0.5, 4), "░░░░");
        assert_eq!(bar(2.0, 4), "████");
        assert_eq!(bar(f64::NAN, 4).chars().count(), 4);
        assert_eq!(bar(0.5, 0), "");
    }

    #[test]
    fn long_labels_are_truncated_with_an_ellipsis() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("a-very-long-option", 8), "a-very-…");
        assert_eq!(truncate("exactly10!", 10), "exactly10!");
    }
}
