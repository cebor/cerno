//! The modal for adding or changing one question.
//!
//! Every field is a `TextArea`, including the single-line ones. A second input library would buy
//! nothing here and cost a dependency: a one-line `TextArea` that ignores Enter already gives
//! correct cursor movement and Unicode handling.

use crate::draft::{Kind, QuestionDraft};
use ratatui::crossterm::event::{KeyCode, KeyEvent};
use ratatui_textarea::{CursorMove, TextArea};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Id,
    Kind,
    Question,
    Options,
    Levels,
}

impl Field {
    pub fn label(self) -> &'static str {
        match self {
            Field::Id => "id",
            Field::Kind => "type",
            Field::Question => "question",
            Field::Options => "options",
            Field::Levels => "levels",
        }
    }
}

pub struct Editor {
    kind: Kind,
    /// `Some(index)` when changing an existing question, `None` when adding one.
    existing: Option<usize>,
    field: Field,
    /// The id to use when the field is left blank, shown as a hint on the field.
    fallback_id: String,

    id: TextArea<'static>,
    question: TextArea<'static>,
    options: TextArea<'static>,
    levels: TextArea<'static>,
}

impl Editor {
    /// The rubric used when the levels field is left blank.
    pub const DEFAULT_LEVELS: &'static str = "5";

    /// A new question. The id and levels fields start **empty**, with their fallbacks offered
    /// as hints on the field itself.
    ///
    /// Pre-filling a field and leaving the cursor in it is what turned a typed "urgent" into
    /// "urgentq1" and a typed rubric into "5unkritisch, gering, …". Both were found by driving
    /// the real binary, and both are the same mistake: a default belongs beside the field, not
    /// inside it. An empty field types cleanly, and leaving it blank still gives something
    /// usable.
    pub fn adding(fallback_id: String) -> Self {
        Self::new(
            QuestionDraft {
                levels: String::new(),
                ..Default::default()
            },
            None,
            fallback_id,
        )
    }

    pub fn editing(draft: QuestionDraft, index: usize) -> Self {
        let fallback = draft.id.clone();
        Self::new(draft, Some(index), fallback)
    }

    fn new(draft: QuestionDraft, existing: Option<usize>, fallback_id: String) -> Self {
        Self {
            kind: draft.kind,
            existing,
            field: Field::Id,
            fallback_id,
            id: line(&draft.id),
            question: line(&draft.question),
            options: line(&draft.options),
            levels: line(&draft.levels),
        }
    }

    /// The id that will be used if the field is left blank.
    pub fn fallback_id(&self) -> &str {
        &self.fallback_id
    }

    /// Whether `field` is empty and will therefore fall back to its default.
    pub fn is_blank(&self, field: Field) -> bool {
        self.field(field).lines().concat().trim().is_empty()
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn focused(&self) -> Field {
        self.field
    }

    pub fn is_editing(&self) -> bool {
        self.existing.is_some()
    }

    /// The fields this kind actually uses, in tab order.
    ///
    /// Options belong to a choice and levels to a score, so tabbing skips the one that would do
    /// nothing. The text behind it is kept either way — see `QuestionDraft`.
    pub fn fields(&self) -> Vec<Field> {
        let mut fields = vec![Field::Id, Field::Kind, Field::Question];
        match self.kind {
            Kind::Noul => {}
            Kind::Choice => fields.push(Field::Options),
            Kind::Score => fields.push(Field::Levels),
        }
        fields
    }

    pub fn field(&self, field: Field) -> &TextArea<'static> {
        match field {
            Field::Id => &self.id,
            Field::Question => &self.question,
            Field::Options => &self.options,
            Field::Levels => &self.levels,
            Field::Kind => &self.id, // not a text field; never drawn through this path
        }
    }

    /// Replace a field's contents outright.
    pub fn set(&mut self, field: Field, value: &str) {
        let target = match field {
            Field::Id => &mut self.id,
            Field::Question => &mut self.question,
            Field::Options => &mut self.options,
            Field::Levels => &mut self.levels,
            Field::Kind => return,
        };
        *target = line(value);
    }

    pub fn focus_next(&mut self) {
        let fields = self.fields();
        let index = fields.iter().position(|f| *f == self.field).unwrap_or(0);
        self.field = fields[(index + 1) % fields.len()];
    }

    pub fn focus_previous(&mut self) {
        let fields = self.fields();
        let index = fields.iter().position(|f| *f == self.field).unwrap_or(0);
        self.field = fields[(index + fields.len() - 1) % fields.len()];
    }

    pub fn cycle_kind(&mut self, forward: bool) {
        self.kind = if forward {
            self.kind.next()
        } else {
            self.kind.previous()
        };

        // The field that was focused may no longer exist under the new kind.
        if !self.fields().contains(&self.field) {
            self.field = Field::Kind;
        }
    }

    /// Feed a key to the focused field. Returns false when the key was not consumed, so the
    /// caller can treat it as a shortcut.
    pub fn input(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Tab | KeyCode::Down => {
                self.focus_next();
                true
            }
            KeyCode::BackTab | KeyCode::Up => {
                self.focus_previous();
                true
            }
            KeyCode::Left | KeyCode::Right if self.field == Field::Kind => {
                self.cycle_kind(key.code == KeyCode::Right);
                true
            }
            // Enter commits the modal rather than inserting a newline; these are single-line
            // fields and a hidden second line would be invisible and confusing.
            KeyCode::Enter => false,
            _ => {
                let target = match self.field {
                    Field::Id => &mut self.id,
                    Field::Question => &mut self.question,
                    Field::Options => &mut self.options,
                    Field::Levels => &mut self.levels,
                    Field::Kind => return false,
                };
                target.input(key)
            }
        }
    }

    /// The draft as typed, plus the slot it belongs in.
    pub fn finish(self) -> (QuestionDraft, Option<usize>) {
        let typed_id = first_line(&self.id);
        let draft = QuestionDraft {
            id: if typed_id.trim().is_empty() {
                self.fallback_id.clone()
            } else {
                typed_id
            },
            kind: self.kind,
            question: first_line(&self.question),
            options: first_line(&self.options),
            levels: {
                let typed = first_line(&self.levels);
                if typed.trim().is_empty() {
                    Self::DEFAULT_LEVELS.to_string()
                } else {
                    typed
                }
            },
        };
        (draft, self.existing)
    }
}

/// A one-line field with the cursor **after** its contents, so typing appends.
fn line(value: &str) -> TextArea<'static> {
    let mut area = TextArea::new(vec![value.to_string()]);
    area.move_cursor(CursorMove::End);
    area
}

fn first_line(area: &TextArea<'static>) -> String {
    area.lines().first().cloned().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyEvent, KeyModifiers};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_str(editor: &mut Editor, text: &str) {
        for ch in text.chars() {
            editor.input(key(KeyCode::Char(ch)));
        }
    }

    #[test]
    fn a_noul_offers_no_options_or_levels_field() {
        let editor = Editor::adding("q1".into());

        assert_eq!(
            editor.fields(),
            vec![Field::Id, Field::Kind, Field::Question]
        );
    }

    #[test]
    fn each_kind_offers_the_field_it_uses() {
        let mut editor = Editor::adding("q1".into());

        editor.cycle_kind(true);
        assert_eq!(editor.kind(), Kind::Choice);
        assert!(editor.fields().contains(&Field::Options));
        assert!(!editor.fields().contains(&Field::Levels));

        editor.cycle_kind(true);
        assert_eq!(editor.kind(), Kind::Score);
        assert!(editor.fields().contains(&Field::Levels));
        assert!(!editor.fields().contains(&Field::Options));
    }

    #[test]
    fn tab_walks_the_fields_and_wraps() {
        let mut editor = Editor::adding("q1".into());
        assert_eq!(editor.focused(), Field::Id);

        editor.input(key(KeyCode::Tab));
        assert_eq!(editor.focused(), Field::Kind);

        editor.input(key(KeyCode::Tab));
        assert_eq!(editor.focused(), Field::Question);

        editor.input(key(KeyCode::Tab));
        assert_eq!(editor.focused(), Field::Id, "wraps");

        editor.input(key(KeyCode::BackTab));
        assert_eq!(editor.focused(), Field::Question);
    }

    /// Switching away from a kind while its own field is focused must not leave focus on a
    /// field that is no longer shown.
    #[test]
    fn changing_kind_moves_focus_off_a_field_that_disappeared() {
        let mut editor = Editor::adding("q1".into());
        editor.cycle_kind(true);
        editor.field = Field::Options;

        editor.cycle_kind(true);

        assert_eq!(editor.kind(), Kind::Score);
        assert_eq!(editor.focused(), Field::Kind);
    }

    #[test]
    fn left_and_right_change_the_kind_only_on_the_kind_field() {
        let mut editor = Editor::adding("q1".into());
        editor.field = Field::Kind;

        editor.input(key(KeyCode::Right));
        assert_eq!(editor.kind(), Kind::Choice);

        editor.input(key(KeyCode::Left));
        assert_eq!(editor.kind(), Kind::Noul);
    }

    #[test]
    fn typing_lands_in_the_focused_field() {
        let mut editor = Editor::adding("q1".into());

        type_str(&mut editor, "urgent");
        editor.input(key(KeyCode::Tab));
        editor.input(key(KeyCode::Tab));
        type_str(&mut editor, "Is this urgent?");

        let (draft, existing) = editor.finish();

        assert_eq!(draft.id, "urgent");
        assert_eq!(draft.question, "Is this urgent?");
        assert_eq!(existing, None);
    }

    /// Enter is the commit key. The editor must hand it back unconsumed and leave the field
    /// exactly as it was, rather than splitting it across an invisible second line.
    #[test]
    fn enter_is_not_consumed_by_a_field() {
        let mut editor = Editor::adding("q1".into());
        type_str(&mut editor, "urgent");

        assert!(
            !editor.input(key(KeyCode::Enter)),
            "Enter must reach the caller"
        );

        let (draft, _) = editor.finish();
        assert_eq!(draft.id, "urgent");
        assert!(!draft.id.contains('\n'));
    }

    /// Found by driving the real binary: the suggested id used to sit in the field with the
    /// cursor in front of it, so typing "urgent" produced "urgentq1".
    #[test]
    fn typing_an_id_replaces_the_suggestion_rather_than_joining_it() {
        let mut editor = Editor::adding("q1".into());

        type_str(&mut editor, "urgent");

        let (draft, _) = editor.finish();
        assert_eq!(draft.id, "urgent");
    }

    /// Found by driving the real binary: the levels field was pre-filled with "5", so typing a
    /// rubric produced "5unkritisch, gering, mittel, hoch, kritisch".
    #[test]
    fn typing_a_rubric_replaces_the_default_rather_than_joining_it() {
        let mut editor = Editor::adding("q1".into());
        editor.cycle_kind(true);
        editor.cycle_kind(true);
        assert_eq!(editor.kind(), Kind::Score);

        editor.field = Field::Levels;
        type_str(&mut editor, "gering, mittel, hoch");

        let (draft, _) = editor.finish();
        assert_eq!(draft.levels, "gering, mittel, hoch");
    }

    #[test]
    fn a_blank_rubric_falls_back_to_five_levels() {
        let mut editor = Editor::adding("q1".into());
        editor.cycle_kind(true);
        editor.cycle_kind(true);

        let (draft, _) = editor.finish();

        assert_eq!(draft.levels, "5");
    }

    /// Someone who does not care about ids should still get a usable one.
    #[test]
    fn a_blank_id_falls_back_to_the_suggestion() {
        let editor = Editor::adding("q3".into());

        let (draft, _) = editor.finish();

        assert_eq!(draft.id, "q3");
    }

    /// The same cursor rule applies when changing an existing question: typing appends.
    #[test]
    fn editing_puts_the_cursor_after_the_existing_text() {
        let mut editor = Editor::editing(
            QuestionDraft {
                id: "team".into(),
                ..Default::default()
            },
            0,
        );

        type_str(&mut editor, "s");

        let (draft, _) = editor.finish();
        assert_eq!(draft.id, "teams");
    }

    /// Clearing the field by hand must still fall back rather than sending an empty id.
    #[test]
    fn an_id_cleared_to_nothing_falls_back_too() {
        let mut editor = Editor::editing(
            QuestionDraft {
                id: "team".into(),
                ..Default::default()
            },
            0,
        );
        editor.set(Field::Id, "   ");

        let (draft, _) = editor.finish();

        assert_eq!(draft.id, "team");
    }

    #[test]
    fn editing_remembers_which_slot_it_came_from() {
        let editor = Editor::editing(
            QuestionDraft {
                id: "team".into(),
                kind: Kind::Choice,
                options: "IT, HR".into(),
                ..Default::default()
            },
            3,
        );
        assert!(editor.is_editing());

        let (draft, existing) = editor.finish();

        assert_eq!(existing, Some(3));
        assert_eq!(draft.id, "team");
        assert_eq!(draft.options, "IT, HR");
    }

    /// Text typed under one kind is still there after switching away and back.
    #[test]
    fn a_kind_change_does_not_discard_the_other_kinds_text() {
        let mut editor = Editor::adding("q1".into());
        editor.set(Field::Options, "IT, Facility");
        editor.set(Field::Levels, "gering, hoch");

        editor.cycle_kind(true);
        editor.cycle_kind(true);
        editor.cycle_kind(true);

        let (draft, _) = editor.finish();
        assert_eq!(draft.kind, Kind::Noul);
        assert_eq!(draft.options, "IT, Facility");
        assert_eq!(draft.levels, "gering, hoch");
    }
}
