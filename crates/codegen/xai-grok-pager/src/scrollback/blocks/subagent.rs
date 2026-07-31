//! SubagentBlock — scrollback entries for subagent lifecycle.
//!
//! Similar to BgTaskBlock: always collapsed, with warm orange identity chrome,
//! an animated orange bullet while running, and outcome colors when done.
//! Enter / Ctrl-F opens the subagent view.
//!
//! Two modes:
//! - **Blocking** (sync): Single `Started` block. Blinks while running,
//!   turns green/red when done. Text: `Subagent "description"`
//! - **Background** (async): `Started` block stays forever (turns gray).
//!   A separate `Completed`/`Failed` block is added when done.
//!   Started text: `Subagent [type] › started: "description"`
//!   Completed text: `Subagent › completed in 43s: "description"`

use std::time::Duration;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::app::subagent::format_subagent_meta;
use crate::render::color::blend_color;
use crate::render::line_utils::truncate_str;
use crate::scrollback::block::BlockContent;
use crate::scrollback::types::{AccentStyle, BlockContext, BlockOutput, DisplayMode};
use crate::theme::Theme;
use crate::util::format_duration;

/// What kind of subagent lifecycle event this block represents.
#[derive(Debug, Clone)]
pub enum SubagentBlockKind {
    /// Subagent is running (or was running — `finish_running` stops animation).
    Started,
    /// Subagent completed successfully.
    Completed { elapsed: Duration },
    /// Subagent failed.
    Failed {
        elapsed: Duration,
        error: Option<String>,
    },
    /// Subagent was cancelled.
    Cancelled { elapsed: Duration },
}

/// Subagent scrollback block.
///
/// Always collapsed, not foldable, groupable, selectable.
/// Enter / Ctrl-F opens the subagent view.
#[derive(Debug, Clone)]
pub struct SubagentBlock {
    /// Human-readable description of the task.
    pub description: String,
    /// Child session ID (for opening the subagent view).
    pub child_session_id: String,
    /// Subagent type (e.g. "general-purpose", "explore").
    pub subagent_type: String,
    /// Named persona applied to this subagent, if any.
    pub persona: Option<String>,
    /// Role that supplied defaults for this subagent, if any.
    pub role: Option<String>,
    /// Effective model ID used by the subagent, if available.
    pub model: Option<String>,
    /// Whether the subagent was launched in background mode.
    pub is_background: bool,
    /// Lifecycle kind.
    pub kind: SubagentBlockKind,
    /// Live activity label from the child session's turn tracker.
    ///
    /// Updated on each `SubagentProgress` tick while the subagent is running.
    /// Shown inline in the collapsed scrollback line (e.g. "Thinking",
    /// "Running: cargo build") so the user sees interactive progress without
    /// opening the subagent view.
    pub activity_label: Option<String>,
}

impl SubagentBlock {
    /// Create a "Subagent started" block (for both sync and async).
    pub fn started(
        description: impl Into<String>,
        child_session_id: impl Into<String>,
        subagent_type: impl Into<String>,
        persona: Option<String>,
        role: Option<String>,
        model: Option<String>,
        is_background: bool,
    ) -> Self {
        Self {
            description: description.into(),
            child_session_id: child_session_id.into(),
            subagent_type: subagent_type.into(),
            persona,
            role,
            model,
            is_background,
            kind: SubagentBlockKind::Started,
            activity_label: None,
        }
    }

    /// Create a "Subagent completed" block (background mode only).
    pub fn completed(
        description: impl Into<String>,
        child_session_id: impl Into<String>,
        elapsed: Duration,
    ) -> Self {
        Self {
            description: description.into(),
            child_session_id: child_session_id.into(),
            subagent_type: String::new(),
            persona: None,
            role: None,
            model: None,
            is_background: true,
            kind: SubagentBlockKind::Completed { elapsed },
            activity_label: None,
        }
    }

    /// Create a "Subagent failed" block (background mode only).
    pub fn failed(
        description: impl Into<String>,
        child_session_id: impl Into<String>,
        elapsed: Duration,
        error: Option<String>,
    ) -> Self {
        Self {
            description: description.into(),
            child_session_id: child_session_id.into(),
            subagent_type: String::new(),
            persona: None,
            role: None,
            model: None,
            is_background: true,
            kind: SubagentBlockKind::Failed { elapsed, error },
            activity_label: None,
        }
    }

    /// Create a "Subagent cancelled" block (background mode only).
    pub fn cancelled(
        description: impl Into<String>,
        child_session_id: impl Into<String>,
        elapsed: Duration,
    ) -> Self {
        Self {
            description: description.into(),
            child_session_id: child_session_id.into(),
            subagent_type: String::new(),
            persona: None,
            role: None,
            model: None,
            is_background: true,
            kind: SubagentBlockKind::Cancelled { elapsed },
            activity_label: None,
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.kind, SubagentBlockKind::Started)
    }
}

/// Truncate description and wrap in quotes for display.
fn quoted_desc(desc: &str, max_width: usize) -> String {
    // Reserve 2 chars for quotes
    if max_width <= 2 {
        return "\u{201C}\u{2026}\u{201D}".to_string(); // "…"
    }
    let inner = truncate_str(desc, max_width - 2);
    format!("\u{201C}{inner}\u{201D}")
}

fn subagent_identity_style(theme: &Theme, selected: bool) -> Style {
    let style = theme.fg(theme.path).add_modifier(Modifier::BOLD);
    if selected {
        style.add_modifier(Modifier::UNDERLINED)
    } else {
        style
    }
}

fn subagent_prefix(
    theme: &Theme,
    selected: bool,
    subagent_type: &str,
) -> (Vec<Span<'static>>, usize) {
    let identity = theme.fg(theme.path);
    let mut width = "Subagent".width();
    let mut spans = vec![Span::styled(
        "Subagent",
        subagent_identity_style(theme, selected),
    )];
    if !subagent_type.is_empty() {
        let badge = format!(" [{subagent_type}]");
        width += badge.width();
        spans.push(Span::styled(badge, identity));
    }
    width += " \u{203a} ".width();
    spans.push(Span::styled(" \u{203a} ", identity));
    (spans, width)
}

impl BlockContent for SubagentBlock {
    fn output(&self, ctx: &BlockContext) -> BlockOutput {
        let theme = Theme::current();
        let muted = theme.muted();
        let w = ctx.width as usize;

        let line = match (&self.kind, self.is_background) {
            (SubagentBlockKind::Started, bg) => {
                let verb = if bg { "started: " } else { "running: " };
                let activity_suffix: String = self
                    .activity_label
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .map(|a| format!(" \u{2014} {a}"))
                    .unwrap_or_default();
                let meta = format_subagent_meta(
                    self.persona.as_deref(),
                    self.role.as_deref(),
                    self.model.as_deref(),
                );
                let (mut spans, prefix_width) =
                    subagent_prefix(&theme, ctx.is_selected, &self.subagent_type);
                let overhead = prefix_width + verb.width() + meta.width() + activity_suffix.width();
                let desc = quoted_desc(&self.description, w.saturating_sub(overhead));
                spans.push(Span::styled(verb, muted));
                spans.push(Span::styled(desc, muted));
                if !activity_suffix.is_empty() {
                    spans.push(Span::styled(activity_suffix, muted));
                }
                spans.push(Span::styled(meta, muted));
                Line::from(spans)
            }
            // Completed: Subagent completed in Xs: "description"
            (SubagentBlockKind::Completed { elapsed }, _) => {
                let time_str = format_duration(*elapsed);
                let detail = format!("completed in {time_str}: ");
                let (mut spans, prefix_width) =
                    subagent_prefix(&theme, ctx.is_selected, &self.subagent_type);
                let prefix_len = prefix_width + detail.width();
                let desc = quoted_desc(&self.description, w.saturating_sub(prefix_len));
                spans.push(Span::styled(detail, muted));
                spans.push(Span::styled(desc, muted));
                Line::from(spans)
            }
            // Failed: Subagent failed in Xs: "description"
            (SubagentBlockKind::Failed { elapsed, error }, _) => {
                let time_str = format_duration(*elapsed);
                let detail = error
                    .as_deref()
                    .map(|e| format!(" ({e})"))
                    .unwrap_or_default();
                let status = format!("failed in {time_str}{detail}: ");
                let (mut spans, prefix_width) =
                    subagent_prefix(&theme, ctx.is_selected, &self.subagent_type);
                let prefix_len = prefix_width + status.width();
                let desc = quoted_desc(&self.description, w.saturating_sub(prefix_len));
                spans.push(Span::styled(status, muted));
                spans.push(Span::styled(desc, muted));
                Line::from(spans)
            }
            // Cancelled: Subagent cancelled in Xs: "description"
            (SubagentBlockKind::Cancelled { elapsed }, _) => {
                let time_str = format_duration(*elapsed);
                let detail = format!("cancelled in {time_str}: ");
                let (mut spans, prefix_width) =
                    subagent_prefix(&theme, ctx.is_selected, &self.subagent_type);
                let prefix_len = prefix_width + detail.width();
                let desc = quoted_desc(&self.description, w.saturating_sub(prefix_len));
                spans.push(Span::styled(detail, muted));
                spans.push(Span::styled(desc, muted));
                Line::from(spans)
            }
        };

        BlockOutput {
            lines: vec![line.into()],
        }
    }

    fn accent(&self, ctx: &BlockContext) -> Option<AccentStyle> {
        let theme = Theme::current();
        match &self.kind {
            SubagentBlockKind::Started if ctx.is_running => {
                Some(AccentStyle::static_color(theme.path))
            }
            _ => None,
        }
    }

    fn bullet(&self, ctx: &BlockContext) -> Option<AccentStyle> {
        let theme = Theme::current();
        match &self.kind {
            SubagentBlockKind::Started => {
                if ctx.is_running {
                    let dim = ctx.appearance.scrollback.display.dim_accent;
                    let dimmed = blend_color(theme.bg_base, theme.path, dim).unwrap_or(theme.path);
                    Some(AccentStyle::animated(dimmed))
                } else {
                    // Finished — gray bullet (same as bg task "started" after completion)
                    None
                }
            }
            SubagentBlockKind::Completed { .. } => {
                Some(AccentStyle::static_color(theme.accent_success))
            }
            SubagentBlockKind::Failed { .. } | SubagentBlockKind::Cancelled { .. } => {
                Some(AccentStyle::static_color(theme.accent_error))
            }
        }
    }

    fn has_vpad(&self, _ctx: &BlockContext) -> bool {
        false
    }

    fn has_raw_mode(&self) -> bool {
        false
    }

    fn is_foldable(&self) -> bool {
        false
    }

    fn default_display_mode(&self) -> DisplayMode {
        DisplayMode::Collapsed
    }

    fn is_selectable(&self) -> bool {
        true
    }

    fn has_bullet(&self, _ctx: &BlockContext) -> bool {
        true
    }

    fn is_groupable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_identity_uses_warm_theme_color() {
        let theme = Theme::groknight();
        let style = subagent_identity_style(&theme, false);
        assert_eq!(style.fg, Some(theme.path));
        assert!(style.add_modifier.contains(Modifier::BOLD));

        let (spans, width) = subagent_prefix(&theme, true, "explore");
        assert_eq!(spans[0].content.as_ref(), "Subagent");
        assert_eq!(spans[1].content.as_ref(), " [explore]");
        assert_eq!(spans[2].content.as_ref(), " \u{203a} ");
        assert_eq!(width, "Subagent [explore] \u{203a} ".width());
        assert!(spans[0].style.add_modifier.contains(Modifier::UNDERLINED));
    }
}
