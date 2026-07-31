//! Grouped swarm progress block.
//!
//! One expandable scrollback card represents all members of a shell-reported
//! swarm. The parent keeps child-session tracking separately, so opening a
//! child view remains unchanged.
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::render::line_utils::truncate_str;
use crate::scrollback::block::BlockContent;
use crate::scrollback::types::{AccentStyle, BlockContext, BlockOutput, DisplayMode};
use crate::theme::Theme;
use crate::util::format_duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwarmMemberStatus {
    Queued,
    Running,
    Waiting,
    Completed,
    Failed,
    Cancelled,
}

impl SwarmMemberStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SwarmMember {
    pub index: u32,
    pub item: String,
    pub description: String,
    pub child_session_id: Option<String>,
    pub status: SwarmMemberStatus,
    pub turns: Option<u32>,
    pub tools: Option<u32>,
    pub duration_ms: Option<u64>,
    pub context_usage_pct: Option<u8>,
    pub activity: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SwarmBlock {
    pub swarm_id: String,
    pub description: String,
    pub expected_members: u32,
    pub members: Vec<SwarmMember>,
}

impl SwarmBlock {
    pub fn new(
        swarm_id: impl Into<String>,
        description: impl Into<String>,
        expected_members: Option<u32>,
    ) -> Self {
        let expected_members = expected_members.unwrap_or(0);
        let mut members = Vec::with_capacity(expected_members as usize);
        for index in 0..expected_members {
            members.push(SwarmMember {
                index,
                item: String::new(),
                description: String::new(),
                child_session_id: None,
                status: SwarmMemberStatus::Queued,
                turns: None,
                tools: None,
                duration_ms: None,
                context_usage_pct: None,
                activity: None,
            });
        }
        Self {
            swarm_id: swarm_id.into(),
            description: description.into(),
            expected_members,
            members,
        }
    }

    fn member_mut(&mut self, index: u32) -> &mut SwarmMember {
        if index >= self.expected_members {
            self.expected_members = index + 1;
        }
        while self.members.len() <= index as usize {
            let next = self.members.len() as u32;
            self.members.push(SwarmMember {
                index: next,
                item: String::new(),
                description: String::new(),
                child_session_id: None,
                status: SwarmMemberStatus::Queued,
                turns: None,
                tools: None,
                duration_ms: None,
                context_usage_pct: None,
                activity: None,
            });
        }
        &mut self.members[index as usize]
    }

    pub fn spawn(
        &mut self,
        index: Option<u32>,
        item: Option<String>,
        description: String,
        child_session_id: String,
        expected_members: Option<u32>,
    ) {
        if let Some(expected) = expected_members {
            self.expected_members = self.expected_members.max(expected);
            while self.members.len() < expected as usize {
                let next = self.members.len() as u32;
                self.member_mut(next);
            }
        }
        let index = index.unwrap_or(self.members.len() as u32);
        let member = self.member_mut(index);
        member.item = item.unwrap_or_default();
        member.description = description;
        member.child_session_id = Some(child_session_id);
        member.status = SwarmMemberStatus::Running;
        member.activity = None;
    }

    pub fn progress(
        &mut self,
        child_session_id: &str,
        turns: u32,
        tools: u32,
        duration_ms: u64,
        context: u8,
    ) {
        if let Some(member) = self
            .members
            .iter_mut()
            .find(|m| m.child_session_id.as_deref() == Some(child_session_id))
        {
            if member.status != SwarmMemberStatus::Waiting {
                member.status = SwarmMemberStatus::Running;
                member.activity = None;
            }
            member.turns = Some(turns);
            member.tools = Some(tools);
            member.duration_ms = Some(duration_ms);
            member.context_usage_pct = Some(context);
        }
    }

    pub fn status(&mut self, child_session_id: &str, status: &str, activity: String) {
        if let Some(member) = self
            .members
            .iter_mut()
            .find(|m| m.child_session_id.as_deref() == Some(child_session_id))
        {
            member.status = match status {
                "rate_limit_waiting" => SwarmMemberStatus::Waiting,
                _ => SwarmMemberStatus::Running,
            };
            member.activity = Some(activity);
        }
    }

    pub fn is_waiting(&self, child_session_id: &str) -> bool {
        self.members.iter().any(|member| {
            member.child_session_id.as_deref() == Some(child_session_id)
                && member.status == SwarmMemberStatus::Waiting
        })
    }

    pub fn finish(
        &mut self,
        child_session_id: &str,
        status: &str,
        turns: u32,
        tools: u32,
        duration_ms: u64,
    ) {
        if let Some(member) = self
            .members
            .iter_mut()
            .find(|m| m.child_session_id.as_deref() == Some(child_session_id))
        {
            member.status = match status {
                "completed" => SwarmMemberStatus::Completed,
                "cancelled" => SwarmMemberStatus::Cancelled,
                _ => SwarmMemberStatus::Failed,
            };
            member.turns = Some(turns);
            member.tools = Some(tools);
            member.duration_ms = Some(duration_ms);
            member.activity = None;
        }
    }

    fn counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        let mut completed = 0;
        let mut failed = 0;
        let mut cancelled = 0;
        let mut running = 0;
        let mut waiting = 0;
        let mut queued = 0;
        for member in &self.members {
            match member.status {
                SwarmMemberStatus::Completed => completed += 1,
                SwarmMemberStatus::Failed => failed += 1,
                SwarmMemberStatus::Cancelled => cancelled += 1,
                SwarmMemberStatus::Running => running += 1,
                SwarmMemberStatus::Waiting => waiting += 1,
                SwarmMemberStatus::Queued => queued += 1,
            }
        }
        (completed, failed, cancelled, running, waiting, queued)
    }
}

fn swarm_identity_style(theme: &Theme, selected: bool) -> Style {
    let style = theme
        .fg(theme.accent_assistant)
        .add_modifier(Modifier::BOLD);
    if selected {
        style.add_modifier(Modifier::UNDERLINED)
    } else {
        style
    }
}

fn swarm_member_status_style(theme: &Theme, status: SwarmMemberStatus) -> Style {
    let color = match status {
        SwarmMemberStatus::Queued => return theme.muted(),
        SwarmMemberStatus::Running => theme.accent_assistant,
        SwarmMemberStatus::Waiting => theme.warning,
        SwarmMemberStatus::Completed => theme.accent_success,
        SwarmMemberStatus::Failed | SwarmMemberStatus::Cancelled => theme.accent_error,
    };
    theme.fg(color).add_modifier(Modifier::BOLD)
}

impl BlockContent for SwarmBlock {
    fn output(&self, ctx: &BlockContext) -> BlockOutput {
        let theme = Theme::current();
        let purple = theme.fg(theme.accent_assistant);
        let identity = swarm_identity_style(&theme, ctx.is_selected);
        let muted = theme.muted();
        let (completed, failed, cancelled, running, waiting, queued) = self.counts();
        let header = Line::from(vec![
            Span::styled("Swarm", identity),
            Span::styled(
                format!(" \u{00d7}{} \u{203a} ", self.expected_members),
                purple,
            ),
            Span::styled(
                format!(
                    "{} — {completed} done · {failed} failed · {cancelled} cancelled · {running} running · {waiting} waiting · {queued} queued",
                    truncate_str(&self.description, ctx.width.saturating_sub(72) as usize)
                ),
                muted,
            ),
        ]);
        if ctx.mode == DisplayMode::Collapsed {
            return BlockOutput {
                lines: vec![header.into()],
            };
        }
        let mut lines = vec![header.into()];
        for (position, member) in self.members.iter().enumerate() {
            let title = if member.item.is_empty() {
                &member.description
            } else {
                &member.item
            };
            let mut detail = String::new();
            if let Some(turns) = member.turns {
                detail.push_str(&format!(" · {turns} turns"));
            }
            if let Some(tools) = member.tools {
                detail.push_str(&format!(" · {tools} tools"));
            }
            if let Some(ms) = member.duration_ms {
                detail.push_str(&format!(
                    " · {}",
                    format_duration(std::time::Duration::from_millis(ms))
                ));
            }
            if let Some(context) = member.context_usage_pct {
                detail.push_str(&format!(" · {context}% ctx"));
            }
            if let Some(activity) = member.activity.as_deref() {
                detail.push_str(&format!(" · {activity}"));
            }
            let connector = if position + 1 == self.members.len() {
                "  \u{2514}\u{2500}"
            } else {
                "  \u{251c}\u{2500}"
            };
            let index = format!(" {:02} ", member.index + 1);
            let reserved = connector.width()
                + index.width()
                + member.status.label().width()
                + detail.width()
                + 3;
            let title = truncate_str(title, usize::from(ctx.width).saturating_sub(reserved));
            lines.push(
                Line::from(vec![
                    Span::styled(connector, purple),
                    Span::styled(index, identity),
                    Span::styled(title, muted),
                    Span::styled("  ", muted),
                    Span::styled(
                        member.status.label(),
                        swarm_member_status_style(&theme, member.status),
                    ),
                    Span::styled(detail, muted),
                ])
                .into(),
            );
        }
        BlockOutput { lines }
    }

    fn accent(&self, _ctx: &BlockContext) -> Option<AccentStyle> {
        let (_, failed, cancelled, running, waiting, queued) = self.counts();
        let theme = Theme::current();
        if failed > 0 || cancelled > 0 {
            Some(AccentStyle::static_color(theme.accent_error))
        } else if running > 0 || waiting > 0 || queued > 0 {
            Some(AccentStyle::animated(theme.accent_assistant))
        } else {
            Some(AccentStyle::static_color(theme.accent_assistant))
        }
    }
    fn bullet(&self, ctx: &BlockContext) -> Option<AccentStyle> {
        self.accent(ctx)
    }
    fn has_vpad(&self, _ctx: &BlockContext) -> bool {
        false
    }
    fn has_raw_mode(&self) -> bool {
        false
    }
    fn is_foldable(&self) -> bool {
        true
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
    fn swarm_identity_and_states_use_purple_palette() {
        let theme = Theme::groknight();
        let identity = swarm_identity_style(&theme, false);
        assert_eq!(identity.fg, Some(theme.accent_assistant));
        assert!(identity.add_modifier.contains(Modifier::BOLD));
        assert_eq!(
            swarm_member_status_style(&theme, SwarmMemberStatus::Running).fg,
            Some(theme.accent_assistant)
        );
        assert_eq!(
            swarm_member_status_style(&theme, SwarmMemberStatus::Waiting).fg,
            Some(theme.warning)
        );
    }

    #[test]
    fn fixed_slots_preserve_input_order_and_updates_merge() {
        let mut swarm = SwarmBlock::new("s", "review", Some(3));
        swarm.spawn(
            Some(2),
            Some("third".into()),
            "d3".into(),
            "c3".into(),
            Some(3),
        );
        swarm.spawn(
            Some(0),
            Some("first".into()),
            "d1".into(),
            "c1".into(),
            Some(3),
        );
        swarm.progress("c1", 2, 4, 1_000, 30);
        swarm.finish("c3", "completed", 3, 5, 2_000);
        assert_eq!(swarm.members[0].item, "first");
        assert_eq!(swarm.members[1].status, SwarmMemberStatus::Queued);
        assert_eq!(swarm.members[2].status, SwarmMemberStatus::Completed);
        assert_eq!(swarm.counts(), (1, 0, 0, 1, 0, 1));
    }

    #[test]
    fn rate_limit_waiting_survives_progress_until_retrying() {
        let mut swarm = SwarmBlock::new("s", "review", Some(1));
        swarm.spawn(
            Some(0),
            Some("first".into()),
            "d1".into(),
            "c1".into(),
            Some(1),
        );
        swarm.status(
            "c1",
            "rate_limit_waiting",
            "Rate limited · retrying in 3s · attempt 1".into(),
        );
        swarm.progress("c1", 1, 0, 2_000, 10);
        assert_eq!(swarm.members[0].status, SwarmMemberStatus::Waiting);
        assert!(swarm.members[0].activity.as_deref().unwrap().contains("3s"));

        swarm.status(
            "c1",
            "rate_limit_retrying",
            "Retrying after rate limit · attempt 1".into(),
        );
        assert_eq!(swarm.members[0].status, SwarmMemberStatus::Running);
    }
}
