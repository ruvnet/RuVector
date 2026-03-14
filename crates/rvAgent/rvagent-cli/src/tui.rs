//! Terminal UI for rvAgent CLI using ratatui + crossterm.
//!
//! Provides:
//! - Input area at bottom
//! - Scrollable message area
//! - Status bar showing model, token count, session ID
//! - Tool call display with collapsible output

use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};

use rvagent_core::messages::Message;

use crate::app::TuiEvent;
use crate::display;

// ---------------------------------------------------------------------------
// Tui state
// ---------------------------------------------------------------------------

/// Terminal UI state.
pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    /// Rendered message lines for the scrollable area.
    messages: Vec<DisplayMessage>,
    /// Current input buffer.
    input: String,
    /// Cursor position within the input buffer.
    cursor: usize,
    /// Scroll offset for the message area.
    scroll_offset: u16,
    /// Status text shown in the status bar.
    status: String,
    /// Model identifier displayed in the status bar.
    model: String,
    /// Session ID displayed in the status bar.
    session_id: String,
    /// Approximate token count for display.
    token_count: usize,
}

/// A rendered message for display.
struct DisplayMessage {
    role: String,
    content: String,
    tool_calls: Vec<DisplayToolCall>,
}

/// A tool call for display.
struct DisplayToolCall {
    name: String,
    output: String,
    collapsed: bool,
}

impl Tui {
    /// Create a new TUI, entering the alternate screen and raw mode.
    pub fn new(model: &str, session_id: &str) -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            terminal,
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll_offset: 0,
            status: "Ready".into(),
            model: model.to_string(),
            session_id: session_id.to_string(),
            token_count: 0,
        })
    }

    /// Add a message to the display.
    pub fn add_message(&mut self, msg: &Message) {
        let (role, content, tool_calls) = match msg {
            Message::Human(h) => ("you".to_string(), h.content.clone(), vec![]),
            Message::Ai(a) => {
                let tcs: Vec<DisplayToolCall> = a
                    .tool_calls
                    .iter()
                    .map(|tc| DisplayToolCall {
                        name: tc.name.clone(),
                        output: serde_json::to_string_pretty(&tc.args)
                            .unwrap_or_default(),
                        collapsed: true,
                    })
                    .collect();
                ("assistant".to_string(), a.content.clone(), tcs)
            }
            Message::System(s) => ("system".to_string(), s.content.clone(), vec![]),
            Message::Tool(t) => (
                format!("tool:{}", t.tool_call_id),
                t.content.clone(),
                vec![],
            ),
        };

        // Rough token estimate for display.
        self.token_count += content.len() / 4;

        self.messages.push(DisplayMessage {
            role,
            content,
            tool_calls,
        });

        // Auto-scroll to bottom.
        self.scroll_to_bottom();
    }

    /// Update the status bar text.
    pub fn set_status(&mut self, status: &str) {
        self.status = status.to_string();
    }

    /// Force a redraw of the terminal.
    pub fn redraw(&mut self) -> Result<()> {
        let messages = &self.messages;
        let input = &self.input;
        let cursor = self.cursor;
        let scroll_offset = self.scroll_offset;
        let status = &self.status;
        let model = &self.model;
        let session_id = &self.session_id;
        let token_count = self.token_count;

        self.terminal.draw(|f| {
            render_frame(
                f, messages, input, cursor, scroll_offset, status, model,
                session_id, token_count,
            );
        })?;
        Ok(())
    }

    /// Shut down the TUI, restoring terminal state.
    pub fn shutdown(&mut self) -> Result<()> {
        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;
        Ok(())
    }

    /// Wait for the next TUI event (input, quit, resize).
    pub async fn next_event(&mut self) -> Result<TuiEvent> {
        loop {
            // Draw current state.
            {
                let messages = &self.messages;
                let input = &self.input;
                let cursor = self.cursor;
                let scroll_offset = self.scroll_offset;
                let status = &self.status;
                let model = &self.model;
                let session_id = &self.session_id;
                let token_count = self.token_count;

                self.terminal.draw(|f| {
                    render_frame(
                        f, messages, input, cursor, scroll_offset, status, model,
                        session_id, token_count,
                    );
                })?;
            }

            // Poll for events with a 100ms timeout for responsiveness.
            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    Event::Key(key) => {
                        if let Some(ev) = self.handle_key(key) {
                            return Ok(ev);
                        }
                    }
                    Event::Resize(_, _) => {
                        return Ok(TuiEvent::Resize);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Handle a key event. Returns `Some(TuiEvent)` if it should be dispatched.
    fn handle_key(&mut self, key: KeyEvent) -> Option<TuiEvent> {
        match (key.modifiers, key.code) {
            // Ctrl+C / Ctrl+D → quit.
            (KeyModifiers::CONTROL, KeyCode::Char('c'))
            | (KeyModifiers::CONTROL, KeyCode::Char('d')) => Some(TuiEvent::Quit),

            // Enter → submit input.
            (_, KeyCode::Enter) => {
                if self.input.trim().is_empty() {
                    return None;
                }
                let text = std::mem::take(&mut self.input);
                self.cursor = 0;
                Some(TuiEvent::Input(text))
            }

            // Backspace.
            (_, KeyCode::Backspace) => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.input.remove(self.cursor);
                }
                None
            }

            // Delete.
            (_, KeyCode::Delete) => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                }
                None
            }

            // Left/Right cursor movement.
            (_, KeyCode::Left) => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                }
                None
            }
            (_, KeyCode::Right) => {
                if self.cursor < self.input.len() {
                    self.cursor += 1;
                }
                None
            }

            // Home / End.
            (_, KeyCode::Home) => {
                self.cursor = 0;
                None
            }
            (_, KeyCode::End) => {
                self.cursor = self.input.len();
                None
            }

            // Page Up / Page Down for scrolling.
            (_, KeyCode::PageUp) => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
                None
            }
            (_, KeyCode::PageDown) => {
                self.scroll_offset = self.scroll_offset.saturating_add(10);
                None
            }

            // Regular character input.
            (_, KeyCode::Char(c)) => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
                None
            }

            _ => None,
        }
    }

    /// Scroll the message view to the bottom.
    fn scroll_to_bottom(&mut self) {
        // Estimate total lines and set scroll offset so the last messages are visible.
        let total_lines: usize = self
            .messages
            .iter()
            .map(|m| {
                let content_lines = m.content.lines().count().max(1);
                let tc_lines: usize = m.tool_calls.len();
                content_lines + tc_lines + 2 // header + blank
            })
            .sum();

        // Terminal height isn't known here without the frame, so use a high offset
        // and let ratatui clamp it. In practice the Paragraph scroll handles this.
        if total_lines > 20 {
            self.scroll_offset = (total_lines - 20) as u16;
        } else {
            self.scroll_offset = 0;
        }
    }
}
