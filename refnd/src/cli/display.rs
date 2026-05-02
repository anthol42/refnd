use indicatif::{ProgressBar, ProgressStyle, ProgressState};
use rich_rust::prelude::*;
use std::fmt::{Write as FmtWrite};
use std::sync::OnceLock;
use std::time::Duration;
use std::collections::{BTreeMap, HashMap};
use rich_rust::console::{ConsoleBuilder, PrintOptions};
use regex::Regex;

// ── Global console ────────────────────────────────────────────────────────────
static CONSOLE: OnceLock<Console> = OnceLock::new();

// Default style: https://github.com/Dicklesworthstone/rich_rust/blob/9155058c01440042e08bf9298d405004cc2e8bbc/src/default_styles.tsv#L4
pub fn con() -> &'static Console {
    let mut theme = HashMap::new();
    theme.insert("repr.str".into(), Style::new().bold());
    CONSOLE.get_or_init(|| {
        ConsoleBuilder::default()
            .theme(Theme::new(
                Some(theme),
                true
            ))
            .build()
    })
}

// ── Spinner style ─────────────────────────────────────────────────────────────

const TICK_CHARS: &str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.cyan} {msg}")
        .unwrap()
        .tick_chars(TICK_CHARS)
}

// ── Number formatting ─────────────────────────────────────────────────────────

/// Format an integer with thousands separators and cyan markup: `1234567` → `"[cyan]1,234,567[/]"`.
pub fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut digits = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { digits.push(','); }
        digits.push(c);
    }
    let digits: String = digits.chars().rev().collect();
    format!("[cyan]{digits}[/]")
}

// ── Log helpers ───────────────────────────────────────────────────────────────

/// Print a success line.
pub fn success(msg: &str) {
    con().print_with_options(&format!("[bold green]✓[/] {msg}"), &PrintOptions::new().with_highlight(false));
}

/// Print a warning line.
pub fn warn(msg: &str) {
    con().print_with_options(&format!("[bold yellow] ⚠ Warning:  {msg}[/]"), &PrintOptions::new().with_highlight(false));
}

/// Print an error line.
pub fn error(msg: &str) {
    let mut txt = Text::new("");
    txt.append_styled("✗ Error: ", Style::new().bold().color(Color::parse("red").unwrap()));
    txt.append(&format!("{}\n", msg));
    con().print_renderable(&txt);
}

/// Render an error message inside a bordered panel with title "Error".
/// Quoted tokens ('…') are highlighted in bold red.
pub fn cli_error(msg: &str) {
    let width = con().width();
    let msg = &{
        let mut c = msg.chars();
        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        }
    };
    let re = Regex::new(r"'([^']+)'").unwrap();
    let mut text = Text::new("");
    let mut last = 0;
    for m in re.find_iter(msg) {
        if m.start() > last {
            text.append(&msg[last..m.start()]);
        }
        text.append_styled(m.as_str(), Style::new().bold().color(Color::parse("red").unwrap()));
        last = m.end();
    }
    if last < msg.len() {
        text.append(&msg[last..]);
    }
    let panel = Panel::from_rich_text(&text, width.saturating_sub(2))
        .title_from_markup(" [bold red]Error[/bold red] ")
        .title_align(JustifyMethod::Left)
        .border_style(Style::new().color(Color::parse("red").unwrap()));
    con().print_renderable(&panel);
}

/// Render a tip inside a bordered panel with title "Tip".
pub fn tip(msg: &str) {
    let width = con().width();
    let capitalized = {
        let mut c = msg.chars();
        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        }
    };
    let mut text = Text::new("");
    text.append(&capitalized);
    let panel = Panel::from_rich_text(&text, width.saturating_sub(2))
        .title_from_markup(" [bold cyan]Tip[/bold cyan] ")
        .title_align(JustifyMethod::Left)
        .border_style(Style::new().dim().color(Color::parse("cyan").unwrap()));
    con().print_renderable(&panel);
}

/// Print a colorized Usage line from a clap error.
/// Highlights `--flags` as bold, `<PLACEHOLDERS>` as dim, the rest as plain.
pub fn usage(line: &str) {
    let rest = line.strip_prefix("Usage: ").unwrap_or(line);
    let mut out = Text::new("");
    out.append_styled("Usage: ", Style::new().bold().color(Color::parse("yellow").unwrap()));
    for token in rest.split_whitespace() {
        if token.starts_with('-') {
            out.append_styled(&format!("{token} "), Style::new().bold());
        } else if token.starts_with('<') || token.ends_with('>') {
            out.append_styled(&format!("{token} "), Style::new().dim());
        } else {
            out.append(&format!("{token} "));
        }
    }
    con().print_with_options(line, &PrintOptions::new().with_highlight(false));
}

/// Print a section header using a rich Rule.
pub fn section(title: &str) {
    let rule = Rule::with_title(title);
    con().print_renderable(&rule);
}

/// Render a parameter group inside a bordered panel with darkened styling.
/// Keys are dim, values are plain (no syntax highlighting).
pub fn parameter_panel(title: &str, params: &BTreeMap<String, String>) {
    let width = con().width();
    let mut text = Text::new("");
    let last = params.len().saturating_sub(1);
    for (i, (key, val)) in params.iter().enumerate() {
        text.append_styled(&format!("  {key}: "), Style::new().dim());
        text.append(val);
        if i < last { text.append("\n"); }
    }
    let panel = Panel::from_rich_text(&text, width.saturating_sub(2))
        .title(title)
        .border_style(Style::new().dim());
    con().print_renderable(&panel);
}

// ── Progress bars ─────────────────────────────────────────────────────────────

// ── HNSW-specific progress bar ────────────────────────────────────────────────

/// Stirling approximation of log(n!) — accurate enough for ETA estimation at n ≥ 2.
/// Basis: inserting element k costs O(log k), so total work ∝ Σ log(k) = log(n!).
pub fn log_factorial(n: f64) -> f64 {
    if n <= 1.0 { return 0.0; }
    n * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI * n).ln()
}

fn format_eta(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        format!("{:.0}m {:02.0}s", (secs / 60.0).floor(), secs % 60.0)
    } else {
        format!("{:.0}h {:02.0}m", (secs / 3600.0).floor(), (secs % 3600.0) / 60.0)
    }
}

/// Build progress bar for the HNSW index construction.
/// Uses a log-factorial work model for accurate ETA estimation.
pub fn logfacto_progress_bar(n: usize, msg: impl Into<String>) -> ProgressBar {
    let total_work = log_factorial(n as f64);
    let pb = ProgressBar::new(n as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg} {bar:40.cyan/white.dim} \
             {pos}/{len} ETA {log_eta} {elapsed_fmt:.white.dim}",
        )
        .unwrap()
        .progress_chars("━━─")
        .with_key("elapsed_fmt", |state: &ProgressState, w: &mut dyn FmtWrite| {
            let s = state.elapsed().as_secs();
            write!(w, "[{:02}:{:02}:{:02}]", s / 3600, (s % 3600) / 60, s % 60).unwrap();
        })
        .with_key("log_eta", move |state: &ProgressState, w: &mut dyn FmtWrite| {
            let i = state.pos() as f64;
            let elapsed = state.elapsed().as_secs_f64();
            if i < 2.0 || elapsed < 0.5 {
                write!(w, "estimating…").unwrap();
                return;
            }
            let fraction = log_factorial(i) / total_work;
            if fraction <= 0.0 || fraction >= 1.0 {
                write!(w, "-").unwrap();
                return;
            }
            let eta = elapsed * (1.0 - fraction) / fraction;
            write!(w, "{}", format_eta(eta)).unwrap();
        }),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message(msg.into());
    pb
}

// ── Progress bars ─────────────────────────────────────────────────────────────

/// Format a count with SI suffix: `1_200` → `"1.2k"`, `5_500_000` → `"5.5M"`, etc.
pub fn si_num(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Create a determinate progress bar with a leading message.
/// Position and length are displayed with SI suffixes (e.g. `1.2k`, `5.5M`).
pub fn linear_progress_bar(len: u64, msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg} {bar:40.cyan/white.dim} \
             {si_pos}/{si_len} ETA {eta} {elapsed_fmt:.white.dim}",
        )
        .unwrap()
        .progress_chars("━━─")
        .with_key("si_pos", |state: &ProgressState, w: &mut dyn FmtWrite| {
            write!(w, "{}", si_num(state.pos())).unwrap();
        })
        .with_key("si_len", move |_state: &ProgressState, w: &mut dyn FmtWrite| {
            write!(w, "{}", si_num(len)).unwrap();
        })
        .with_key("elapsed_fmt", |state: &ProgressState, w: &mut dyn FmtWrite| {
            let s = state.elapsed().as_secs();
            write!(w, "[{:02}:{:02}:{:02}]", s / 3600, (s % 3600) / 60, s % 60).unwrap();
        })
    );
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Create an indeterminate spinner with a leading message.
pub fn spinner(msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(spinner_style());
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Finish a progress bar / spinner with a success message.
pub fn finish_success(pb: &ProgressBar, msg: impl Into<String>) {
    pb.finish_and_clear();
    success(&msg.into());
}

/// Finish a progress bar / spinner with an error message, then exit the process.
pub fn finish_error(pb: &ProgressBar, msg: impl Into<String>) -> ! {
    pb.finish_and_clear();
    error(&msg.into());
    std::process::exit(1);
}
