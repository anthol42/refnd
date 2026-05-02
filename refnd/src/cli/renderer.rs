use clap::{ArgAction, Command};
use rich_rust::console::PrintOptions;
use rich_rust::prelude::*;
use rich_rust::r#box::SIMPLE;
use rich_rust::segment::split_lines;
use rich_rust::terminal::get_terminal_width;
use super::autocomplete;

// ── Console factory ───────────────────────────────────────────────────────────

fn make_console(width: usize) -> Console {
    Console::builder()
        .color_system(ColorSystem::TrueColor)
        .force_terminal(true)
        .width(width)
        .build()
}

fn print_markup(console: &Console, buf: &mut Vec<u8>, text: &str, with_highlight: bool) {
    let _ = console.print_to(
        buf,
        text,
        &PrintOptions::new().with_markup(true).with_highlight(with_highlight).with_end(""),
    );
}

/// Drop trailing lines that are empty (all-whitespace segments or zero segments).
fn trim_trailing_empty_lines(mut lines: Vec<Vec<Segment<'static>>>) -> Vec<Vec<Segment<'static>>> {
    while let Some(last) = lines.last() {
        let is_blank = last.iter().all(|s| s.text.trim().is_empty());
        if is_blank {
            lines.pop();
        } else {
            break;
        }
    }
    lines
}

fn render_panel(panel: &Panel<'_>, console: &Console, buf: &mut Vec<u8>) {
    let segs = panel.render(console.width());
    let _ = console.print_segments_to(buf, &segs);
}

// ── Usage line ────────────────────────────────────────────────────────────────


pub fn build_usage(cmd: &Command) -> String {
    let name = cmd
        .get_bin_name()
        .map(|s| s.to_owned())
        .unwrap_or_else(|| cmd.get_name().to_owned());

    let mut parts = vec![format!("[bold]{name}[/bold]")];

    for p in cmd.get_positionals().filter(|a| !a.is_hide_set()) {
        let vname = value_name(p);
        if p.is_required_set() {
            parts.push(format!("[bold green]<{vname}>[/bold green]"));
        } else {
            parts.push(format!("[[green]{vname}[/green]]"));
        }
    }

    let has_opts = cmd.get_opts().any(|a| !a.is_hide_set());
    if has_opts {
        parts.push("[bold cyan]\\[OPTIONS][/bold cyan]".to_owned());
    }

    let has_subcmds = cmd.get_subcommands().any(|s| !s.is_hide_set());
    if has_subcmds {
        parts.push("[bold cyan]\\[COMMAND][/bold cyan]".to_owned());
    }

    format!("[bold yellow]Usage:[/bold yellow] {}", parts.join(" "))
}

// ── Argument helpers ──────────────────────────────────────────────────────────

fn value_name(arg: &clap::Arg) -> String {
    arg.get_value_names()
        .and_then(|v| v.first())
        .map(|s| s.as_str().to_owned())
        .unwrap_or_else(|| arg.get_id().as_str().to_uppercase())
}

/// Human-readable type label for an argument.
/// Prefers the explicit `value_name` set on the arg; falls back to inspecting
/// the value parser's debug representation for built-in types.
fn type_label(arg: &clap::Arg) -> String {
    if let Some(names) = arg.get_value_names() {
        if let Some(name) = names.first() {
            return name.as_str().to_owned();
        }
    }
    let dbg = format!("{:?}", arg.get_value_parser());
    if dbg.contains("f32") || dbg.contains("f64") {
        "FLOAT"
    } else if dbg.contains("i8")
        || dbg.contains("i16")
        || dbg.contains("i32")
        || dbg.contains("i64")
        || dbg.contains("u8")
        || dbg.contains("u16")
        || dbg.contains("u32")
        || dbg.contains("u64")
        || dbg.contains("usize")
        || dbg.contains("isize")
    {
        "INTEGER"
    } else if dbg.contains("path_buf") || dbg.contains("OsString") {
        "PATH"
    } else {
        "TEXT"
    }
    .to_owned()
}

fn default_annotation(arg: &clap::Arg) -> String {
    let defaults = arg.get_default_values();
    if !defaults.is_empty() {
        let s = defaults
            .iter()
            .filter_map(|v| v.to_str())
            .collect::<Vec<_>>()
            .join(", ");
        return format!(" [dim]\\[default: {s}][/dim]");
    }
    String::new()
}

fn render_enum_variants(arg: &clap::Arg) -> String {
    let variants: Vec<_> = arg.get_possible_values()
        .into_iter()
        .filter(|pv| !pv.is_hide_set())
        .collect();
    if variants.is_empty() {
        return String::new();
    }

    let has_docs = variants.iter().any(|pv| pv.get_help().is_some());

    if !has_docs {
        let inline = variants
            .iter()
            .map(|pv| format!("[green]{}[/green]", pv.get_name()))
            .collect::<Vec<_>>()
            .join(", ");
        return format!("\n[dim]Variants: {inline}[/dim]");
    }

    let lines: String = variants
        .iter()
        .map(|pv| {
            let name = pv.get_name();
            let help = pv.get_help().map(|h| h.to_string()).unwrap_or_default();
            format!("\n[dim green]{name}:[/dim green]\n  {help}")
        })
        .collect();
    lines
}

// ── Arguments panel (positionals) ─────────────────────────────────────────────

fn build_positionals_panel<'a>(args: &[&clap::Arg], width: usize) -> Panel<'a> {
    let inner = width.saturating_sub(4);

    let mut table = Table::new()
        .box_style(&SIMPLE)
        .show_edge(false)
        .show_header(false)
        .show_lines(false)
        .padding(1, 0)
        .with_column(Column::new("").width(3))
        .with_column(Column::new("").no_wrap())
        .with_column(Column::new("").no_wrap())
        .with_column(Column::new(""));

    for arg in args {
        let marker = if arg.is_required_set() {
            "[bold red]*[/bold red]"
        } else {
            " "
        };
        let name = format!("[green]{}[green]", arg.get_id().as_str());
        let type_str = format!("[bold yellow]{}[/bold yellow]", type_label(arg));
        let mut desc = arg
            .get_help()
            .map(|h| h.to_string())
            .unwrap_or_default();
        if arg.is_required_set() {
            desc.push_str(" [dim red]\\[required][/dim red]");
        }
        table.add_row_markup([&marker.to_owned(), &name, &type_str, &desc]);
    }

    let segs = table.render(inner);
    let lines = trim_trailing_empty_lines(split_lines(segs.into_iter()));

    Panel::new(lines)
        .title_from_markup(" [dim]Arguments[/dim] ")
        .title_align(JustifyMethod::Left)
        .border_style(Style::new().dim())
}

// ── Options panel (flags + options) ───────────────────────────────────────────

fn build_options_panel<'a>(
    heading: &str,
    opts: &[&clap::Arg],
    use_long: bool,
    width: usize,
) -> Panel<'a> {
    let inner = width.saturating_sub(4);

    // Pre-compute column widths to prevent rich_rust from wrapping fixed columns.
    let flag_col_width = opts.iter().map(|arg| {
        match arg.get_long() {
            Some(long) => 2 + long.len(),   // "--long"
            None => arg.get_short().map(|_| 2).unwrap_or(0), // "-c"
        }
    }).max().unwrap_or(5);

    let short_col_width = opts.iter().map(|arg| {
        arg.get_short().map(|_| 2).unwrap_or(0) // "-c"
    }).max().unwrap_or(0);

    let type_col_width = opts.iter().map(|arg| {
        let action = arg.get_action();
        let skip = matches!(action,
            ArgAction::SetTrue | ArgAction::SetFalse |
            ArgAction::Help | ArgAction::HelpShort | ArgAction::HelpLong | ArgAction::Version
        );
        if skip { 0 } else { type_label(arg).len() }
    }).max().unwrap_or(0);

    let mut table = Table::new()
        .box_style(&SIMPLE)
        .show_edge(false)
        .show_header(false)
        .show_lines(false)
        .padding(1, 0)
        .with_column(Column::new("").no_wrap().width(flag_col_width))
        .with_column(Column::new("").no_wrap().width(short_col_width))
        .with_column(Column::new("").no_wrap().width(type_col_width))
        .with_column(Column::new(""));

    for arg in opts {
        let action = arg.get_action();
        let is_bool = matches!(action, ArgAction::SetTrue | ArgAction::SetFalse);
        let is_meta = matches!(
            action,
            ArgAction::Help | ArgAction::HelpShort | ArgAction::HelpLong | ArgAction::Version
        );

        // ── Flag column ───────────────────────────────────────────────────────
        let flag_col = match arg.get_long() {
            Some(long) if is_bool => {
                format!("[bold cyan]--{long}[/bold cyan]")
            }
            Some(long) => format!("[bold cyan]--{long}[/bold cyan]"),
            None => {
                // short-only flag
                match arg.get_short() {
                    Some(c) => format!("[bold cyan]-{c}[/bold cyan]"),
                    None => continue,
                }
            }
        };

        // ── Short flag column ─────────────────────────────────────────────────
        let short_col = arg
            .get_short()
            .map(|c| format!("[bold green]-{c}[/bold green]"))
            .unwrap_or_default();

        // ── Type column ───────────────────────────────────────────────────────
        let type_col = if is_meta {
            String::new()
        } else if is_bool {
            "[bold yellow]BOOL[/bold yellow]".into()
        } else {
          format!("[bold yellow]{}[/bold yellow]", type_label(arg))
        };

        // ── Description column ────────────────────────────────────────────────
        let raw_help = if use_long {
            arg.get_long_help()
                .or_else(|| arg.get_help())
                .map(|h| h.to_string())
                .unwrap_or_default()
        } else {
            arg.get_help().map(|h| h.to_string()).unwrap_or_default()
        };

        let annotation = if arg.is_required_set() {
            " [dim]\\[required][/dim]".to_owned()
        } else {
            default_annotation(arg)
        };

        let enum_block = render_enum_variants(arg);
        let desc_col = format!("{raw_help}{annotation}{enum_block}");

        table.add_row_markup([&flag_col, &short_col, &type_col, &desc_col]);
    }

    let segs = table.render(inner);
    let lines = trim_trailing_empty_lines(split_lines(segs.into_iter()));
    let title_markup = format!(" [dim]{heading}[/dim] ");
    Panel::new(lines)
        .title_from_markup(&title_markup)
        .title_align(JustifyMethod::Left)
        .border_style(Style::new().dim())
}

// ── Subcommands panel ─────────────────────────────────────────────────────────

fn build_commands_panel<'a>(cmds: &[&Command], width: usize) -> Panel<'a> {
    let inner = width.saturating_sub(4);

    let mut table = Table::new()
        .box_style(&SIMPLE)
        .show_edge(false)
        .show_header(false)
        .show_lines(false)
        .padding(1, 0)
        .with_column(Column::new("").no_wrap())
        .with_column(Column::new(""));

    for cmd in cmds {
        let name = format!("[bold cyan]{}[/bold cyan]", cmd.get_name());
        let desc = cmd
            .get_about()
            .map(|a| a.to_string())
            .unwrap_or_default();
        table.add_row_markup([&name, &desc]);
    }

    let segs = table.render(inner);
    let lines = trim_trailing_empty_lines(split_lines(segs.into_iter()));

    Panel::new(lines)
        .title_from_markup(" [dim]Commands[/] ")
        .title_align(JustifyMethod::Left)
        .border_style(Style::new().dim())
}

// ── Renderer impl ─────────────────────────────────────────────────────────────


pub fn render_help(cmd: &Command, use_long: bool) -> Option<String> {
        let width = get_terminal_width().max(60).min(220);
        let console = make_console(width);
        let mut buf: Vec<u8> = Vec::new();

        // 1. Usage
        let usage = build_usage(cmd);
        print_markup(&console, &mut buf, &format!("\n {usage}\n"), false);

        // 2. Description
        let about = if use_long {
            cmd.get_long_about().or_else(|| cmd.get_about())
        } else {
            cmd.get_about()
        };
        if let Some(about) = about {
            let text = about.to_string();
            print_markup(&console, &mut buf, "\n", true);
            for line in text.lines() {
                print_markup(&console, &mut buf, &format!(" {line}\n"), true);
            }
        }
        print_markup(&console, &mut buf, "\n", true);

        // 3. Before-help
        let before = if use_long {
            cmd.get_before_long_help().or_else(|| cmd.get_before_help())
        } else {
            cmd.get_before_help()
        };
        if let Some(before) = before {
            let text = before.to_string();
            for line in text.lines() {
                print_markup(&console, &mut buf, &format!(" {line}\n"), true);
            }
            print_markup(&console, &mut buf, "\n", true);
        }

        // 4. Arguments panel (positionals)
        let positionals: Vec<&clap::Arg> = cmd
            .get_positionals()
            .filter(|a| !a.is_hide_set())
            .collect();
        if !positionals.is_empty() {
            let panel = build_positionals_panel(&positionals, width);
            render_panel(&panel, &console, &mut buf);
        }

        // 5. Options panels — one per help heading
        let all_opts: Vec<&clap::Arg> = cmd
            .get_arguments()
            .filter(|a| !a.is_positional() && !a.is_hide_set())
            .collect();

        if !all_opts.is_empty() {
            // Group by help heading; None → "Options"
            let mut groups: Vec<(&str, Vec<&clap::Arg>)> = Vec::new();

            for arg in &all_opts {
                let heading = arg.get_help_heading().unwrap_or("Options");
                if let Some(g) = groups.iter_mut().find(|(h, _)| *h == heading) {
                    g.1.push(arg);
                } else {
                    groups.push((heading, vec![arg]));
                }
            }

            for (heading, opts) in groups {
                let panel = build_options_panel(heading, &opts, use_long, width);
                render_panel(&panel, &console, &mut buf);
            }
        }

        // 6. Subcommands panel
        let subcmds: Vec<&Command> = cmd
            .get_subcommands()
            .filter(|s| !s.is_hide_set())
            .collect();
        if !subcmds.is_empty() {
            let panel = build_commands_panel(&subcmds, width);
            render_panel(&panel, &console, &mut buf);
        }

        // 7. After-help
        let after = if use_long {
            cmd.get_after_long_help().or_else(|| cmd.get_after_help())
        } else {
            cmd.get_after_help()
        };
        if let Some(after) = after {
            let text = after.to_string();
            print_markup(&console, &mut buf, "\n", true);
            for line in text.lines() {
                print_markup(&console, &mut buf, &format!(" {line}\n"), true);
            }
        }

        // 8. Autocomplete tip (shown once until installed)
        if !autocomplete::is_installed() {
            let mut tip_text = Text::new("");
            tip_text.append("Tab-completion is not installed. Run ");
            tip_text.append_styled(
                "refnd install-autocomplete",
                Style::new().bold().color(Color::parse("cyan").unwrap()),
            );
            tip_text.append(" to enable it.");
            let panel = Panel::from_rich_text(&tip_text, width.saturating_sub(2))
                .title_from_markup(" [bold cyan]Tip[/bold cyan] ")
                .title_align(JustifyMethod::Left)
                .border_style(Style::new().dim().color(Color::parse("cyan").unwrap()));
            render_panel(&panel, &console, &mut buf);
        }

        print_markup(&console, &mut buf, "\n", true);

        Some(String::from_utf8(buf).unwrap())
}
