use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod cli;

use clap::{CommandFactory, Parser};
use cli::{Cli, Command, renderer::{render_help, build_usage}};

fn try_render_help(argv: &[String]) -> bool {
    let has_help = argv.iter().any(|a| a == "--help" || a == "-h");
    if !has_help {
        return false;
    }
    let use_long = argv.iter().any(|a| a == "--help");
    let mut cmd = Cli::command();
    cmd.build();
    // Walk positional args to reach the deepest matching subcommand
    let mut current: &clap::Command = &cmd;
    for arg in argv {
        if arg.starts_with('-') {
            break;
        }
        match current.find_subcommand(arg.as_str()) {
            Some(sub) => current = sub,
            None => break,
        }
    }
    if let Some(help) = render_help(current, use_long) {
        print!("{help}");
    }
    true
}
fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if try_render_help(&argv) {
        std::process::exit(0);
    }

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            use clap::error::ErrorKind;
            match e.kind() {
                // --help / --version: let clap render its (now themed) output
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                    let _ = e.print();
                    std::process::exit(0);
                }
                // Real errors: route through the display theme
                _ => {
                    let raw = e.to_string();
                    let mut lines = raw.lines();
                    let first = lines.next().unwrap_or("");
                    let msg = first.strip_prefix("error: ").unwrap_or(first);

                    // Collect tip from remaining lines; discard Usage and "For more information"
                    let mut tip_msg: Option<String> = None;
                    for line in lines.filter(|l| !l.is_empty()) {
                        let t = line.trim();
                        if let Some(rest) = t.strip_prefix("tip:") {
                            tip_msg = Some(rest.trim().to_owned());
                        }
                    }

                    // Error panel
                    cli::display::cli_error(msg);

                    // Usage via our renderer, walking to the failing subcommand
                    let mut cmd = Cli::command();
                    cmd.build();
                    let mut current: &clap::Command = &cmd;
                    for arg in &argv {
                        if arg.starts_with('-') { break; }
                        match current.find_subcommand(arg.as_str()) {
                            Some(sub) => current = sub,
                            None => break,
                        }
                    }
                    println!();
                    cli::display::usage(&build_usage(current));

                    // Tip panel (if present)
                    if let Some(tip) = tip_msg {
                        println!();
                        cli::display::tip(&tip);
                    }

                    std::process::exit(2);
                }
            }
        }
    };
    match cli.command {
        Command::Split(args) => args.run(),
        Command::Cluster(args) => args.run(),
        Command::Knn(args) => args.run(),
        Command::Index(args) => args.run(),
        Command::Edge(args) => args.run(),
        Command::InstallAutocomplete { shell } => cli::autocomplete::install(shell),
    }
}
