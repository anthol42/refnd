use clap::CommandFactory;
use clap_complete::{generate_to, Shell};
use std::{env, fs, path::PathBuf};
use super::display;

pub fn detect_shell() -> Option<Shell> {
    let shell_path = env::var("SHELL").ok()?;
    let name = PathBuf::from(&shell_path)
        .file_name()?
        .to_string_lossy()
        .to_lowercase();
    match name.as_str() {
        "bash" => Some(Shell::Bash),
        "zsh" => Some(Shell::Zsh),
        "fish" => Some(Shell::Fish),
        "elvish" => Some(Shell::Elvish),
        _ => None,
    }
}

fn completion_dir(shell: Shell) -> Option<PathBuf> {
    let home = PathBuf::from(env::var("HOME").ok()?);
    let dir = if shell == Shell::Bash {
        home.join(".local/share/bash-completion/completions")
    } else if shell == Shell::Zsh {
        home.join(".zfunc")
    } else if shell == Shell::Fish {
        home.join(".config/fish/completions")
    } else {
        return None;
    };
    Some(dir)
}

// Returns the filename clap_complete generates for a given shell.
fn completion_filename(shell: Shell) -> &'static str {
    if shell == Shell::Zsh        { "_refnd"      }
    else if shell == Shell::Fish  { "refnd.fish"  }
    else if shell == Shell::Elvish { "refnd.elv"  }
    else                          { "refnd"        }
}

/// Returns `true` if the completion file already exists (or the shell is unknown).
pub fn is_installed() -> bool {
    let Some(shell) = detect_shell() else { return true };
    let Some(dir)   = completion_dir(shell) else { return true };
    dir.join(completion_filename(shell)).exists()
}

fn post_install_hint(shell: Shell) -> &'static str {
    if shell == Shell::Zsh {
        "Zsh: add to ~/.zshrc if not already present:\n  fpath=(~/.zfunc $fpath)\n  autoload -Uz compinit && compinit"
    } else if shell == Shell::Bash {
        "Bash: restart your shell or run: source ~/.bashrc"
    } else {
        "Restart your shell to activate completions."
    }
}

pub fn install(shell: Option<Shell>) {
    let shell = match shell.or_else(detect_shell) {
        Some(s) => s,
        None => {
            display::error(
                "could not detect shell. Pass a shell name explicitly: \
                 bash, zsh, fish, elvish, powershell."
            );
            std::process::exit(1);
        }
    };

    let dir = match completion_dir(shell) {
        Some(d) => d,
        None => {
            display::error(&format!("no known completion directory for {shell}."));
            std::process::exit(1);
        }
    };

    if let Err(e) = fs::create_dir_all(&dir) {
        display::error(&format!("cannot create '{}': {e}", dir.display()));
        std::process::exit(1);
    }

    let mut cmd = super::Cli::command();
    match generate_to(shell, &mut cmd, "refnd", &dir) {
        Ok(path) => {
            display::success(&format!("autocomplete installed → {}", path.display()));
            display::tip(post_install_hint(shell));
        }
        Err(e) => {
            display::error(&format!("failed to write completion file: {e}"));
            std::process::exit(1);
        }
    }
}
