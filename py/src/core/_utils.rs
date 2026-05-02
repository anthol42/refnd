use std::fmt::{Write as FmtWrite};
use std::time::Duration;
use indicatif::{ProgressBar, ProgressStyle, ProgressState};

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

/// Create a determinate progress bar with a leading message.
/// Position and length are displayed with SI suffixes (e.g. `1.2k`, `5.5M`).
pub fn linear_progress_bar(len: usize, msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new(len as u64);
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
                write!(w, "{}", si_num(len as u64)).unwrap();
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