## 15. Exclusions and Not-Yet-Implemented Features

This section is the authoritative scope boundary. It separates **out-of-scope**
features (not planned) from **planned-but-not-yet-implemented** features.

### 15.1 Out of Scope (Not Planned)

| Feature | Reason |
|---------|--------|
| Jupyter/IPython integration | Python-specific |
| Legacy Windows (cmd.exe) | Use modern VT sequences via crossterm |

### 15.2 Implemented (With Notes)

| Feature | Status |
|---------|--------|
| Theme + named styles | Implemented (`Theme`, `Console::get_style`, `.ini` loading via `Theme::read`) |
| Pretty / Inspect | Implemented (`renderables::Pretty`, `renderables::Inspect`, `renderables::inspect`; `Debug`-based output + explicit, documented extraction rules) |
| Traceback rendering | Implemented (`renderables::Traceback`, `Console::print_exception`; explicit frames for deterministic fixtures; optional `Traceback::capture()` via `backtrace` feature; code context via `extra_lines` + `source_context` or filesystem source) |
| Live display (`Live`) | Implemented (process-wide stdout/stderr redirection in interactive terminals; no Jupyter integration) |
| Layout engine (`Layout`) | Implemented (ratio splits + named lookup; no render-map caching) |
| Logging handler integration | Implemented (`RichLogger` for `log` crate; optional Rich-style tracebacks for error logs) |
| Console export (HTML/SVG) | Implemented (Rich-style templates + optional window chrome; `export_html_with_options` / `export_svg_with_options` for advanced knobs) |

### 15.3 Implemented (No Longer Excluded)

The following were previously listed as Phase 2+ items and are **now implemented**
in `rich_rust` (some behind feature flags):

- Progress bars & spinners (`renderables::progress`)
- Emoji code replacement (`:name:`) and `Emoji` renderable (`emoji`, `renderables::emoji`)
- Syntax highlighting (feature `syntax`, `renderables::syntax`)
- Markdown rendering (feature `markdown`, `renderables::markdown`)
- JSON pretty-printing (feature `json`, `renderables::json`)
- Traceback rendering (`renderables::traceback`)

---
