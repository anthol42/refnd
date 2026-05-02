## 18. Logging Handler Integration

**Implementation note (Rust):** `src/logging.rs` provides `RichLogger`, a `log`-crate
logger with level/time/path formatting, keyword highlighting, and optional Rich-style
tracebacks for `ERROR` logs. An optional `RichTracingLayer` is available behind the
`tracing` feature and uses the same formatting and traceback behavior.

> Source: `rich/logging.py` (298 lines), `rich/_log_render.py` (95 lines)

### 18.1 Overview

Rich provides `RichHandler`, a Python logging handler that renders log records with syntax highlighting, colored log levels, and optional rich tracebacks. In Rust, this integrates with the `log` or `tracing` ecosystems.

### 18.2 RichHandler Constructor

```rust
struct RichHandler {
    // Display configuration
    console: Console,                    // Output console (default: global console)
    show_time: bool,                     // Show time column (default: true)
    omit_repeated_times: bool,           // Skip duplicate times (default: true)
    show_level: bool,                    // Show level column (default: true)
    show_path: bool,                     // Show file:line column (default: true)
    enable_link_path: bool,              // Enable terminal hyperlinks (default: true)

    // Message rendering
    highlighter: Option<Box<dyn Highlighter>>,  // Message highlighter (default: ReprHighlighter)
    markup: bool,                        // Parse Rich markup in messages (default: false)
    keywords: Option<Vec<String>>,       // Words to highlight (default: HTTP methods)
    log_time_format: TimeFormat,         // strftime or callable (default: "[%x %X]")

    // Traceback configuration
    rich_tracebacks: bool,               // Enable rich tracebacks (default: false)
    tracebacks_width: Option<usize>,     // Traceback width (default: None = full)
    tracebacks_code_width: Option<usize>, // Code width (default: 88)
    tracebacks_extra_lines: usize,       // Context lines (default: 3)
    tracebacks_theme: Option<String>,    // Pygments theme override
    tracebacks_word_wrap: bool,          // Wrap long lines (default: true)
    tracebacks_show_locals: bool,        // Show local variables (default: false)
    tracebacks_suppress: Vec<PathBuf>,   // Modules/paths to exclude
    tracebacks_max_frames: usize,        // Max stack frames (default: 100)
    locals_max_length: usize,            // Container abbreviation limit (default: 10)
    locals_max_string: usize,            // String truncation limit (default: 80)
}
```

### 18.3 Default Keywords

Class variable `KEYWORDS` contains HTTP method names for automatic highlighting:

```rust
const KEYWORDS: &[&str] = &[
    "GET", "POST", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE", "PATCH"
];
```

These are highlighted with style `logging.keyword`.

### 18.4 Level Styling

Log levels are styled using semantic style names:

| Level    | Style Name               | Typical Rendering    |
|----------|--------------------------|----------------------|
| DEBUG    | `logging.level.debug`    | Blue, dim            |
| INFO     | `logging.level.info`     | Green                |
| WARNING  | `logging.level.warning`  | Yellow               |
| ERROR    | `logging.level.error`    | Red, bold            |
| CRITICAL | `logging.level.critical` | Red background, bold |

**Implementation:**

```rust
fn get_level_text(&self, level: Level) -> Text {
    let name = level.as_str();
    // Left-justify to 8 characters for alignment
    let padded = format!("{:<8}", name);
    let style_name = format!("logging.level.{}", name.to_lowercase());
    Text::styled(padded, style_name)
}
```

### 18.5 LogRender: Columnar Output

The `LogRender` helper formats log records as a grid table with columns:

```
| TIME       | LEVEL    | MESSAGE                  | PATH:LINE |
|------------|----------|--------------------------|-----------|
| [12:34:56] | INFO     | Server starting...       | main.rs:42|
```

**Column Styles:**
- Time column: `log.time`
- Level column: `log.level` (fixed width 8)
- Message column: `log.message` (ratio=1, overflow=fold)
- Path column: `log.path`

**Grid Construction:**

```rust
fn render_log(
    &self,
    console: &Console,
    renderables: Vec<Box<dyn Renderable>>,
    log_time: Option<DateTime>,
    level: Text,
    path: Option<&str>,
    line_no: Option<u32>,
    link_path: Option<&Path>,
) -> Table {
    let mut grid = Table::grid().padding((0, 1));
    grid.expand = true;

    if self.show_time {
        grid.add_column(Column::new().style("log.time"));
    }
    if self.show_level {
        grid.add_column(Column::new().style("log.level").width(self.level_width));
    }
    grid.add_column(Column::new().ratio(1).style("log.message").overflow(Overflow::Fold));
    if self.show_path && path.is_some() {
        grid.add_column(Column::new().style("log.path"));
    }

    // Build row...
    grid
}
```

### 18.6 Time Format

**Time Display Options:**

1. **strftime string** (default `"[%x %X]"`):
    - `%x` = locale-appropriate date
    - `%X` = locale-appropriate time

2. **Callable**: `fn(DateTime) -> Text` for custom formatting

**Repeated Time Omission:**

When `omit_repeated_times` is true, consecutive identical times are replaced with spaces:

```rust
if log_time_display == self.last_time && self.omit_repeated_times {
    row.push(Text::new(" ".repeat(log_time_display.len())));
} else {
    row.push(log_time_display.clone());
    self.last_time = Some(log_time_display);
}
```

### 18.7 Path Column with Hyperlinks

The path column shows `filename:line` with optional terminal hyperlinks:

```rust
fn render_path(&self, path: &str, line_no: u32, link_path: Option<&Path>) -> Text {
    let mut text = Text::new();

    if let Some(link) = link_path {
        // Terminal hyperlink to file
        text.append(path, Style::new().link(format!("file://{}", link.display())));
    } else {
        text.append(path, Style::default());
    }

    text.append(":", Style::default());

    if let Some(link) = link_path {
        // Hyperlink to specific line
        text.append(
            &line_no.to_string(),
            Style::new().link(format!("file://{}#{}", link.display(), line_no))
        );
    } else {
        text.append(&line_no.to_string(), Style::default());
    }

    text
}
```

### 18.8 Message Rendering Pipeline

1. **Format message** using standard logging formatter
2. **Parse markup** if enabled (per-record override via `record.markup`)
3. **Apply highlighter** (per-record override via `record.highlighter`)
4. **Highlight keywords** from the keywords list

```rust
fn render_message(&self, record: &LogRecord, message: &str) -> Box<dyn Renderable> {
    // Check for per-record markup override
    let use_markup = record.extras.get("markup")
        .and_then(|v| v.as_bool())
        .unwrap_or(self.markup);

    let mut text = if use_markup {
        Text::from_markup(message)
    } else {
        Text::new(message)
    };

    // Apply highlighter (may be overridden per-record)
    let highlighter = record.extras.get("highlighter")
        .and_then(|v| v.as_highlighter())
        .or(self.highlighter.as_ref());

    if let Some(h) = highlighter {
        text = h.highlight(text);
    }

    // Highlight keywords
    if let Some(keywords) = &self.keywords {
        text.highlight_words(keywords, "logging.keyword");
    }

    Box::new(text)
}
```

### 18.9 Rich Tracebacks

**Implementation note (Rust):** `renderables::Traceback` supports deterministic rendering
from explicit frames for conformance/tests, and optional automatic capture via
`Traceback::capture()` when the `backtrace` feature is enabled. Locals rendering is
supported when locals are provided explicitly (`TracebackFrame::locals` +
`Traceback::show_locals(true)`); automatic locals capture is not available in Rust.

When `rich_tracebacks` is enabled and an exception is attached to the record:

```rust
fn emit(&mut self, record: &LogRecord) {
    let traceback = if self.rich_tracebacks && record.exc_info.is_some() {
        let (exc_type, exc_value, exc_tb) = record.exc_info.unwrap();
        Some(Traceback::from_exception(
            exc_type,
            exc_value,
            exc_tb,
            TracebackConfig {
                width: self.tracebacks_width,
                code_width: self.tracebacks_code_width,
                extra_lines: self.tracebacks_extra_lines,
                theme: self.tracebacks_theme.clone(),
                word_wrap: self.tracebacks_word_wrap,
                show_locals: self.tracebacks_show_locals,
                locals_max_length: self.locals_max_length,
                locals_max_string: self.locals_max_string,
                suppress: self.tracebacks_suppress.clone(),
                max_frames: self.tracebacks_max_frames,
            }
        ))
    } else {
        None
    };

    // When traceback exists, message content changes
    let message = if traceback.is_some() {
        record.get_message()  // Raw message without formatter processing
    } else {
        self.format(record)   // Full formatted message
    };

    // Combine message and optional traceback
    let renderables: Vec<Box<dyn Renderable>> = if let Some(tb) = traceback {
        vec![Box::new(message_text), Box::new(tb)]
    } else {
        vec![Box::new(message_text)]
    };

    // Render and output
    let log_output = self.render(record, &renderables);
    self.console.print(log_output);
}
```

### 18.10 NullFile Handling

For environments where stdout/stderr are null (e.g., `pythonw` on Windows):

```rust
fn emit(&mut self, record: &LogRecord) {
    // ... render log_output ...

    if self.console.file().is_null() {
        // Still create the record for compatibility, but don't output
        self.handle_error(record);
    } else {
        if let Err(e) = self.console.print(log_output) {
            self.handle_error(record);
        }
    }
}
```

### 18.11 Per-Record Overrides

Individual log records can override handler settings via extras:

```python
# Python example
log.info("Message with [bold]markup[/bold]", extra={"markup": True})
log.info("Custom highlighting", extra={"highlighter": my_highlighter})
```

**Supported Overrides:**
- `markup: bool` - Enable/disable Rich markup parsing
- `highlighter: Highlighter` - Custom highlighter instance

### 18.12 Rust Integration Considerations

**For `log` crate:**
```rust
use log::{Log, Record, Level, Metadata};

impl Log for RichHandler {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            self.emit(record);
        }
    }

    fn flush(&self) {
        // Console handles buffering
    }
}
```

**For `tracing` crate:**
```rust
use tracing_subscriber::Layer;

impl<S> Layer<S> for RichLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        // Convert tracing event to Rich log output
    }
}
```

### 18.13 Output Format Example

```
[01/24/26 14:30:45] INFO     Server starting...                          main.rs:42
                    INFO     Listening on http://127.0.0.1:8080           main.rs:43
                    WARNING  GET /favicon.ico 404 242                     server.rs:128
                    ERROR    Unable to find 'pomelo' in database!         db.rs:256
```

Note how:
- Time is omitted when repeated (replaced with spaces)
- Level names are left-padded to 8 chars
- HTTP methods (GET) are highlighted with `logging.keyword`
- Each level has distinct styling

### 18.14 Edge Cases

1. **Null console file:** Log record created but no output written
2. **Markup in untrusted logs:** Disable `markup` for third-party libraries
3. **Very long paths:** Use only filename, not full path
4. **Concurrent logging:** Console mutex protects output
5. **Exception without traceback:** Normal message formatting used
6. **Custom time formats:** Callable allows arbitrary Text return
7. **Empty keywords list:** No keyword highlighting applied

---
