## 7. Console

> Source: `rich/console.py` (2680 lines)

### 7.1 ConsoleDimensions

```rust
struct ConsoleDimensions {
    width: usize,   // Console width in cells
    height: usize,  // Console height in rows
}
```

### 7.2 ConsoleOptions Data Structure

```rust
/// Options passed to renderables during rendering
struct ConsoleOptions {
    size: ConsoleDimensions,          // Terminal dimensions
    legacy_windows: bool,             // Using legacy Windows console
    min_width: usize,                 // Minimum width constraint
    max_width: usize,                 // Maximum width constraint
    is_terminal: bool,                // Output is a terminal (vs file/pipe)
    encoding: String,                 // Output encoding (e.g., "utf-8")
    max_height: usize,                // Maximum height for rendering
    justify: Option<JustifyMethod>,   // Default justification
    overflow: Option<OverflowMethod>, // Default overflow handling
    no_wrap: Option<bool>,            // Default no_wrap setting
    highlight: Option<bool>,          // Enable highlighting
    markup: Option<bool>,             // Parse markup in strings
    height: Option<usize>,            // Explicit height override
}

impl ConsoleOptions {
    /// Create new options with different max_width
    fn update_width(&self, width: usize) -> Self {
        ConsoleOptions {
            max_width: width.min(self.max_width),
            ..self.clone()
        }
    }

    /// Create options for rendering within container (reduces width)
    fn update_dimensions(&self, width: usize, height: usize) -> Self {
        ConsoleOptions {
            max_width: width.min(self.max_width),
            height: Some(height),
            ..self.clone()
        }
    }
}
```

### 7.3 Console Structure

```rust
struct Console {
    // Configuration
    color_system: Option<ColorSystem>,  // None = auto-detect
    force_terminal: Option<bool>,       // Force terminal mode
    tab_size: usize,                    // Tab expansion (default 8)
    record: bool,                       // Buffer output for export
    markup: bool,                       // Parse markup by default
    emoji: bool,                        // Enable emoji rendering
    highlight: bool,                    // Enable syntax highlighting
    width: Option<usize>,               // Override width
    height: Option<usize>,              // Override height
    safe_box: bool,                     // Use ASCII-safe box chars

    // State
    file: Box<dyn Write>,               // Output stream
    buffer: Vec<Segment>,               // Recording buffer
    is_terminal: bool,                  // Cached terminal detection
    encoding: String,                   // Output encoding
}
```

### 7.4 Color System Detection

```rust
fn detect_color_system() -> Option<ColorSystem> {
    // Check NO_COLOR env var (https://no-color.org/)
    if std::env::var("NO_COLOR").is_ok() {
        return None;
    }

    // Check COLORTERM for truecolor
    if let Ok(colorterm) = std::env::var("COLORTERM") {
        if colorterm == "truecolor" || colorterm == "24bit" {
            return Some(ColorSystem::TRUECOLOR);
        }
    }

    // Check TERM for 256 color support
    if let Ok(term) = std::env::var("TERM") {
        if term.contains("256color") || term.contains("256") {
            return Some(ColorSystem::EIGHT_BIT);
        }
        if term == "dumb" {
            return None;
        }
    }

    // Default to standard colors if terminal
    Some(ColorSystem::STANDARD)
}
```

### 7.5 Rendering Pipeline

```rust
impl Console {
    /// Main print method
    fn print(&mut self, renderable: impl Renderable, options: PrintOptions) {
        // 1. Collect all renderables
        let renderables = self.collect_renderables(renderable);

        // 2. Create console options
        let console_options = self.make_options();

        // 3. Render to segments
        let segments = self.render(renderables, &console_options);

        // 4. Write or buffer
        if self.record {
            self.buffer.extend(segments);
        } else {
            self.write_segments(segments);
        }
    }

    /// Collect renderables, handling strings and conversions
    fn collect_renderables(&self, obj: impl Renderable) -> Vec<Box<dyn Renderable>> {
        // If object implements __rich_console__, use it
        // If object implements __rich__, convert to Text
        // If string, convert to Text (with optional markup parsing)
    }

    /// Render all objects to flat segment list
    fn render(&self, renderables: Vec<Box<dyn Renderable>>, options: &ConsoleOptions) -> Vec<Segment> {
        let mut result = Vec::new();

        for renderable in renderables {
            // Call rich_console to get segments/nested renderables
            for item in renderable.rich_console(self, options) {
                match item {
                    RenderItem::Segment(seg) => result.push(seg),
                    RenderItem::Renderable(nested) => {
                        // Recursive render
                        result.extend(self.render(vec![nested], options));
                    }
                }
            }
        }

        result
    }

    /// Write segments to output with ANSI codes
    fn write_segments(&mut self, segments: Vec<Segment>) {
        let mut current_style = Style::null();
        let color_system = self.color_system.unwrap_or(ColorSystem::STANDARD);

        for segment in segments {
            if segment.is_control() {
                // Handle control codes
                self.write_control(&segment);
                continue;
            }

            let style = segment.style.unwrap_or_default();

            // Generate style transition
            if style != current_style {
                // Reset then apply new style
                if !current_style.is_null() {
                    write!(self.file, "\x1b[0m").ok();
                }
                if !style.is_null() {
                    let codes = style.make_ansi_codes(color_system);
                    write!(self.file, "\x1b[{}m", codes).ok();
                }
                current_style = style;
            }

            // Write text
            write!(self.file, "{}", segment.text).ok();
        }

        // Reset at end
        if !current_style.is_null() {
            write!(self.file, "\x1b[0m").ok();
        }
    }
}
```

### 7.6 render_lines Helper

```rust
/// Render to list of lines, each line being a list of segments
fn render_lines(
    &self,
    renderable: &dyn Renderable,
    options: &ConsoleOptions,
    style: Option<&Style>,
    pad: bool,
    new_lines: bool,
) -> Vec<Vec<Segment>> {
    let segments = self.render(vec![renderable], options);

    // Split into lines
    let mut lines = Segment::split_lines(segments.into_iter());

    // Adjust each line to width
    if pad || options.max_width > 0 {
        for line in &mut lines {
            *line = Segment::adjust_line_length(
                std::mem::take(line),
                options.max_width,
                style.cloned(),
                pad,
            );
        }
    }

    // Add newlines if requested
    if new_lines {
        for line in &mut lines {
            line.push(Segment::line());
        }
    }

    lines
}
```

---
