## 19. HTML/SVG Export

> Source: `rich/console.py` (export_html, export_svg, save_html, save_svg methods), `rich/_export_format.py`, `rich/terminal_theme.py`, `rich/style.py` (get_html_style)

### 19.1 Overview

Rich can export recorded console output to HTML and SVG formats, preserving colors, styles, and formatting. This enables sharing Rich output as static documents, embedding in web pages, or generating terminal screenshots.

**Implementation note (Rust):** `Console::export_html` / `Console::export_svg` mirror Python Rich's
HTML/SVG exporters, including the Rich template formats and optional terminal-window chrome.
Advanced knobs are exposed via `Console::export_html_with_options(...)` and
`Console::export_svg_with_options(...)`.

**Requirements:**
- Console must be created with `record=True` to capture output
- Export reads from internal `_record_buffer` which stores all printed segments
- Buffer can optionally be cleared after export

### 19.2 TerminalTheme

Color themes define how ANSI colors map to RGB values for export:

```rust
struct TerminalTheme {
    background_color: ColorTriplet,   // Default background
    foreground_color: ColorTriplet,   // Default text color
    ansi_colors: Palette,             // 16 ANSI colors (8 normal + 8 bright)
}
```

**Constructor:**
```rust
impl TerminalTheme {
    fn new(
        background: (u8, u8, u8),
        foreground: (u8, u8, u8),
        normal: [(u8, u8, u8); 8],    // Colors 0-7 (black, red, green, yellow, blue, magenta, cyan, white)
        bright: Option<[(u8, u8, u8); 8]>,  // Colors 8-15, defaults to normal if None
    ) -> Self;
}
```

**Built-in Themes:**

| Theme                    | Background     | Foreground     | Use Case                  |
|-------------------------|----------------|----------------|---------------------------|
| `DEFAULT_TERMINAL_THEME`| White          | Black          | Light HTML export         |
| `MONOKAI`               | Dark (#0C0C0C) | Light (#D9D9D9)| Dark theme export         |
| `DIMMED_MONOKAI`        | Dark (#191919) | Muted          | Subdued dark export       |
| `NIGHT_OWLISH`          | White          | Dark           | Night Owl-inspired        |
| `SVG_EXPORT_THEME`      | Dark (#292929) | Light (#C5C8C6)| Default for SVG export    |

### 19.3 HTML Export

#### 19.3.1 API

```rust
impl Console {
    fn export_html(
        &self,
        theme: Option<&TerminalTheme>,  // Default: DEFAULT_TERMINAL_THEME
        clear: bool,                     // Clear buffer after export (default: true)
        code_format: Option<&str>,       // Custom template (default: CONSOLE_HTML_FORMAT)
        inline_styles: bool,             // Inline vs stylesheet styles (default: false)
    ) -> String;

    fn save_html(
        &self,
        path: &Path,
        theme: Option<&TerminalTheme>,
        clear: bool,
        code_format: &str,
        inline_styles: bool,
    ) -> io::Result<()>;
}
```

#### 19.3.2 HTML Template

Default `CONSOLE_HTML_FORMAT`:

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
{stylesheet}
body {
    color: {foreground};
    background-color: {background};
}
</style>
</head>
<body>
    <pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
        <code style="font-family:inherit">{code}</code>
    </pre>
</body>
</html>
```

**Template Variables:**
- `{stylesheet}` - CSS rules (when `inline_styles=false`)
- `{foreground}` - Theme foreground hex color
- `{background}` - Theme background hex color
- `{code}` - Rendered HTML content

#### 19.3.3 Style Modes

**Inline Styles (`inline_styles=true`):**
```html
<span style="color: #ff0000; font-weight: bold">text</span>
```
- Larger file size
- Easier to copy/paste fragments
- Each styled segment gets inline CSS

**Stylesheet Mode (`inline_styles=false`):**
```html
<style>
.r1 {color: #ff0000; font-weight: bold}
.r2 {color: #00ff00}
</style>
...
<span class="r1">text</span>
```
- Smaller file size
- Classes named `.r1`, `.r2`, etc.
- Styles deduplicated in stylesheet

#### 19.3.4 Style to CSS Conversion

The `get_html_style` method converts Rich styles to CSS:

```rust
impl Style {
    fn get_html_style(&self, theme: &TerminalTheme) -> String {
        let mut css = Vec::new();

        // Handle reverse (swap fg/bg)
        let (color, bgcolor) = if self.reverse {
            (self.bgcolor, self.color)
        } else {
            (self.color, self.bgcolor)
        };

        // Dim: blend foreground toward background
        let color = if self.dim {
            let fg = color.unwrap_or(theme.foreground_color);
            Some(blend_rgb(fg, theme.background_color, 0.5))
        } else {
            color
        };

        // Foreground color
        if let Some(c) = color {
            let hex = c.get_truecolor(theme).hex();
            css.push(format!("color: {}", hex));
            css.push(format!("text-decoration-color: {}", hex));
        }

        // Background color
        if let Some(c) = bgcolor {
            let hex = c.get_truecolor(theme).hex();
            css.push(format!("background-color: {}", hex));
        }

        // Text attributes
        if self.bold { css.push("font-weight: bold".into()); }
        if self.italic { css.push("font-style: italic".into()); }
        if self.underline { css.push("text-decoration: underline".into()); }
        if self.strike { css.push("text-decoration: line-through".into()); }
        if self.overline { css.push("text-decoration: overline".into()); }

        css.join("; ")
    }
}
```

#### 19.3.5 Link Handling

Hyperlinks are preserved as HTML `<a>` tags:

```rust
if let Some(link) = &style.link {
    if inline_styles {
        format!(r#"<a href="{}">{}</a>"#, link, text)
    } else {
        format!(r#"<a class="r{}" href="{}">{}</a>"#, class_num, link, text)
    }
}
```

### 19.4 SVG Export

#### 19.4.1 API

```rust
impl Console {
    fn export_svg(
        &self,
        title: &str,                     // Tab title (default: "Rich")
        theme: Option<&TerminalTheme>,   // Default: SVG_EXPORT_THEME
        clear: bool,                     // Clear buffer (default: true)
        code_format: &str,               // SVG template (default: CONSOLE_SVG_FORMAT)
        font_aspect_ratio: f64,          // Width/height ratio (default: 0.61 for Fira Code)
        unique_id: Option<&str>,         // CSS/element ID prefix (default: computed)
    ) -> String;

    fn save_svg(
        &self,
        path: &Path,
        title: &str,
        theme: Option<&TerminalTheme>,
        clear: bool,
        code_format: &str,
        font_aspect_ratio: f64,
        unique_id: Option<&str>,
    ) -> io::Result<()>;
}
```

#### 19.4.2 SVG Structure

The SVG output creates a terminal-style window with:

1. **Window Chrome:** Rounded rectangle with macOS-style traffic lights (red/yellow/green circles)
2. **Title Bar:** Centered title text
3. **Content Area:** Clipped region containing text
4. **Text Matrix:** Positioned `<text>` elements for each styled segment
5. **Backgrounds:** `<rect>` elements behind text with background colors

```
┌──────────────────────────────────────┐
│ 🔴 🟡 🟢        Rich                 │  ← Chrome + Title
├──────────────────────────────────────┤
│                                      │
│  [Rendered terminal content here]    │  ← Matrix (clipped)
│                                      │
└──────────────────────────────────────┘
```

#### 19.4.3 SVG Template Variables

```rust
struct SvgTemplateVars {
    unique_id: String,          // Prefix for CSS classes and IDs
    char_width: f64,            // Character width in pixels
    char_height: f64,           // Character height (default: 20)
    line_height: f64,           // Line height (char_height * 1.22)
    terminal_width: f64,        // Content area width
    terminal_height: f64,       // Content area height
    width: f64,                 // Total SVG width
    height: f64,                // Total SVG height
    terminal_x: f64,            // Content X offset
    terminal_y: f64,            // Content Y offset
    styles: String,             // Generated CSS rules
    chrome: String,             // Window decoration SVG
    backgrounds: String,        // Background rects
    matrix: String,             // Text elements
    lines: String,              // ClipPath definitions
}
```

#### 19.4.4 Font Configuration

Default uses Fira Code with web font fallback:

```css
@font-face {
    font-family: "Fira Code";
    src: local("FiraCode-Regular"),
         url("https://cdnjs.cloudflare.com/...") format("woff2");
    font-weight: 400;
}
@font-face {
    font-family: "Fira Code";
    src: local("FiraCode-Bold"),
         url("https://cdnjs.cloudflare.com/...") format("woff2");
    font-weight: 700;
}
```

**Font Aspect Ratio:** The `font_aspect_ratio` (default 0.61) determines character positioning:
```rust
let char_width = char_height * font_aspect_ratio;  // 20 * 0.61 = 12.2px
```

#### 19.4.5 Text Positioning

Each text segment is positioned precisely:

```rust
fn render_text_element(
    text: &str,
    style: &Style,
    x: usize,      // Character column
    y: usize,      // Line number
    unique_id: &str,
    class_name: &str,
    char_width: f64,
    char_height: f64,
    line_height: f64,
) -> String {
    format!(
        r#"<text class="{}-{}" x="{}" y="{}" textLength="{}" clip-path="url(#{}-line-{})">{}</text>"#,
        unique_id,
        class_name,
        x as f64 * char_width,
        y as f64 * line_height + char_height,
        char_width * text.len() as f64,
        unique_id,
        y,
        escape_text(text)
    )
}
```

#### 19.4.6 Background Rectangles

Styled backgrounds are rendered as `<rect>` elements:

```rust
fn render_background(
    x: usize,
    y: usize,
    width: usize,
    color: &str,
    char_width: f64,
    line_height: f64,
) -> String {
    format!(
        r#"<rect fill="{}" x="{}" y="{}" width="{}" height="{}" shape-rendering="crispEdges"/>"#,
        color,
        x as f64 * char_width,
        y as f64 * line_height + 1.5,
        char_width * width as f64,
        line_height + 0.25
    )
}
```

#### 19.4.7 Unique ID Generation

When not provided, `unique_id` is computed from content hash:

```rust
fn compute_unique_id(segments: &[Segment], title: &str) -> String {
    let content = segments.iter()
        .map(|s| format!("{:?}", s))
        .collect::<String>();
    let hash = adler32(&[content.as_bytes(), title.as_bytes()].concat());
    format!("terminal-{}", hash)
}
```

#### 19.4.8 Style to SVG CSS

SVG uses `fill` instead of `color`:

```rust
fn get_svg_style(&self, theme: &TerminalTheme) -> String {
    let mut css = Vec::new();

    let (color, bgcolor) = if self.reverse {
        (self.bgcolor, self.color)
    } else {
        (self.color, self.bgcolor)
    };

    // Dim: blend toward background
    let color = if self.dim {
        blend_rgb(color, bgcolor, 0.4)
    } else {
        color
    };

    css.push(format!("fill: {}", color.hex()));

    if self.bold { css.push("font-weight: bold".into()); }
    if self.italic { css.push("font-style: italic".into()); }
    if self.underline { css.push("text-decoration: underline".into()); }
    if self.strike { css.push("text-decoration: line-through".into()); }

    css.join(";")
}
```

### 19.5 Segment Processing

Both export methods process segments similarly:

1. **Filter Control:** Remove control segments (cursor movement, etc.)
2. **Simplify:** Merge adjacent segments with identical styles
3. **Split Lines:** Break into lines for SVG row positioning
4. **Escape Text:** Convert special chars (`<`, `>`, `&`, spaces → `&#160;`)

```rust
fn process_for_export(buffer: &[Segment]) -> Vec<Segment> {
    Segment::simplify(
        Segment::filter_control(buffer.iter().cloned())
    ).collect()
}
```

### 19.6 Layout Constants (SVG)

```rust
// Character dimensions
const CHAR_HEIGHT: f64 = 20.0;
const LINE_HEIGHT_FACTOR: f64 = 1.22;

// Margins (around entire SVG)
const MARGIN_TOP: f64 = 1.0;
const MARGIN_RIGHT: f64 = 1.0;
const MARGIN_BOTTOM: f64 = 1.0;
const MARGIN_LEFT: f64 = 1.0;

// Padding (inside terminal window)
const PADDING_TOP: f64 = 40.0;    // Space for title bar
const PADDING_RIGHT: f64 = 8.0;
const PADDING_BOTTOM: f64 = 8.0;
const PADDING_LEFT: f64 = 8.0;

// Window chrome
const CORNER_RADIUS: f64 = 8.0;
const TRAFFIC_LIGHT_RADIUS: f64 = 7.0;
const TRAFFIC_LIGHT_SPACING: f64 = 22.0;
```

### 19.7 Edge Cases

1. **Empty buffer:** Exports minimal valid HTML/SVG with just theme colors
2. **Control characters:** Filtered out before export
3. **Very long lines:** Clipped to console width in SVG
4. **Unicode width:** `cell_len()` used for proper character positioning
5. **Missing theme:** Falls back to `DEFAULT_TERMINAL_THEME` (HTML) or `SVG_EXPORT_THEME` (SVG)
6. **Concurrent access:** `_record_buffer_lock` protects buffer during export
7. **Whitespace-only text:** Skipped in SVG matrix (backgrounds still rendered)
8. **HTML special chars:** Properly escaped (`<` → `&lt;`, etc.)

### 19.8 Rust Implementation Notes

```rust
// Re-export format templates
pub const CONSOLE_HTML_FORMAT: &str = include_str!("html_template.html");
pub const CONSOLE_SVG_FORMAT: &str = include_str!("svg_template.svg");

// Theme presets
pub static DEFAULT_TERMINAL_THEME: Lazy<TerminalTheme> = Lazy::new(|| {
    TerminalTheme::new(
        (255, 255, 255),  // white background
        (0, 0, 0),        // black foreground
        STANDARD_NORMAL_COLORS,
        Some(STANDARD_BRIGHT_COLORS),
    )
});

pub static SVG_EXPORT_THEME: Lazy<TerminalTheme> = Lazy::new(|| {
    TerminalTheme::new(
        (41, 41, 41),      // dark background
        (197, 200, 198),   // light foreground
        SVG_NORMAL_COLORS,
        Some(SVG_BRIGHT_COLORS),
    )
});
```

---
