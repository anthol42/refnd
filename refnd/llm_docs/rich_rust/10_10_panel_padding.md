## 10. Panel & Padding

> Source: `rich/panel.py` (317 lines), `rich/padding.py` (141 lines)

### 10.1 Padding Data Structure

```rust
/// CSS-style padding values
struct PaddingDimensions {
    top: usize,
    right: usize,
    bottom: usize,
    left: usize,
}

impl PaddingDimensions {
    /// Parse CSS-style padding specification
    /// 1 value:  (all,)        -> all sides equal
    /// 2 values: (vert, horiz) -> top/bottom, left/right
    /// 4 values: (top, right, bottom, left) -> individual sides
    fn unpack(pad: impl Into<PaddingInput>) -> Self {
        match pad.into() {
            PaddingInput::Single(n) =>
                PaddingDimensions { top: n, right: n, bottom: n, left: n },
            PaddingInput::Two(v, h) =>
                PaddingDimensions { top: v, right: h, bottom: v, left: h },
            PaddingInput::Four(t, r, b, l) =>
                PaddingDimensions { top: t, right: r, bottom: b, left: l },
        }
    }
}
```

### 10.2 Padding Renderable

```rust
struct Padding {
    renderable: Box<dyn Renderable>,
    pad: PaddingDimensions,
    style: Style,
    expand: bool,
}

impl Padding {
    /// Create indented padding (left indent only)
    fn indent(renderable: impl Renderable, level: usize) -> Self {
        Padding {
            renderable: Box::new(renderable),
            pad: PaddingDimensions { top: 0, right: 0, bottom: 0, left: level },
            style: Style::null(),
            expand: true,
        }
    }
}

impl Renderable for Padding {
    fn rich_console(&self, console: &Console, options: &ConsoleOptions) -> Vec<RenderItem> {
        let mut segments = Vec::new();
        let width = options.max_width;

        // Calculate inner width
        let inner_width = width
            .saturating_sub(self.pad.left)
            .saturating_sub(self.pad.right);

        // Create inner options
        let inner_options = options.update_width(inner_width);

        // Create padding strings
        let left_pad = " ".repeat(self.pad.left);
        let right_pad = " ".repeat(self.pad.right);
        let blank_line = " ".repeat(width);

        // Top padding
        for _ in 0..self.pad.top {
            segments.push(Segment::new(&blank_line, Some(self.style.clone())));
            segments.push(Segment::line());
        }

        // Render inner content
        let inner_lines = console.render_lines(
            &*self.renderable,
            &inner_options,
            Some(&self.style),
            self.expand,
            false,
        );

        for line in inner_lines {
            // Left padding
            segments.push(Segment::new(&left_pad, Some(self.style.clone())));

            // Content
            segments.extend(line);

            // Right padding
            segments.push(Segment::new(&right_pad, Some(self.style.clone())));
            segments.push(Segment::line());
        }

        // Bottom padding
        for _ in 0..self.pad.bottom {
            segments.push(Segment::new(&blank_line, Some(self.style.clone())));
            segments.push(Segment::line());
        }

        segments.into_iter().map(RenderItem::Segment).collect()
    }
}
```

### 10.3 Panel Data Structure

```rust
struct Panel {
    renderable: Box<dyn Renderable>,
    box_style: Box,
    safe_box: Option<bool>,
    expand: bool,
    style: Style,
    border_style: Style,
    width: Option<usize>,
    height: Option<usize>,
    padding: PaddingDimensions,
    highlight: bool,

    // Title/subtitle
    title: Option<Text>,
    title_align: JustifyMethod,
    subtitle: Option<Text>,
    subtitle_align: JustifyMethod,
}

impl Panel {
    /// Create panel that fits content width
    fn fit(
        renderable: impl Renderable,
        box_style: Box,
        padding: impl Into<PaddingInput>,
    ) -> Self {
        Panel {
            renderable: Box::new(renderable),
            box_style,
            padding: PaddingDimensions::unpack(padding),
            expand: false,  // Key difference: don't expand
            ..Default::default()
        }
    }

    /// Process title text
    fn make_title(&self, text: &Text, width: usize) -> Text {
        let mut title = text.clone();
        title.truncate(width.saturating_sub(4), OverflowMethod::Ellipsis, false);
        title.plain = format!(" {} ", title.plain);  // Add surrounding spaces
        title
    }
}
```

### 10.4 Panel Rendering

```rust
impl Renderable for Panel {
    fn rich_console(&self, console: &Console, options: &ConsoleOptions) -> Vec<RenderItem> {
        let safe_box = self.safe_box.unwrap_or(console.safe_box);
        let box_style = if safe_box {
            self.box_style.substitute_ascii()
        } else {
            self.box_style.clone()
        };

        // Calculate dimensions
        let width = if self.expand {
            options.max_width
        } else if let Some(w) = self.width {
            w
        } else {
            // Measure content
            let inner_options = options.update_width(options.max_width.saturating_sub(4)); // 2 borders + 2 min padding
            let measurement = Measurement::get(console, &inner_options, &*self.renderable);
            measurement.maximum + 4
        };

        let inner_width = width.saturating_sub(2); // Minus border characters
        let content_width = inner_width
            .saturating_sub(self.padding.left)
            .saturating_sub(self.padding.right);

        // Render content
        let content_options = options.update_dimensions(content_width, self.height.unwrap_or(usize::MAX));
        let content_lines = console.render_lines(
            &*self.renderable,
            &content_options,
            None,
            true,
            false,
        );

        let mut segments = Vec::new();

        // Top border with optional title
        let top_border = box_style.get_row(&[inner_width], RowLevel::Top, true);
        if let Some(title) = &self.title {
            let title_text = self.make_title(title, inner_width);
            let title_segments = title_text.render(console, "");

            // Insert title into top border at appropriate position
            segments.extend(self.insert_title_into_border(&top_border, &title_segments, self.title_align, &self.border_style));
        } else {
            segments.push(Segment::new(&top_border, Some(self.border_style.clone())));
        }
        segments.push(Segment::line());

        // Content lines with borders
        let left_pad = " ".repeat(self.padding.left);
        let right_pad = " ".repeat(self.padding.right);

        // Top inner padding
        for _ in 0..self.padding.top {
            segments.push(Segment::new(&box_style.head[0].to_string(), Some(self.border_style.clone())));
            segments.push(Segment::new(&" ".repeat(inner_width), Some(self.style.clone())));
            segments.push(Segment::new(&box_style.head[3].to_string(), Some(self.border_style.clone())));
            segments.push(Segment::line());
        }

        // Content
        for line in content_lines {
            segments.push(Segment::new(&box_style.head[0].to_string(), Some(self.border_style.clone())));
            segments.push(Segment::new(&left_pad, Some(self.style.clone())));
            segments.extend(line);
            segments.push(Segment::new(&right_pad, Some(self.style.clone())));
            segments.push(Segment::new(&box_style.head[3].to_string(), Some(self.border_style.clone())));
            segments.push(Segment::line());
        }

        // Bottom inner padding
        for _ in 0..self.padding.bottom {
            segments.push(Segment::new(&box_style.head[0].to_string(), Some(self.border_style.clone())));
            segments.push(Segment::new(&" ".repeat(inner_width), Some(self.style.clone())));
            segments.push(Segment::new(&box_style.head[3].to_string(), Some(self.border_style.clone())));
            segments.push(Segment::line());
        }

        // Bottom border with optional subtitle
        let bottom_border = box_style.get_row(&[inner_width], RowLevel::Bottom, true);
        if let Some(subtitle) = &self.subtitle {
            let subtitle_text = self.make_title(subtitle, inner_width);
            let subtitle_segments = subtitle_text.render(console, "");
            segments.extend(self.insert_title_into_border(&bottom_border, &subtitle_segments, self.subtitle_align, &self.border_style));
        } else {
            segments.push(Segment::new(&bottom_border, Some(self.border_style.clone())));
        }
        segments.push(Segment::line());

        segments.into_iter().map(RenderItem::Segment).collect()
    }
}
```

---
