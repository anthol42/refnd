## 9. Table Layout

> Source: `rich/table.py` (1006 lines)

### 9.1 Table Data Structures

```rust
/// Single table column definition
struct Column {
    header: Text,                   // Column header text
    footer: Text,                   // Column footer text
    header_style: Style,            // Style for header
    footer_style: Style,            // Style for footer
    style: Style,                   // Style for cell content
    justify: JustifyMethod,         // Cell content justification
    vertical: VerticalAlignMethod,  // Vertical alignment
    overflow: OverflowMethod,       // Overflow handling
    width: Option<usize>,           // Fixed width (cells)
    min_width: Option<usize>,       // Minimum width
    max_width: Option<usize>,       // Maximum width
    ratio: Option<usize>,           // Ratio for flexible sizing
    no_wrap: bool,                  // Disable text wrapping
    // Internal state
    _index: usize,                  // Column index
    _cells: Vec<Box<dyn Renderable>>, // Cells in this column
}

/// Single table row
struct Row {
    style: Style,     // Row-level style
    end_section: bool, // Draw separator after this row
}

/// Single table cell (internal)
struct Cell {
    style: Style,               // Cell-specific style
    renderable: Box<dyn Renderable>,
    vertical: VerticalAlignMethod,
}

/// Table configuration
struct Table {
    columns: Vec<Column>,
    rows: Vec<Row>,
    cells: Vec<Vec<Cell>>,     // cells[row_idx][col_idx]

    // Configuration
    title: Option<Text>,
    caption: Option<Text>,
    width: Option<usize>,       // Fixed table width
    min_width: Option<usize>,
    box_style: Box,             // Box drawing style
    safe_box: Option<bool>,     // Force ASCII boxes
    padding: (usize, usize),    // (horizontal, vertical) cell padding
    collapse_padding: bool,     // Remove padding between cells
    pad_edge: bool,             // Pad outer edges
    expand: bool,               // Expand to fill console width
    show_header: bool,
    show_footer: bool,
    show_edge: bool,            // Show left/right edges
    show_lines: bool,           // Show lines between rows
    leading: usize,             // Extra lines between rows
    style: Style,               // Table-level style
    row_styles: Vec<Style>,     // Alternating row styles
    header_style: Style,
    footer_style: Style,
    border_style: Style,
    title_style: Style,
    caption_style: Style,
    title_justify: JustifyMethod,
    caption_justify: JustifyMethod,
    highlight: bool,
}
```

### 9.2 Column Width Calculation (CRITICAL ALGORITHM)

This is the most complex algorithm in Rich. It determines how to distribute available width among columns.

```rust
fn calculate_column_widths(&self, console: &Console, max_width: usize) -> Vec<usize> {
    // Step 1: Get measurement for each column
    let measurements: Vec<Measurement> = self.columns.iter()
        .map(|col| self.measure_column(console, col, max_width))
        .collect();

    // Step 2: Calculate space needed for borders and padding
    let border_width = if self.show_edge { 2 } else { 0 };
    let padding_width = self.padding.0 * 2 * self.columns.len();
    let separator_width = if self.collapse_padding {
        self.columns.len() - 1
    } else {
        (self.columns.len() - 1) * (1 + self.padding.0 * 2)
    };

    let overhead = border_width + padding_width + separator_width;
    let available = max_width.saturating_sub(overhead);

    // Step 3: Get initial widths from measurements
    let mut widths: Vec<usize> = measurements.iter()
        .map(|m| m.maximum)
        .collect();

    // Step 4: Apply fixed widths
    for (i, col) in self.columns.iter().enumerate() {
        if let Some(fixed) = col.width {
            widths[i] = fixed;
        }
    }

    // Step 5: If total exceeds available, collapse
    let total: usize = widths.iter().sum();
    if total > available {
        widths = self.collapse_widths(
            &widths,
            &measurements.iter().map(|m| m.minimum).collect::<Vec<_>>(),
            available,
        );
    }

    // Step 6: If expand=true and total < available, expand ratio columns
    if self.expand {
        let total: usize = widths.iter().sum();
        if total < available {
            widths = self.expand_widths(&widths, available);
        }
    }

    widths
}
```

### 9.3 Column Collapse Algorithm

When total width exceeds available space, shrink columns proportionally:

```rust
fn collapse_widths(
    &self,
    widths: &[usize],
    minimums: &[usize],
    available: usize,
) -> Vec<usize> {
    let mut result = widths.to_vec();
    let total: usize = result.iter().sum();
    let mut excess = total.saturating_sub(available);

    // Calculate how much each column can shrink
    let shrinkable: Vec<usize> = result.iter()
        .zip(minimums.iter())
        .map(|(w, m)| w.saturating_sub(*m))
        .collect();

    let total_shrinkable: usize = shrinkable.iter().sum();
    if total_shrinkable == 0 {
        return result;
    }

    // Shrink proportionally
    for (i, shrink) in shrinkable.iter().enumerate() {
        if *shrink > 0 {
            let reduction = (*shrink * excess) / total_shrinkable;
            result[i] = result[i].saturating_sub(reduction);
        }
    }

    // Handle rounding errors
    let new_total: usize = result.iter().sum();
    if new_total > available {
        let diff = new_total - available;
        // Remove from largest shrinkable column
        for i in (0..result.len()).rev() {
            if result[i] > minimums[i] {
                let can_remove = (result[i] - minimums[i]).min(diff);
                result[i] -= can_remove;
                if result.iter().sum::<usize>() <= available {
                    break;
                }
            }
        }
    }

    result
}
```

### 9.4 Column Measurement

```rust
fn measure_column(&self, console: &Console, column: &Column, max_width: usize) -> Measurement {
    let mut cells_to_measure: Vec<&dyn Renderable> = Vec::new();

    // Include header if shown
    if self.show_header && !column.header.is_empty() {
        cells_to_measure.push(&column.header);
    }

    // Include all data cells
    for cell in &column._cells {
        cells_to_measure.push(&*cell.renderable);
    }

    // Include footer if shown
    if self.show_footer && !column.footer.is_empty() {
        cells_to_measure.push(&column.footer);
    }

    // Measure all cells
    let options = ConsoleOptions {
        max_width,
        ..console.options()
    };

    let measurement = measure_renderables(console, &options, &cells_to_measure);

    // Apply column constraints
    measurement
        .clamp(column.min_width, column.max_width)
        .with_maximum(max_width)
}
```

### 9.5 Table Rendering

```rust
impl Renderable for Table {
    fn rich_console(&self, console: &Console, options: &ConsoleOptions) -> Vec<RenderItem> {
        let max_width = options.max_width;

        // Calculate column widths
        let widths = self.calculate_column_widths(console, max_width);

        let mut segments = Vec::new();

        // Render title
        if let Some(title) = &self.title {
            segments.extend(self.render_title(console, title, &widths));
        }

        // Top border
        if self.show_edge {
            let top_line = self.box_style.get_row(&widths, RowLevel::Top, true);
            segments.push(Segment::new(&top_line, Some(self.border_style.clone())));
            segments.push(Segment::line());
        }

        // Header
        if self.show_header && !self.columns.is_empty() {
            let header_cells: Vec<_> = self.columns.iter()
                .map(|c| &c.header)
                .collect();
            segments.extend(self.render_row(console, &header_cells, &widths, &self.header_style));

            // Header separator
            let head_sep = self.box_style.get_row(&widths, RowLevel::Head, self.show_edge);
            segments.push(Segment::new(&head_sep, Some(self.border_style.clone())));
            segments.push(Segment::line());
        }

        // Data rows
        for (row_idx, row) in self.rows.iter().enumerate() {
            let row_cells: Vec<_> = self.cells[row_idx].iter()
                .map(|c| &*c.renderable)
                .collect();

            // Get row style (may alternate)
            let row_style = if !self.row_styles.is_empty() {
                &self.row_styles[row_idx % self.row_styles.len()]
            } else {
                &row.style
            };

            segments.extend(self.render_row(console, &row_cells, &widths, row_style));

            // Row separator (if show_lines or end_section)
            if self.show_lines || row.end_section {
                let sep = self.box_style.get_row(&widths, RowLevel::Row, self.show_edge);
                segments.push(Segment::new(&sep, Some(self.border_style.clone())));
                segments.push(Segment::line());
            }
        }

        // Footer
        if self.show_footer && !self.columns.is_empty() {
            // Footer separator
            let foot_sep = self.box_style.get_row(&widths, RowLevel::Foot, self.show_edge);
            segments.push(Segment::new(&foot_sep, Some(self.border_style.clone())));
            segments.push(Segment::line());

            let footer_cells: Vec<_> = self.columns.iter()
                .map(|c| &c.footer)
                .collect();
            segments.extend(self.render_row(console, &footer_cells, &widths, &self.footer_style));
        }

        // Bottom border
        if self.show_edge {
            let bottom_line = self.box_style.get_row(&widths, RowLevel::Bottom, true);
            segments.push(Segment::new(&bottom_line, Some(self.border_style.clone())));
            segments.push(Segment::line());
        }

        // Caption
        if let Some(caption) = &self.caption {
            segments.extend(self.render_caption(console, caption, &widths));
        }

        segments.into_iter().map(RenderItem::Segment).collect()
    }
}
```

### 9.6 Row Rendering (Vertical Alignment)

```rust
fn render_row(
    &self,
    console: &Console,
    cells: &[&dyn Renderable],
    widths: &[usize],
    row_style: &Style,
) -> Vec<Segment> {
    // Render each cell to lines
    let mut cell_lines: Vec<Vec<Vec<Segment>>> = Vec::new();
    let mut max_height = 0;

    for (i, (cell, &width)) in cells.iter().zip(widths.iter()).enumerate() {
        let col = &self.columns[i];
        let cell_options = ConsoleOptions {
            max_width: width,
            justify: Some(col.justify),
            overflow: Some(col.overflow),
            no_wrap: Some(col.no_wrap),
            ..console.options()
        };

        let lines = console.render_lines(*cell, &cell_options, Some(&col.style), true, false);
        max_height = max_height.max(lines.len());
        cell_lines.push(lines);
    }

    // Apply vertical alignment to each cell
    for (i, lines) in cell_lines.iter_mut().enumerate() {
        let col = &self.columns[i];
        let width = widths[i];

        *lines = match col.vertical {
            VerticalAlignMethod::Top => {
                Segment::align_top(std::mem::take(lines), width, max_height, col.style.clone())
            }
            VerticalAlignMethod::Middle => {
                Segment::align_middle(std::mem::take(lines), width, max_height, col.style.clone())
            }
            VerticalAlignMethod::Bottom => {
                Segment::align_bottom(std::mem::take(lines), width, max_height, col.style.clone())
            }
        };
    }

    // Combine cells into row output
    let mut result = Vec::new();
    let (h_pad, v_pad) = self.padding;
    let pad_str = " ".repeat(h_pad);

    for line_idx in 0..max_height {
        // Left edge
        if self.show_edge {
            result.push(Segment::new(&self.box_style.head[0].to_string(), Some(self.border_style.clone())));
        }
        if self.pad_edge {
            result.push(Segment::new(&pad_str, Some(row_style.clone())));
        }

        // Cells
        for (col_idx, cell) in cell_lines.iter().enumerate() {
            result.extend(cell[line_idx].clone());

            // Cell separator
            if col_idx < cell_lines.len() - 1 {
                if self.pad_edge || !self.collapse_padding {
                    result.push(Segment::new(&pad_str, Some(row_style.clone())));
                }
                result.push(Segment::new(&self.box_style.head[2].to_string(), Some(self.border_style.clone())));
                if self.pad_edge || !self.collapse_padding {
                    result.push(Segment::new(&pad_str, Some(row_style.clone())));
                }
            }
        }

        // Right edge
        if self.pad_edge {
            result.push(Segment::new(&pad_str, Some(row_style.clone())));
        }
        if self.show_edge {
            result.push(Segment::new(&self.box_style.head[3].to_string(), Some(self.border_style.clone())));
        }

        result.push(Segment::line());
    }

    result
}
```

---
