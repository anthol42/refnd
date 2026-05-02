## 17. Layout System

> Source: `rich/layout.py` (443 lines), `rich/region.py` (11 lines)

The `Layout` class divides a fixed-height terminal area into rows and columns,
enabling dashboard-style interfaces with multiple panes. It uses ratio-based
distribution for flexible sizing.

**Implementation note (Rust):** `src/renderables/layout.rs` provides ratio-based
row/column splitting, named lookup, and placeholder rendering. It does not maintain
an internal render-map cache or debug tree view.

### 17.1 Data Structures

#### Region

```rust
/// Rectangular region of the screen
struct Region {
    x: usize,      // Horizontal position (0 = left edge)
    y: usize,      // Vertical position (0 = top edge)
    width: usize,  // Width in cells
    height: usize, // Height in lines
}
```

#### LayoutRender

```rust
/// Result of rendering a single layout region
struct LayoutRender {
    region: Region,
    render: Vec<Vec<Segment>>,  // Lines of segments
}

type RegionMap = HashMap<Layout, Region>;
type RenderMap = HashMap<Layout, LayoutRender>;
```

#### Layout Configuration

```rust
struct Layout {
    // Content
    renderable: Box<dyn Renderable>,  // Content or placeholder
    name: Option<String>,             // Identifier for lookup

    // Size constraints
    size: Option<usize>,              // Fixed size (None = flexible)
    minimum_size: usize,              // Default: 1
    ratio: usize,                     // Flex ratio, default: 1

    // State
    visible: bool,                    // Default: true
    splitter: Box<dyn Splitter>,      // Row or Column, default: Column
    children: Vec<Layout>,            // Sub-layouts

    // Internal
    render_map: RenderMap,            // Last render result
    lock: RwLock<()>,                 // Thread safety
}
```

### 17.2 Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `renderable` | `Option<RenderableType>` | Placeholder | Content to display |
| `name` | `Option<String>` | `None` | Identifier for `layout["name"]` lookup |
| `size` | `Option<usize>` | `None` | Fixed size in cells/lines |
| `minimum_size` | `usize` | `1` | Minimum allowed size |
| `ratio` | `usize` | `1` | Flex ratio for size distribution |
| `visible` | `bool` | `true` | Whether to render this layout |

### 17.3 Splitter Abstraction

```rust
trait Splitter {
    fn name(&self) -> &str;
    fn get_tree_icon(&self) -> &str;
    fn divide(&self, children: &[Layout], region: Region) -> Vec<(Layout, Region)>;
}
```

#### RowSplitter (Horizontal)

Divides region horizontally (children side by side):

```rust
struct RowSplitter;

impl Splitter for RowSplitter {
    fn name(&self) -> &str { "row" }
    fn get_tree_icon(&self) -> &str { "[layout.tree.row]⬌" }

    fn divide(&self, children: &[Layout], region: Region) -> Vec<(Layout, Region)> {
        let Region { x, y, width, height } = region;
        let render_widths = ratio_resolve(width, children);  // Uses ratio algorithm

        let mut result = Vec::new();
        let mut offset = 0;

        for (child, child_width) in children.iter().zip(render_widths) {
            result.push((child.clone(), Region {
                x: x + offset,
                y,
                width: child_width,
                height,
            }));
            offset += child_width;
        }
        result
    }
}
```

#### ColumnSplitter (Vertical)

Divides region vertically (children stacked):

```rust
struct ColumnSplitter;

impl Splitter for ColumnSplitter {
    fn name(&self) -> &str { "column" }
    fn get_tree_icon(&self) -> &str { "[layout.tree.column]⬍" }

    fn divide(&self, children: &[Layout], region: Region) -> Vec<(Layout, Region)> {
        let Region { x, y, width, height } = region;
        let render_heights = ratio_resolve(height, children);

        let mut result = Vec::new();
        let mut offset = 0;

        for (child, child_height) in children.iter().zip(render_heights) {
            result.push((child.clone(), Region {
                x,
                y: y + offset,
                width,
                height: child_height,
            }));
            offset += child_height;
        }
        result
    }
}
```

### 17.4 Edge Protocol for Ratio Resolution

Layout implements the Edge protocol for ratio_resolve():

```rust
impl Edge for Layout {
    fn size(&self) -> Option<usize> {
        self.size  // Fixed size if set
    }

    fn ratio(&self) -> usize {
        self.ratio  // Flex ratio
    }

    fn minimum_size(&self) -> usize {
        self.minimum_size
    }
}
```

### 17.5 Split Operations

```rust
impl Layout {
    /// Split into multiple sub-layouts
    fn split(&mut self, layouts: Vec<impl Into<Layout>>, splitter: impl Into<Splitter>) {
        let layouts: Vec<Layout> = layouts.into_iter()
            .map(|l| l.into())  // Convert RenderableType to Layout if needed
            .collect();

        self.splitter = splitter.into();
        self.children = layouts;
    }

    /// Convenience: split horizontally (row)
    fn split_row(&mut self, layouts: Vec<impl Into<Layout>>) {
        self.split(layouts, RowSplitter);
    }

    /// Convenience: split vertically (column)
    fn split_column(&mut self, layouts: Vec<impl Into<Layout>>) {
        self.split(layouts, ColumnSplitter);
    }

    /// Add to existing split
    fn add_split(&mut self, layouts: Vec<impl Into<Layout>>) {
        self.children.extend(layouts.into_iter().map(|l| l.into()));
    }

    /// Remove all children
    fn unsplit(&mut self) {
        self.children.clear();
    }
}
```

### 17.6 Named Layout Lookup

```rust
impl Layout {
    /// Get layout by name (recursive search)
    fn get(&self, name: &str) -> Option<&Layout> {
        if self.name.as_deref() == Some(name) {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.get(name) {
                return Some(found);
            }
        }
        None
    }

    /// Get layout by name, panic if not found
    fn index(&self, name: &str) -> &Layout {
        self.get(name).unwrap_or_else(|| panic!("No layout with name {name:?}"))
    }

    /// Mutable access by name
    fn get_mut(&mut self, name: &str) -> Option<&mut Layout> {
        if self.name.as_deref() == Some(name) {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.get_mut(name) {
                return Some(found);
            }
        }
        None
    }
}

// Usage: layout["header"].update(content)
impl Index<&str> for Layout {
    type Output = Layout;
    fn index(&self, name: &str) -> &Self::Output {
        self.get(name).expect("Layout not found")
    }
}
```

### 17.7 Visibility Filtering

The `children` property returns only visible children:

```rust
impl Layout {
    fn children(&self) -> Vec<&Layout> {
        self.children.iter()
            .filter(|c| c.visible)
            .collect()
    }
}
```

Hidden layouts are skipped during splitting but still exist in the tree.

### 17.8 Placeholder Rendering

When no renderable is set, Layout shows a placeholder panel:

```rust
struct Placeholder {
    layout: Layout,
    style: Style,
}

impl Renderable for Placeholder {
    fn rich_console(&self, console: &Console, options: &ConsoleOptions) -> Vec<RenderItem> {
        let width = options.max_width;
        let height = options.height.unwrap_or(options.size.height);

        let title = match &self.layout.name {
            Some(name) => format!("{name:?} ({width} x {height})"),
            None => format!("({width} x {height})"),
        };

        Panel::new(
            Align::center(Pretty::new(&self.layout)).vertical_middle()
        )
        .style(self.style.clone())
        .title(ReprHighlighter::highlight(&title))
        .border_style(Style::parse("blue").unwrap())
        .height(height)
        .rich_console(console, options)
    }
}
```

### 17.9 Region Map Generation

The `_make_region_map` method recursively assigns regions to all layouts:

```rust
impl Layout {
    fn make_region_map(&self, width: usize, height: usize) -> RegionMap {
        let mut stack = vec![(self, Region { x: 0, y: 0, width, height })];
        let mut layout_regions = Vec::new();

        // Depth-first traversal
        while let Some((layout, region)) = stack.pop() {
            layout_regions.push((layout, region));

            let children = layout.children();
            if !children.is_empty() {
                // Divide region among children
                for (child, child_region) in layout.splitter.divide(&children, region) {
                    stack.push((child, child_region));
                }
            }
        }

        // Sort by region (top-to-bottom, left-to-right)
        layout_regions.sort_by_key(|(_, r)| (r.y, r.x));
        layout_regions.into_iter().collect()
    }
}
```

### 17.10 Rendering Algorithm

```rust
impl Layout {
    fn render(&self, console: &Console, options: &ConsoleOptions) -> RenderMap {
        let render_width = options.max_width;
        let render_height = options.height.unwrap_or(console.height());

        // Build region map
        let region_map = self.make_region_map(render_width, render_height);

        // Render only leaf layouts (no children)
        let leaf_layouts: Vec<_> = region_map.iter()
            .filter(|(layout, _)| layout.children().is_empty())
            .collect();

        let mut render_map = RenderMap::new();

        for (layout, region) in leaf_layouts {
            let lines = console.render_lines(
                layout.renderable(),
                &options.update_dimensions(region.width, region.height),
            );
            render_map.insert(layout.clone(), LayoutRender {
                region: *region,
                render: lines,
            });
        }

        render_map
    }
}

impl Renderable for Layout {
    fn rich_console(&self, console: &Console, options: &ConsoleOptions) -> Vec<RenderItem> {
        let _guard = self.lock.read();

        let width = options.max_width.unwrap_or(console.width());
        let height = options.height.unwrap_or(console.height());

        let render_map = self.render(console, &options.update_dimensions(width, height));
        self.render_map = render_map.clone();

        // Build output buffer (height lines)
        let mut layout_lines: Vec<Vec<Segment>> = (0..height).map(|_| Vec::new()).collect();

        // Place each rendered region into the buffer
        for LayoutRender { region, render } in render_map.values() {
            for (row_idx, line) in render.iter().enumerate() {
                let y = region.y + row_idx;
                if y < height {
                    layout_lines[y].extend(line.clone());
                }
            }
        }

        // Yield lines with newlines
        let mut segments = Vec::new();
        for line in layout_lines {
            segments.extend(line);
            segments.push(Segment::line());
        }
        segments
    }
}
```

### 17.11 Partial Screen Refresh

For efficiency, individual layouts can be refreshed without re-rendering everything:

```rust
impl Layout {
    fn refresh_screen(&mut self, console: &Console, layout_name: &str) {
        let _guard = self.lock.write();

        let layout = self.get_mut(layout_name).expect("Layout not found");
        let LayoutRender { region, .. } = self.render_map.get(layout).expect("Layout not rendered");

        let Region { x, y, width, height } = *region;

        // Re-render just this layout
        let lines = console.render_lines(
            layout.renderable(),
            &console.options.update_dimensions(width, height),
        );

        // Update render map
        self.render_map.insert(layout.clone(), LayoutRender {
            region: *region,
            render: lines.clone(),
        });

        // Write directly to screen at position
        console.update_screen_lines(&lines, x, y);
    }
}
```

### 17.12 Update Content

```rust
impl Layout {
    fn update(&mut self, renderable: impl Into<RenderableType>) {
        let _guard = self.lock.write();
        self.renderable = Box::new(renderable.into());
    }

    /// Get the renderable (self if has children, otherwise content)
    fn renderable(&self) -> &dyn Renderable {
        if self.children.is_empty() {
            &*self.renderable
        } else {
            self
        }
    }
}
```

### 17.13 Tree Visualization

Layout provides a tree view for debugging structure:

```rust
impl Layout {
    fn tree(&self) -> Tree {
        fn summary(layout: &Layout) -> Table {
            let icon = layout.splitter.get_tree_icon();
            let text = if layout.visible {
                Pretty::new(layout)
            } else {
                Styled::new(Pretty::new(layout), "dim")
            };

            Table::grid().padding((0, 1, 0, 0))
                .add_row(vec![icon, text])
        }

        fn recurse(tree: &mut Tree, layout: &Layout) {
            for child in &layout.children {
                let child_tree = tree.add(
                    summary(child),
                    format!("layout.tree.{}", child.splitter.name()),
                );
                recurse(child_tree, child);
            }
        }

        let mut tree = Tree::new(summary(self))
            .guide_style(format!("layout.tree.{}", self.splitter.name()))
            .highlight(true);

        recurse(&mut tree, self);
        tree
    }
}
```

### 17.14 Default Styles

| Style Name | Purpose |
|------------|---------|
| `layout.tree.row` | Guide style for row splits in tree view |
| `layout.tree.column` | Guide style for column splits in tree view |

### 17.15 Error Types

```rust
/// Layout-related errors
enum LayoutError {
    /// Requested splitter does not exist
    NoSplitter(String),
    /// Named layout not found
    NotFound(String),
}
```

### 17.16 Usage Example

```rust
// Create root layout
let mut layout = Layout::new();

// Split into header, main, footer
layout.split_column(vec![
    Layout::new().name("header").size(3),
    Layout::new().name("main").ratio(1),
    Layout::new().name("footer").size(10),
]);

// Split main into sidebar and body
layout["main"].split_row(vec![
    Layout::new().name("side"),
    Layout::new().name("body").ratio(2),
]);

// Update content
layout["header"].update(Clock::new());
layout["body"].update(some_content);

// Render with Live
with Live(layout, screen=true) {
    // Updates happen automatically
}
```

### 17.17 Edge Cases

1. **Zero-size region:** minimum_size ensures at least 1 cell/line
2. **More children than space:** ratio_resolve handles gracefully
3. **Hidden children:** Excluded from division, remaining children get more space
4. **Deeply nested:** Stack-based traversal avoids recursion limits
5. **Thread safety:** Lock protects all mutations and render_map updates
6. **Empty layout:** Shows placeholder with dimensions

---
