## 16. Live Display System

> Source: `rich/live.py` (401 lines), `rich/live_render.py` (107 lines)

The `Live` class provides auto-updating display of renderables with cursor manipulation
and screen refresh. It's the foundation for progress bars, status spinners, and any
dynamic terminal UI.

**Implementation note (Rust):** `src/live.rs` implements Live with nested Live stacking,
alternate screen support, overflow handling, and an auto-refresh thread. Stdout/stderr
redirection is supported via process-wide stdio overrides in interactive terminals
(`LiveOptions.redirect_stdout` / `redirect_stderr`), and proxy writers are also available
(`Live::stdout_proxy()` / `Live::stderr_proxy()`). Jupyter-specific behavior is not supported.

### 16.1 Data Structures

#### VerticalOverflowMethod

```rust
enum VerticalOverflowMethod {
    Crop,      // Truncate lines that exceed terminal height
    Ellipsis,  // Show "..." indicator for overflow
    Visible,   // Allow content to overflow (used on final render)
}
```

#### Live Configuration

```rust
struct Live {
    // Core state
    renderable: Option<Box<dyn Renderable>>,
    console: Console,
    started: bool,
    nested: bool,                              // True if nested inside another Live

    // Display options
    screen: bool,                              // Use alternate screen buffer
    alt_screen: bool,                          // Alternate screen is currently active
    transient: bool,                           // Clear output on exit (auto-true if screen=true)
    vertical_overflow: VerticalOverflowMethod, // Default: Ellipsis

    // Refresh control
    auto_refresh: bool,                        // Default: true
    refresh_per_second: f64,                   // Default: 4.0
    refresh_thread: Option<RefreshThread>,

    // I/O redirection
    redirect_stdout: bool,                     // Default: true
    redirect_stderr: bool,                     // Default: true
    restore_stdout: Option<Box<dyn Write>>,
    restore_stderr: Option<Box<dyn Write>>,

    // Internal
    lock: RwLock<()>,                          // Thread-safe refresh
    live_render: LiveRender,
    get_renderable: Option<Box<dyn Fn() -> Box<dyn Renderable>>>,
}
```

### 16.2 Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `renderable` | `Option<RenderableType>` | `None` | Initial content to display |
| `console` | `Option<Console>` | Global console | Target console for output |
| `screen` | `bool` | `false` | Use alternate screen mode |
| `auto_refresh` | `bool` | `true` | Enable automatic refresh thread |
| `refresh_per_second` | `f64` | `4.0` | Refresh rate (must be > 0) |
| `transient` | `bool` | `false` | Clear display on exit |
| `redirect_stdout` | `bool` | `true` | Redirect stdout through console |
| `redirect_stderr` | `bool` | `true` | Redirect stderr through console |
| `vertical_overflow` | `VerticalOverflowMethod` | `Ellipsis` | Overflow handling |
| `get_renderable` | `Option<Fn() -> Renderable>` | `None` | Dynamic content callback |

**Validation:**
- `refresh_per_second` must be > 0 (assertion in Python)
- If `screen=true`, `transient` is forced to `true`

### 16.3 Refresh Thread

When `auto_refresh=true`, a daemon thread periodically calls `refresh()`:

```rust
struct RefreshThread {
    live: Arc<Live>,
    refresh_per_second: f64,
    done: AtomicBool,
}

impl RefreshThread {
    fn run(&self) {
        let interval = Duration::from_secs_f64(1.0 / self.refresh_per_second);
        while !self.done.load(Ordering::Relaxed) {
            thread::sleep(interval);
            if !self.done.load(Ordering::Relaxed) {
                self.live.refresh();
            }
        }
    }

    fn stop(&self) {
        self.done.store(true, Ordering::Relaxed);
    }
}
```

### 16.4 Lifecycle: start() and stop()

#### start() Sequence

```rust
fn start(&mut self, refresh: bool) {
    if self.started { return; }
    self.started = true;

    // 1. Register with console (returns false if already has active Live)
    if !self.console.set_live(self) {
        self.nested = true;
        return;  // Nested Live delegates to parent
    }

    // 2. Enable alternate screen if requested
    if self.screen {
        self.alt_screen = self.console.set_alt_screen(true);
    }

    // 3. Hide cursor
    self.console.show_cursor(false);

    // 4. Enable I/O redirection
    self.enable_redirect_io();

    // 5. Push render hook for output interception
    self.console.push_render_hook(self);

    // 6. Initial refresh (optional, if renderable provided)
    if refresh {
        if let Err(e) = self.refresh() {
            self.stop();  // Clean up on error
            return Err(e);
        }
    }

    // 7. Start refresh thread
    if self.auto_refresh {
        self.refresh_thread = Some(RefreshThread::new(self, self.refresh_per_second));
        self.refresh_thread.as_ref().unwrap().start();
    }
}
```

#### stop() Sequence

```rust
fn stop(&mut self) {
    if !self.started { return; }
    self.started = false;

    // 1. Clear console's live reference
    self.console.clear_live();

    // 2. Handle nested case
    if self.nested {
        if !self.transient {
            self.console.print(&self.renderable);
        }
        return;
    }

    // 3. Stop refresh thread
    if self.auto_refresh {
        if let Some(thread) = self.refresh_thread.take() {
            thread.stop();
        }
    }

    // 4. Final render with full overflow visibility
    self.vertical_overflow = VerticalOverflowMethod::Visible;

    // 5. Clean up
    if !self.alt_screen && !self.console.is_jupyter() {
        self.refresh();
    }

    self.disable_redirect_io();
    self.console.pop_render_hook();

    if !self.alt_screen && self.console.is_terminal() {
        self.console.line();  // Add final newline
    }

    self.console.show_cursor(true);

    if self.alt_screen {
        self.console.set_alt_screen(false);
    }

    // 6. Clear transient output
    if self.transient && !self.alt_screen {
        self.console.control(self.live_render.restore_cursor());
    }
}
```

### 16.5 Context Manager Usage

```rust
impl Live {
    fn enter(&mut self) -> &mut Self {
        self.start(self.renderable.is_some());
        self
    }

    fn exit(&mut self) {
        self.stop();
    }
}

// Usage:
// with Live(table) as live:
//     live.update(new_table)
```

### 16.6 LiveRender: Cursor Positioning

`LiveRender` tracks the rendered shape for cursor restoration:

```rust
struct LiveRender {
    renderable: Box<dyn Renderable>,
    style: Style,
    vertical_overflow: VerticalOverflowMethod,
    shape: Option<(usize, usize)>,  // (width, height) of last render
}

impl LiveRender {
    /// Generate control codes to position cursor at render start
    fn position_cursor(&self) -> Control {
        if let Some((_, height)) = self.shape {
            Control::new(vec![
                ControlCode::CarriageReturn,
                ControlCode::EraseInLine(2),
                // Move up and erase for each line
                ...(0..height-1).flat_map(|_| vec![
                    ControlCode::CursorUp(1),
                    ControlCode::EraseInLine(2),
                ])
            ])
        } else {
            Control::new(vec![])
        }
    }

    /// Generate control codes to clear render and restore cursor
    fn restore_cursor(&self) -> Control {
        if let Some((_, height)) = self.shape {
            Control::new(vec![
                ControlCode::CarriageReturn,
                ...(0..height).flat_map(|_| vec![
                    ControlCode::CursorUp(1),
                    ControlCode::EraseInLine(2),
                ])
            ])
        } else {
            Control::new(vec![])
        }
    }
}
```

### 16.7 Rendering with Overflow Handling

```rust
impl Renderable for LiveRender {
    fn rich_console(&self, console: &Console, options: &ConsoleOptions) -> Vec<RenderItem> {
        let lines = console.render_lines(&*self.renderable, options, Some(&self.style), false);
        let shape = Segment::get_shape(&lines);
        let (_, height) = shape;

        let mut result_lines = lines;

        // Handle overflow
        if height > options.size.height {
            match self.vertical_overflow {
                VerticalOverflowMethod::Crop => {
                    result_lines = result_lines[..options.size.height].to_vec();
                }
                VerticalOverflowMethod::Ellipsis => {
                    result_lines = result_lines[..options.size.height - 1].to_vec();
                    let ellipsis = Text::new("...")
                        .overflow(OverflowMethod::Crop)
                        .justify(JustifyMethod::Center)
                        .style_name("live.ellipsis");
                    result_lines.push(console.render(&ellipsis));
                }
                VerticalOverflowMethod::Visible => {
                    // Allow overflow (used for final render)
                }
            }
        }

        // Update shape for cursor positioning
        self.shape = Some(Segment::get_shape(&result_lines));

        // Yield lines with newlines
        let mut segments = Vec::new();
        for (idx, line) in result_lines.iter().enumerate() {
            segments.extend(line.clone());
            if idx < result_lines.len() - 1 {
                segments.push(Segment::line());
            }
        }
        segments
    }
}
```

### 16.8 Console Integration: RenderHook

Live implements `RenderHook` to intercept all console output:

```rust
trait RenderHook {
    fn process_renderables(&self, renderables: Vec<ConsoleRenderable>) -> Vec<ConsoleRenderable>;
}

impl RenderHook for Live {
    fn process_renderables(&self, renderables: Vec<ConsoleRenderable>) -> Vec<ConsoleRenderable> {
        self.live_render.vertical_overflow = self.vertical_overflow;

        if self.console.is_interactive() {
            // Active terminal: prepend cursor reset, append live render
            let reset = if self.alt_screen {
                Control::home()
            } else {
                self.live_render.position_cursor()
            };
            vec![reset, ...renderables, self.live_render.clone()]
        } else if !self.started && !self.transient {
            // Non-TTY final output
            vec![...renderables, self.live_render.clone()]
        } else {
            renderables
        }
    }
}
```

### 16.9 Nested Live Handling

Multiple Live instances can be active simultaneously via the Console's `_live_stack`:

```rust
// In Console:
struct Console {
    live_stack: Vec<Arc<Live>>,
    // ...
}

impl Console {
    fn set_live(&mut self, live: &Live) -> bool {
        if self.live_stack.is_empty() {
            self.live_stack.push(Arc::new(live.clone()));
            true  // First Live, proceed normally
        } else {
            self.live_stack.push(Arc::new(live.clone()));
            false // Nested Live
        }
    }

    fn clear_live(&mut self) {
        self.live_stack.pop();
    }
}

// In Live.renderable property:
fn renderable(&self) -> Box<dyn Renderable> {
    let live_stack = &self.console.live_stack;
    if !live_stack.is_empty() && Arc::ptr_eq(&live_stack[0], &Arc::new(self)) {
        // First Live renders entire stack as Group
        Group::new(live_stack.iter().map(|l| l.get_renderable()).collect())
    } else {
        self.get_renderable()
    }
}
```

### 16.10 I/O Redirection

When active, Live intercepts stdout/stderr to prevent output from disrupting the display:

```rust
struct FileProxy {
    console: Console,
    original: Box<dyn Write>,
}

impl Write for FileProxy {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Route through console which handles cursor positioning
        let text = String::from_utf8_lossy(buf);
        self.console.print(&text);
        Ok(buf.len())
    }
}

impl Live {
    fn enable_redirect_io(&mut self) {
        if self.console.is_terminal() || self.console.is_jupyter() {
            if self.redirect_stdout {
                self.restore_stdout = Some(io::stdout());
                // Redirect stdout to FileProxy
            }
            if self.redirect_stderr {
                self.restore_stderr = Some(io::stderr());
                // Redirect stderr to FileProxy
            }
        }
    }

    fn disable_redirect_io(&mut self) {
        if let Some(stdout) = self.restore_stdout.take() {
            // Restore original stdout
        }
        if let Some(stderr) = self.restore_stderr.take() {
            // Restore original stderr
        }
    }
}
```

### 16.11 update() and refresh()

```rust
impl Live {
    /// Update the renderable content
    fn update(&mut self, renderable: impl Into<RenderableType>, refresh: bool) {
        let renderable = renderable.into();

        // Convert string to Text if needed
        let renderable = if let RenderableType::String(s) = renderable {
            self.console.render_str(&s)
        } else {
            renderable
        };

        let _guard = self.lock.write();
        self.renderable = Some(Box::new(renderable));

        if refresh {
            self.refresh();
        }
    }

    /// Refresh the display
    fn refresh(&self) {
        let _guard = self.lock.read();
        self.live_render.set_renderable(self.renderable());

        if self.nested {
            // Delegate to parent Live
            if let Some(parent) = self.console.live_stack.first() {
                parent.refresh();
            }
            return;
        }

        if self.console.is_terminal() && !self.console.is_dumb_terminal() {
            self.console.print(Control::new(vec![]));  // Triggers render hook
        } else if !self.started && !self.transient {
            // Non-TTY or dumb terminal: allow final output
            self.console.print(Control::new(vec![]));
        }
    }
}
```

### 16.12 Non-TTY and Dumb Terminal Behavior

| Scenario | Behavior |
|----------|----------|
| Interactive TTY | Full live updating with cursor positioning |
| Non-interactive (piped) | No live updates; final render only if `transient=false` |
| Dumb terminal | No live updates; final render only if `transient=false` |
| Jupyter | IPython widget display with `clear_output(wait=True)` |

### 16.13 Alternate Screen Mode

When `screen=true`, Live uses the alternate screen buffer:

- On start: `set_alt_screen(true)` switches to alternate buffer
- Cursor positioning uses `Control::home()` instead of `position_cursor()`
- On stop: `set_alt_screen(false)` restores primary buffer
- `transient` is forced to `true` (alternate screen is always cleared)

### 16.14 Default Styles

| Style Name | Purpose |
|------------|---------|
| `live.ellipsis` | Style for the "..." overflow indicator |

### 16.15 Thread Safety

- `_lock` (RLock in Python, RwLock in Rust) protects all state modifications
- Refresh thread acquires lock before calling refresh()
- User code calling update()/refresh() also acquires lock
- Nested Live instances delegate refreshes atomically

### 16.16 Edge Cases

1. **Exception during refresh:** If initial refresh fails, `stop()` is called to clean up
2. **Zero-height terminal:** Overflow handling still applies; content may be fully cropped
3. **Rapid updates:** Lock ensures only one refresh at a time; missed updates are fine
4. **Nested Live with transient parent:** Each Live tracks its own transient flag
5. **Already started:** Calling `start()` when already started is a no-op
6. **Already stopped:** Calling `stop()` when not started is a no-op

---
