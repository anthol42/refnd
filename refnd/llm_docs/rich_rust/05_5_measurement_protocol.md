## 5. Measurement Protocol

> Source: `rich/measure.py` (151 lines)

### 5.1 Measurement Structure

```rust
struct Measurement {
    minimum: usize,  // Minimum cells required
    maximum: usize,  // Maximum cells required
}

impl Measurement {
    fn span(&self) -> usize { self.maximum - self.minimum }

    fn normalize(&self) -> Self {
        let min = self.minimum.min(self.maximum).max(0);
        let max = self.maximum.max(self.minimum).max(0);
        Measurement { minimum: min, maximum: max }
    }

    fn with_maximum(&self, width: usize) -> Self {
        Measurement {
            minimum: self.minimum.min(width),
            maximum: self.maximum.min(width),
        }
    }

    fn with_minimum(&self, width: usize) -> Self {
        let width = width.max(0);
        Measurement {
            minimum: self.minimum.max(width),
            maximum: self.maximum.max(width),
        }
    }

    fn clamp(&self, min_width: Option<usize>, max_width: Option<usize>) -> Self {
        let mut m = *self;
        if let Some(min) = min_width { m = m.with_minimum(min); }
        if let Some(max) = max_width { m = m.with_maximum(max); }
        m
    }
}
```

### 5.2 Measurement.get()

```rust
fn get(console: &Console, options: &ConsoleOptions, renderable: &dyn Renderable) -> Measurement {
    let max_width = options.max_width;
    if max_width < 1 { return Measurement { minimum: 0, maximum: 0 }; }

    if let Some(measure_fn) = renderable.rich_measure() {
        measure_fn(console, options)
            .normalize()
            .with_maximum(max_width)
            .normalize()
    } else {
        Measurement { minimum: 0, maximum: max_width }
    }
}
```

### 5.3 measure_renderables()

```rust
fn measure_renderables(console: &Console, options: &ConsoleOptions, renderables: &[&dyn Renderable]) -> Measurement {
    if renderables.is_empty() {
        return Measurement { minimum: 0, maximum: 0 };
    }

    let measurements: Vec<_> = renderables.iter()
        .map(|r| Measurement::get(console, options, *r))
        .collect();

    Measurement {
        minimum: measurements.iter().map(|m| m.minimum).max().unwrap(),
        maximum: measurements.iter().map(|m| m.maximum).max().unwrap(),
    }
}
```

**Aggregation Rules:**
- Combined minimum = max of all minimums (tightest constraint)
- Combined maximum = max of all maximums (most flexible)

---
