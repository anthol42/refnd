## Appendix A: Rust Trait Summary

```rust
/// Primary rendering trait (equivalent to Python Rich `__rich_console__`).
trait Renderable {
    fn render<'a>(&'a self, console: &Console, options: &ConsoleOptions) -> Vec<Segment<'a>>;
}

/// Measurement trait (equivalent to Python Rich `__rich_measure__`).
trait RichMeasure {
    fn rich_measure(&self, console: &Console, options: &ConsoleOptions) -> Measurement;
}

/// Casting trait (equivalent to Python Rich `__rich__`).
trait RichCast {
    fn rich_cast(&self) -> RichCastOutput;
}

/// Helper that mirrors Python Rich `rich.protocol.rich_cast` recursion.
fn rich_cast(value: &dyn RichCast) -> RichCastOutput;
```

---
