## Appendix B: Recommended Crate Mappings

| Python | Rust Crate | Purpose |
|--------|------------|---------|
| `colorsys` | `palette` | Color conversion (RGB/HLS) |
| `wcwidth` | `unicode-width` | Character cell width |
| `re` | `regex` | Regular expressions |
| `sys.stdout` | `crossterm` | Terminal detection/manipulation |
| `functools.lru_cache` | `lru` or `cached` | Memoization |
| `dataclasses` | Native structs | Data modeling |
| `typing` | Native types | Type annotations |
| `enum.IntEnum` | `num_enum` | Integer enums |
| `fractions.Fraction` | `num-rational` | Exact ratio arithmetic |

---

*Specification extracted from Python Rich v13.x source code, 2026-01-17*
