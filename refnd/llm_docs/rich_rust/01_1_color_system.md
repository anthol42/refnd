## 1. Color System

> Source: `rich/color.py` (621 lines), `rich/color_triplet.py`, `rich/_palettes.py`

### 1.1 Data Structures

#### ColorTriplet

NamedTuple representing RGB color components.

```rust
struct ColorTriplet {
    red: u8,        // Red component in 0-255 range
    green: u8,      // Green component in 0-255 range
    blue: u8,       // Blue component in 0-255 range
}
```

**Properties:**
- `hex()` -> String: Returns CSS-style hex format `#rrggbb` (e.g., `#FF0000`)
- `rgb()` -> String: Returns CSS-style rgb format `rgb(r,g,b)` (e.g., `rgb(255,0,0)`)
- `normalized()` -> (f64, f64, f64): Returns (red, green, blue) as floats in range 0.0-1.0

#### ColorSystem (IntEnum)

Represents the color system capability of terminals.

```rust
enum ColorSystem {
    STANDARD = 1,    // 4-bit ANSI colors (16 colors)
    EIGHT_BIT = 2,   // 8-bit colors (256 colors)
    TRUECOLOR = 3,   // 24-bit RGB colors (16 million colors)
    WINDOWS = 4,     // Windows 10+ console palette (16 colors)
}
```

#### ColorType (IntEnum)

Type of color stored in Color structure.

```rust
enum ColorType {
    DEFAULT = 0,     // Default terminal color (no RGB/number)
    STANDARD = 1,    // 4-bit ANSI standard color (0-15)
    EIGHT_BIT = 2,   // 8-bit color (0-255)
    TRUECOLOR = 3,   // 24-bit RGB color
    WINDOWS = 4,     // Windows console color (0-15)
}
```

#### Color Structure

```rust
struct Color {
    name: String,                    // Name of the color (input that was parsed)
    color_type: ColorType,           // Type of color
    number: Option<u8>,             // Color number (for STANDARD, EIGHT_BIT, WINDOWS)
    triplet: Option<ColorTriplet>,  // RGB components (for TRUECOLOR)
}
```

**Methods:**
- `system()` -> ColorSystem: Returns the native color system for this color
- `is_system_defined()` -> bool: Returns true if system is STANDARD or WINDOWS
- `is_default()` -> bool: Returns true if color_type == DEFAULT
- `get_truecolor(theme, foreground)` -> ColorTriplet: Converts color to RGB triplet
- `from_ansi(number: u8)` -> Color: Create from 8-bit ANSI number
- `from_triplet(triplet)` -> Color: Create from RGB triplet as TRUECOLOR
- `from_rgb(red, green, blue)` -> Color: Create from RGB components
- `default()` -> Color: Create default color
- `parse(color: &str)` -> Result<Color, ColorParseError>: Parse color string (cached, LRU 1024)
- `get_ansi_codes(foreground: bool)` -> Vec<String>: Get ANSI escape codes
- `downgrade(system: ColorSystem)` -> Color: Convert to lower-capability color system

### 1.2 Color Parsing

The `Color::parse()` method accepts these formats (case-insensitive):

| Format | Example | Result |
|--------|---------|--------|
| Named colors | `red`, `bright_blue` | STANDARD (0-15) or EIGHT_BIT (16-255) |
| Hex format | `#FF0000` | TRUECOLOR with RGB triplet |
| Color number | `color(196)` | STANDARD if 0-15, EIGHT_BIT if 16-255 |
| RGB format | `rgb(255,0,0)` | TRUECOLOR with RGB triplet |
| Default | `default` | ColorType::DEFAULT |

**Regex Pattern:**
```
^#([0-9a-f]{6})$|color\(([0-9]{1,3})\)$|rgb\(([\d\s,]+)\)$
```

**Parsing Rules:**
- Input is lowercased and trimmed
- Whitespace allowed in rgb() format
- Color numbers must be <= 255
- RGB components must be <= 255
- Results cached with LRU cache (max 1024 entries)

### 1.3 Color Palettes

#### STANDARD_PALETTE (16 colors)

```
Index  RGB
0      (0,     0,     0)      # Black
1      (170,   0,     0)      # Red
2      (0,     170,   0)      # Green
3      (170,   85,    0)      # Yellow
4      (0,     0,     170)    # Blue
5      (170,   0,     170)    # Magenta
6      (0,     170,   170)    # Cyan
7      (170,   170,   170)    # White
8      (85,    85,    85)     # Bright Black (Gray)
9      (255,   85,    85)     # Bright Red
10     (85,    255,   85)     # Bright Green
11     (255,   255,   85)     # Bright Yellow
12     (85,    85,    255)    # Bright Blue
13     (255,   85,    255)    # Bright Magenta
14     (85,    255,   255)    # Bright Cyan
15     (255,   255,   255)    # Bright White
```

#### EIGHT_BIT_PALETTE (256 colors)

- Indices 0-15: Same as STANDARD_PALETTE
- Indices 16-231: 6x6x6 RGB color cube (216 colors)
    - Grid: 6 levels per component (0, 95, 135, 175, 215, 255)
    - Index formula: `16 + 36 * red_index + 6 * green_index + blue_index`
- Indices 232-255: Grayscale ramp (24 shades)
    - Index 232: (8, 8, 8) ... Index 255: (238, 238, 238)

#### WINDOWS_PALETTE (16 colors)

```
Index  RGB
0      (12,    12,    12)     # Black
1      (197,   15,    31)     # Red
2      (19,    161,   14)     # Green
3      (193,   156,   0)      # Yellow
4      (0,     55,    218)    # Blue
5      (136,   23,    152)    # Magenta
6      (58,    150,   221)    # Cyan
7      (204,   204,   204)    # White
8      (118,   118,   118)    # Bright Black
9      (231,   72,    86)     # Bright Red
10     (22,    198,   12)     # Bright Green
11     (249,   241,   165)    # Bright Yellow
12     (59,    120,   255)    # Bright Blue
13     (180,   0,     158)    # Bright Magenta
14     (97,    214,   214)    # Bright Cyan
15     (242,   242,   242)    # Bright White
```

### 1.4 Color Conversion Algorithms

#### RGB to 8-bit (TRUECOLOR -> EIGHT_BIT)

**Grayscale Detection:** Convert RGB to HLS, check if saturation < 0.15:
- If grayscale, use luminance-based mapping to indices 232-255

**Color Cube Mapping:** For non-grayscale:
```
for each component in [red, green, blue]:
    if component < 95:
        quantized = component / 95
    else:
        quantized = 1 + (component - 95) / 40
    quantized_index = round(quantized)  // 0-5

color_number = 16 + 36 * red_idx + 6 * green_idx + blue_idx
```

#### RGB to Standard (-> STANDARD)

Use weighted CIE76 color distance formula:
```
red_mean = (r1 + r2) / 2
distance = sqrt(
    (((512 + red_mean) * red_diff^2) >> 8)
    + 4 * green_diff^2
    + (((767 - red_mean) * blue_diff^2) >> 8)
)
```

### 1.5 ANSI Code Generation

| ColorType | Foreground | Background |
|-----------|-----------|-----------|
| DEFAULT | ["39"] | ["49"] |
| STANDARD (0-7) | ["30"+n] | ["40"+n] |
| STANDARD (8-15) | ["82"+n] | ["92"+n] |
| EIGHT_BIT | ["38", "5", "N"] | ["48", "5", "N"] |
| TRUECOLOR | ["38", "2", "R", "G", "B"] | ["48", "2", "R", "G", "B"] |

---
