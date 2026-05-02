## 2. Style System

> Source: `rich/style.py` (792 lines)

### 2.1 Style Data Structure

```rust
struct Style {
    color: Option<Color>,           // Foreground color
    bgcolor: Option<Color>,         // Background color
    attributes: u16,                // Bit flags for enabled attributes (13 bits)
    set_attributes: u16,            // Bit flags for which attributes are explicitly set
    link: Option<String>,           // URL for hyperlinks
    link_id: String,                // Random ID for hyperlink tracking
    meta: Option<Vec<u8>>,          // Serialized metadata
    null: bool,                     // True if this is an empty/null style
}
```

### 2.2 Style Attributes (Bitflags)

| Bit | Attribute    | SGR Code | Meaning |
|-----|--------------|----------|---------|
| 0   | bold         | 1        | Bold/bright text |
| 1   | dim          | 2        | Dim/faint text |
| 2   | italic       | 3        | Italic text |
| 3   | underline    | 4        | Single underline |
| 4   | blink        | 5        | Blinking text (slow) |
| 5   | blink2       | 6        | Fast blinking text |
| 6   | reverse      | 7        | Reverse video |
| 7   | conceal      | 8        | Concealed/hidden text |
| 8   | strike       | 9        | Strikethrough text |
| 9   | underline2   | 21       | Double underline |
| 10  | frame        | 51       | Framed text |
| 11  | encircle     | 52       | Encircled text |
| 12  | overline     | 53       | Overlined text |

**Attribute Aliases for Parsing:**
```
bold -> "bold", "b"
dim -> "dim", "d"
italic -> "italic", "i"
underline -> "underline", "u"
reverse -> "reverse", "r"
conceal -> "conceal", "c"
strike -> "strike", "s"
underline2 -> "underline2", "uu"
overline -> "overline", "o"
```

### 2.3 Style Parsing

Supported style string formats:

| Format | Example | Result |
|--------|---------|--------|
| Empty/Null | `""` or `"none"` | NULL_STYLE |
| Attribute | `"bold"`, `"italic"` | Enable attribute |
| Negative | `"not bold"` | Disable attribute |
| Color | `"red"`, `"#ff0000"` | Set foreground |
| Background | `"on red"`, `"on #ff0000"` | Set background |
| Link | `"link https://..."` | Set hyperlink |
| Combined | `"bold red on white"` | Multiple properties |

### 2.4 Style Combination Logic (`style1 + style2`)

```rust
fn combine(self, other: Style) -> Style {
    if other.is_null() { return self; }
    if self.is_null() { return other; }

    Style {
        color: other.color.or(self.color),
        bgcolor: other.bgcolor.or(self.bgcolor),
        attributes: (self.attributes & !other.set_attributes)
                  | (other.attributes & other.set_attributes),
        set_attributes: self.set_attributes | other.set_attributes,
        link: other.link.or(self.link),
        meta: merge(self.meta, other.meta),  // other overwrites
    }
}
```

**Rules:**
1. `style2.color` overrides if set, else keep `style1.color`
2. `style2.bgcolor` overrides if set, else keep `style1.bgcolor`
3. For attributes: if `style2.set_attributes[bit] == 1`, use `style2.attributes[bit]`
4. `style2.link` overrides if set

### 2.5 ANSI Code Generation

```rust
fn make_ansi_codes(&self, color_system: ColorSystem) -> String {
    let mut codes = Vec::new();

    // Enabled attributes
    for (bit, sgr) in STYLE_MAP {
        if self.attributes & self.set_attributes & (1 << bit) != 0 {
            codes.push(sgr);
        }
    }

    // Foreground color
    if let Some(color) = &self.color {
        codes.extend(color.downgrade(color_system).get_ansi_codes(true));
    }

    // Background color
    if let Some(bgcolor) = &self.bgcolor {
        codes.extend(bgcolor.downgrade(color_system).get_ansi_codes(false));
    }

    codes.join(";")
}
```

Final ANSI sequence: `"\x1b[" + codes + "m" + text + "\x1b[0m"`

### 2.6 Hyperlink Support

OSC 8 hyperlink protocol:
```
"\x1b]8;id={link_id};{url}\x1b\\{text}\x1b]8;;\x1b\\"
```

### 2.7 StyleStack

```rust
struct StyleStack {
    stack: Vec<Style>,
}

impl StyleStack {
    fn new(default: Style) -> Self { Self { stack: vec![default] } }
    fn current(&self) -> &Style { self.stack.last().unwrap() }
    fn push(&mut self, style: Style) {
        self.stack.push(self.current().clone() + style);
    }
    fn pop(&mut self) -> &Style {
        self.stack.pop();
        self.current()
    }
}
```

---
