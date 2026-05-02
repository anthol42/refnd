## 4. Markup Parser

> Source: `rich/markup.py` (251 lines)

### 4.1 Markup Syntax

```
[tag_name]text[/tag_name]     # Basic tag
[/]                            # Close most recent tag
[tag=parameter]text[/tag]      # Tag with parameter
[bold red]text[/]              # Multiple styles
[@handler(args)]text[/@handler] # Metadata tag
```

**Tag Name Rules:**
- Must start with: `a-z`, `#`, `@`, or `/`
- Cannot contain `[` or `]`

### 4.2 Regex Patterns

**Main tag pattern:**
```regex
((\\*)\[([a-z#/@][^[]*?)])
```
- Group 1: Full match including escapes
- Group 2: Leading backslashes
- Group 3: Tag content

**Handler pattern:**
```regex
^([\w.]*?)(\(.*?\))?$
```

### 4.3 Parsing Algorithm

```rust
fn render(markup: &str) -> Text {
    // Optimization: if no '[', return plain text
    if !markup.contains('[') {
        return Text::new(markup);
    }

    let mut text = Text::new();
    let mut style_stack: Vec<(usize, Tag)> = Vec::new();

    for (position, plain_text, tag) in parse(markup) {
        if let Some(plain) = plain_text {
            // Replace escaped brackets
            let unescaped = plain.replace("\\[", "[");
            text.append(&unescaped);
        }

        if let Some(tag) = tag {
            if !tag.name.starts_with('/') {
                // Opening tag
                style_stack.push((text.len(), tag));
            } else {
                // Closing tag
                let style_name = &tag.name[1..].trim();
                let (start, open_tag) = if style_name.is_empty() {
                    // Implicit close [/]
                    style_stack.pop().ok_or(MarkupError)?
                } else {
                    // Explicit close [/name]
                    pop_matching(&mut style_stack, style_name)?
                };
                text.add_span(start, text.len(), &open_tag);
            }
        }
    }

    // Auto-close unclosed tags
    while let Some((start, tag)) = style_stack.pop() {
        text.add_span(start, text.len(), &tag);
    }

    text
}
```

### 4.4 Escape Sequences

| Input | Output |
|-------|--------|
| `\[` | Literal `[` |
| `\\[tag]` | Literal `\` + tag applied |
| `\\\[tag]` | Literal `\[tag]` (escaped) |

### 4.5 Tag Nesting

- Tags can nest arbitrarily deep
- `[/]` closes most recent tag (LIFO)
- `[/name]` closes specific tag (searches stack)
- Unclosed tags auto-close at end

### 4.6 Error Conditions

| Error | Message |
|-------|---------|
| `[/]` with empty stack | "closing tag '[/]' has nothing to close" |
| `[/name]` not found | "closing tag '[/name]' doesn't match any open tag" |

---
