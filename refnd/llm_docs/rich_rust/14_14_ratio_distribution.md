## 14. Ratio Distribution

> Source: `rich/_ratio.py` (154 lines)

### 14.1 Edge Protocol

```rust
/// Trait for items that participate in ratio-based distribution
trait Edge {
    fn size(&self) -> Option<usize>;       // Fixed size (None = flexible)
    fn ratio(&self) -> usize;              // Ratio weight (default 1)
    fn minimum_size(&self) -> usize;       // Minimum allowed size (default 1)
}
```

### 14.2 Ratio Resolution Algorithm

This algorithm distributes a total amount among edges based on their ratios:

```rust
use num_rational::Ratio;  // For exact fraction arithmetic

/// Resolve sizes for edges with no fixed size
fn ratio_resolve(total: usize, edges: &[impl Edge]) -> Vec<usize> {
    // Separate fixed and flexible edges
    let mut sizes = vec![0usize; edges.len()];
    let mut flexible_indices = Vec::new();
    let mut fixed_total = 0;
    let mut total_ratio = 0;

    for (i, edge) in edges.iter().enumerate() {
        if let Some(size) = edge.size() {
            sizes[i] = size;
            fixed_total += size;
        } else {
            flexible_indices.push(i);
            total_ratio += edge.ratio();
        }
    }

    // Calculate remaining space for flexible edges
    let remaining = total.saturating_sub(fixed_total);

    if total_ratio == 0 || remaining == 0 {
        // No flexible edges or no space
        for i in flexible_indices {
            sizes[i] = edges[i].minimum_size();
        }
        return sizes;
    }

    // Distribute using exact fractions to avoid rounding errors
    let mut distributed = 0;
    for (idx, &i) in flexible_indices.iter().enumerate() {
        let ratio = Ratio::new(edges[i].ratio(), total_ratio);
        let ideal = ratio * remaining;

        // Round (using nearest integer)
        let size = if idx == flexible_indices.len() - 1 {
            // Last flexible edge gets remainder (avoids accumulation error)
            remaining - distributed
        } else {
            ideal.round().to_integer()
        };

        sizes[i] = size.max(edges[i].minimum_size());
        distributed += sizes[i];
    }

    sizes
}
```

### 14.3 Ratio Reduction Algorithm

When total required exceeds available, reduce proportionally:

```rust
/// Reduce sizes proportionally to fit within total
fn ratio_reduce(
    total: usize,
    ratios: &[usize],
    maximums: &[usize],
    values: &[usize],
) -> Vec<usize> {
    let current_total: usize = values.iter().sum();
    if current_total <= total {
        return values.to_vec();
    }

    let excess = current_total - total;

    // Calculate how much each can shrink (value - 1, weighted by ratio)
    let shrinkable: Vec<usize> = values.iter()
        .zip(ratios.iter())
        .map(|(&v, &r)| (v.saturating_sub(1)) * r)
        .collect();

    let total_shrinkable: usize = shrinkable.iter().sum();
    if total_shrinkable == 0 {
        return values.to_vec();
    }

    // Reduce proportionally
    let mut result = values.to_vec();
    let mut reduced = 0;

    for i in 0..values.len() {
        if shrinkable[i] > 0 {
            let share = Ratio::new(shrinkable[i], total_shrinkable);
            let reduction = (share * excess).round().to_integer().min(values[i] - 1);
            result[i] = values[i] - reduction;
            reduced += reduction;
        }
    }

    // Handle rounding errors by reducing largest values
    while result.iter().sum::<usize>() > total {
        // Find largest value that can still be reduced
        if let Some(i) = result.iter().enumerate()
            .filter(|(_, &v)| v > 1)
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
        {
            result[i] -= 1;
        } else {
            break;
        }
    }

    result
}
```

### 14.4 Ratio Distribution Algorithm

Distribute extra space among ratio-enabled edges:

```rust
/// Distribute remaining space among edges based on ratio
fn ratio_distribute(
    total: usize,
    edges: &[impl Edge],
    minimums: &[usize],
) -> Vec<usize> {
    let mut sizes = minimums.to_vec();
    let current: usize = sizes.iter().sum();

    if current >= total {
        return sizes;
    }

    let remaining = total - current;

    // Get ratio for flexible edges (ratio > 0)
    let ratios: Vec<usize> = edges.iter()
        .zip(sizes.iter())
        .map(|(e, &s)| if e.ratio() > 0 && s < total { e.ratio() } else { 0 })
        .collect();

    let total_ratio: usize = ratios.iter().sum();
    if total_ratio == 0 {
        return sizes;
    }

    // Distribute using fractions
    let mut distributed = 0;
    let flexible_count = ratios.iter().filter(|&&r| r > 0).count();
    let mut flex_idx = 0;

    for (i, &ratio) in ratios.iter().enumerate() {
        if ratio > 0 {
            flex_idx += 1;
            let share = Ratio::new(ratio, total_ratio);
            let extra = if flex_idx == flexible_count {
                remaining - distributed
            } else {
                (share * remaining).round().to_integer()
            };
            sizes[i] += extra;
            distributed += extra;
        }
    }

    sizes
}
```

---
