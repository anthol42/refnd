## Structure
```
kernels/molecules/
  tanimoto/
    binary.rs      BinaryTanimoto — Tanimoto distance on binary fingerprints (FixedBitSet)
    continuous.rs  ContinuousTanimoto — Tanimoto distance on real-valued fingerprints (Vec<f32>)
```

## Formulas

**Binary** (Jaccard distance):
```
distance = 1 - |A ∩ B| / |A ∪ B|
         = 1 - popcount(A & B) / (|A| + |B| - popcount(A & B))
```

**Continuous**:
```
distance = 1 - x·y / (‖x‖² + ‖y‖² − x·y)
```

## Types

**`BitFingerprint`** — wraps a `FixedBitSet` with a precomputed `count` (popcount of the full set).
Constructed via `BitFingerprint::new(bits: FixedBitSet)`.

**`RealFingerprint`** — wraps a `Vec<f32>` with a precomputed `norm_sq` (‖x‖²).
- `RealFingerprint::new(data: Vec<f32>)` — constructs from a vec.
- `RealFingerprint::from_array(arr: Array1<f32>)` — zero-copy move from a numpy/ndarray array into the vec (requires contiguous layout).
- `RealFingerprint::to_array()` — returns a cloned `Array1<f32>`.

## Performance notes

Both kernels are designed for HNSW hot loops where one query is compared against many candidates.

- `BitFingerprint` precomputes `popcount` at construction; the distance call iterates word-by-word over the raw `u32` blocks accumulating `popcount(a & b)`, with no allocation. `u32::count_ones()` compiles to the `POPCNT` CPU instruction when `target-cpu=native` is set (see below).
- `RealFingerprint` precomputes `norm_sq` at construction; the distance call only computes the dot product and reads `a.norm_sq` / `b.norm_sq` directly — no redundant norm computation.
- Both distance structs are `#[inline(always)]` to allow the compiler to fuse calls into the HNSW neighbor loop.

## Enabling POPCNT and SIMD

By default rustc targets a generic x86-64 baseline that excludes `POPCNT` and `AVX2`. Add this to `.cargo/config.toml` to enable all native CPU features:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

Or for a portable binary that only enables `POPCNT`:
```toml
[build]
rustflags = ["-C", "target-feature=+popcnt"]
```
