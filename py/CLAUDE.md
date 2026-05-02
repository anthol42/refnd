# CLAUDE.md

## Mindset

**Short and sweet.**
- *Short*: concise code, no repetition, no boilerplate, no speculative abstractions.
- *Sweet*: readable and maintainable — the next reader should understand intent immediately.

Write like a software engineer: correct, efficient, well-structured. But never write more than the task requires.

---

## What this is

This is a python-binding for the `../proto-core` crate. It gives access to the public rust api from the python 
scripting language. 

---

## Structure

```
src/
  core/
    hnsw/                        HNSW api
    leiden/                      Leiden api
    exact/                       Exact api
    functional.rs                Other core functions
  kernels/
    proteins                     Protein sequence-related kernels
  utils.rs                       Utilitary functions (Ex: read_fasta)
```

---

## Get additional context on the code
Read `../proto-core`

## Test the implementations
A uv project is created in `../pytests` that install this package and is designed solely to test the python-binding 
implementation.

To test the implementation, run:
```shell
cd ../pytests; uv add --editable ../py-proto; <command>
```
This will build the project and install it in the python environment.

---

## Adding bindings: cookbook

After any change, regenerate stubs with:
```shell
cargo run --bin stub_gen
```
This rewrites all `python/py_proto/**/*.pyi` files and `python/py_proto/__init__.py`.

---

example for the `py_proto.kernels.protein.sequence` sub-module, adjust the paths.
### Enum

```rust
// src/kernels/protein/sequence.rs
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, module = "py_proto.kernels.protein.sequence")]
#[derive(Clone, Copy, PartialEq)]
pub enum MyMode { VariantA, VariantB }
```

```rust
// src/lib.rs — inside mod sequence { ... }
#[pymodule_export]
use crate::kernels::protein::sequence::MyMode;
```

---

### Struct (class)

```rust
// src/kernels/protein/sequence.rs
#[gen_stub_pyclass]
#[pyclass(module = "py_proto.kernels.protein.sequence")]
pub struct MyAligner { pub inner: CoreAligner }

#[gen_stub_pymethods]
#[pymethods]
impl MyAligner {
    #[new]
    fn new() -> Self { ... }
    fn call(&self, a: &str, b: &str) -> f32 { ... }
}
```

```rust
// src/lib.rs — inside mod sequence { ... }
#[pymodule_export]
use crate::kernels::protein::sequence::MyAligner;
```

---

### Function

```rust
// src/kernels/protein/sequence.rs
#[gen_stub_pyfunction(module = "py_proto.kernels.protein.sequence")]
#[pyfunction]
pub fn my_function(x: f32) -> f32 { x * 2.0 }
```

```rust
// src/lib.rs — inside mod sequence { ... }
#[pymodule_export]
use crate::kernels::protein::sequence::my_function;
```

---

### Nested submodule (e.g. `py_proto.kernels.protein.pairwise`)

**1.** Create `src/kernels/protein/pairwise.rs` with annotated items (same rules as above, `module = "py_proto.kernels.protein.pairwise"`).

**2.** Add `pub mod pairwise;` to `src/kernels/protein/mod.rs`.

**3.** In `src/lib.rs`, add the `#[pymodule]` block inside `mod protein` and a `sys.modules` entry in `#[pymodule_init]`:

```rust
#[pymodule_init]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ...existing entries
    let pairwise = protein.getattr("pairwise")?;
    modules.set_item("py_proto.kernels.protein.pairwise", &pairwise)?;
    Ok(())
}

mod protein {
    #[pymodule]
    mod pairwise {
        #[pymodule_export]
        use crate::kernels::protein::pairwise::MyItem;
    }
    // ...existing submodules
}
```

---

### Top-level submodule (e.g. `py_proto.core`)

**1.** Create `src/core/mod.rs` with annotated items (`module = "py_proto.core"`).

**2.** Add `pub mod core;` at the top of `src/lib.rs`.

**3.** In `src/lib.rs`, add the `#[pymodule]` block and `sys.modules` entry:

```rust
#[pymodule_init]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ...existing entries
    let core = m.getattr("core")?;
    modules.set_item("py_proto.core", &core)?;
    Ok(())
}

mod core {
    #[pymodule_export]
    use crate::core::MyItem;
}
```

**4.** In `src/bin/stub_gen.rs`, add the submodule to the generated `__init__.py`:

```rust
std::fs::write(
    "python/py_proto/__init__.py",
    "from .py_proto import *\n\
     from .py_proto import kernels\n\
     from .py_proto import core\n",   // ← add one line per top-level submodule
)?;
```

---

### Why `stub_gen.rs` only needs updating for top-level submodules

`from .py_proto import *` loads the extension and triggers `#[pymodule_init]`, which registers all dotted paths in `sys.modules`. The explicit `from .py_proto import core` is additionally needed so `core` is an attribute of the `py_proto` package object itself. Deeper submodules (`kernels.protein.sequence`, etc.) are reachable through their parent once the top-level is exposed — no extra line needed.
