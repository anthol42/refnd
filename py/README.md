# Refnd Python

## Build Package
1. Generate stubs
```shell
cargo run --bin stub_gen
```
2. Build package
```shell
uv build
```
## Build Docs
After generating stubs, run this command:
```bash
uv run --group docs maturin develop && rm -rf docs/build && uv run --group docs sphinx-build docs/source docs/build/html
open docs/build/html/index.html
```
