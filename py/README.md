# py-proto

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
```bash
cd proto/py-proto
uv run sphinx-build docs/source docs/build/html
open docs/build/html/index.html
```
