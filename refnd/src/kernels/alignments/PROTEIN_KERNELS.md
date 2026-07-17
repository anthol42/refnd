## Structure
```
kernels/alignments/
  parasail/                      Sequence alignment kernels (via libparasail FFI)
      global.rs                  GlobalAligner  — Needleman-Wunsch identity
      local.rs                   LocalAligner   — Smith-Waterman identity
      aligner_config.rs          Shared alignment parameters
      matrix.rs                  Substitution matrix enum (BLOSUM/PAM)
  usalign/                       Structure alignment kernels (via libusalign-sys FFI)
      kernel.rs                  USAlignKernel  — TM-score distance (1 - TM-score)
      config.rs                  NormMode enum (Min / Query / Target)
```

## USalign kernel
`USAlignKernel` implements `Distance<String>` where `String` is a path to a PDB file.
It calls `usalign_align()` from `libusalign-sys` (a workspace sys-crate that compiles
USalign's header-only C++ via a thin `extern "C"` wrapper).

`NormMode::Min` (default) returns `1.0 - min(TM1, TM2)`, i.e. the conservative score
independent of which chain is called query vs target.