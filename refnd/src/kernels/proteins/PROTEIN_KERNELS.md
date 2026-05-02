## Structure
```
src/
  main.rs                        CLI: parse args, build index, write edgelist
  kernels/proteins/parasail/     Implemented distance functions
      global.rs                  GlobalAligner  — Needleman-Wunsch identity
      local.rs                   LocalAligner   — Smith-Waterman identity
      aligner_config.rs          Shared alignment parameters
      matrix.rs                  Substitution matrix enum (BLOSUM/PAM)
```