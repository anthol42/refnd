# CLAUDE.md

## Mindset

**Short and sweet.**
- *Short*: concise code, no repetition, no boilerplate, no speculative abstractions.
- *Sweet*: readable and maintainable — the next reader should understand intent immediately.

Write like a software engineer: correct, efficient, well-structured. But never write more than the task requires.

---

## What this is

A RGP dataset toolkit, mainly designed to split dataset. A RGP is a data generation process which generates clusters of 
samples that are related together. In those cases, the classic random train test split is of no use because it induces 
data leakage (related samples can be in different splits (train and test)). We build a framework in rust that is 
generic on the data type and the distance function. 

The core of the algorithm is a HNSW index generic on the Data type and Distance function, and a leiden community 
detection algorithm. These core data structures and algorithms along with traits are implemented in `core/`. 
Pre-implemented distance functions are implemented in `kernels/`. Some utils functions are implemented in `utils/`.

---

## Structure

```
src/
  main.rs                        Entry point referencing to the `cli` module
  cli/                           Command Line Interace code
  core/
    hnsw/                        HNSW implementation
    leiden/                      Leiden implementation
    distance.rs                  Distance<T> trait
  kernels/
    proteins/                    Protein sequence-related kernels
  utils/
    read_fasta.rs                FASTA parser
```

---

## Adding a new distance kernel

1. Implement `Distance<T>` from `core` in `kernels/`.
2. Pass it to `HNSWState::new`.

That's it — no changes to the HNSW core.

## Get additional context on the code
Most self-contained directories such as `hnsw`, `leiden`, `kernels/*` contains their own .md file adding additional 
information on the code structure, and with a more local perspective on the code. Read them before making changes in 
those directories.

## ALWAYS: After a change
After a change, always read the CLAUDE.md or AGENTS.md and the respective directory .md file to see if it needs to be 
updated to reflect the changes. Note that CLAUDE.md or AGENTS.md must be very concise, and provide a global view of the 
system. Respective .md files in the self-contained directories can contain more text to provide a more local view on 
the code base.