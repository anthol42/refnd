#!/usr/bin/env python3
"""
Split a FASTA file into train/test using Hestia (sequence identity threshold).

Usage:
    python hestia_split.py input.fasta --threshold 0.5 --test-size 0.2
    python hestia_split.py input.fasta --threshold 0.4 --nucleotide
"""
import argparse
import os.path
import sys
import pandas as pd
from hestia.dataset_generator import HestiaGenerator, SimArguments

def read_fasta(path: str) -> list[tuple[str, str]]:
    records = []
    header, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def write_fasta(path: str, records: list[tuple[str, str]]):
    with open(path, "w") as f:
        for header, seq in records:
            f.write(f">{header}\n{seq}\n")


def main():
    parser = argparse.ArgumentParser(description="Hestia sequence-identity-based FASTA splitter")
    parser.add_argument("input", help="Input FASTA file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sequence identity threshold (default: 0.5)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--nucleotide", action="store_true",
                        help="Treat sequences as nucleotide (DNA/RNA) instead of protein")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", default="tmp/hestia_train.fasta")
    parser.add_argument("--test-out", default="tmp/hestia_test.fasta")
    args = parser.parse_args()
    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    records = read_fasta(args.input)
    if not records:
        print(f"Error: no sequences found in {args.input}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(records)} sequences from {args.input}")

    df = pd.DataFrame({"header": [h for h, _ in records],
                       "sequence": [s for _, s in records]})
    df["idx"] = df.index

    data_type = "dna sequence" if args.nucleotide else "sequence"

    gen = HestiaGenerator(df, verbose=False)
    gen.calculate_partitions(
        sim_args=SimArguments(data_type=data_type, field_name="sequence",
                              min_threshold=args.threshold),
        test_size=args.test_size,
        valid_size=0.0,
        random_state=args.seed,
        verbose=1,
    )
    parts = gen.get_partitions(return_dict=True)[args.threshold]

    train_idx = [int(x) for x in parts["train"]]
    test_idx  = [int(x) for x in parts["test"]]

    write_fasta(args.train_out, [records[i] for i in train_idx])
    write_fasta(args.test_out,  [records[i] for i in test_idx])

    print(f"Train: {len(train_idx)} sequences -> {args.train_out}")
    print(f"Test:  {len(test_idx)} sequences -> {args.test_out}")


if __name__ == "__main__":
    main()
