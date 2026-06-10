"""
Test the FoldSeek structure kernel on 49 PDB files spanning ~10 fold families.
Checks that within-family distances are consistently lower than cross-family distances.
"""

from refnd.kernels.protein.foldseek import FoldseekKernel, load_structures
from refnd.kernels.protein.sequence import CoverageMode, LocalIdentityMode

PDB_DIR = "tmp"

# Known structural families for annotating results
FAMILIES = {
    "globin":     {"1MBO","1A6M","1HBB","1MBA","1MBD","1VXF","2JHO","1EBD","3SDH","1LHB"},
    "lysozyme":   {"1LYZ","1HEW","132L","3LZM","1AZF"},
    "cytochrome": {"1CYC","1YCC","2YCC","1HRC","3CYT"},
    "inhibitor":  {"4PTI","1BBI","2OVO","1CSO"},
    "ubiquitin":  {"1UBQ","1AAR"},
    "sh3":        {"1CKA","1PGA","1TEN"},
    "tim_barrel": {"1TIM","2YPI","1HGX"},
    "thioredoxin":{"2TRX","1XOB"},
    "rubredoxin": {"1CAA","1IRO"},
}

def family_of(label):
    for fam, members in FAMILIES.items():
        if label in members:
            return fam
    return "other"


def main():
    print(f"Loading structures from '{PDB_DIR}' …")
    structures = load_structures(PDB_DIR)
    print(f"Loaded {len(structures)} structures (skipped files printed above as warnings)\n")

    kernel = FoldseekKernel(
        min_coverage=0.3,
        cov_mode=CoverageMode.ShorterSeq,
        identity_mode=LocalIdentityMode.AlignmentLength,
    )

    # Compute all pairwise distances
    n = len(structures)
    labels = [l for l, _ in structures]
    dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            li, di = structures[i]
            lj, dj = structures[j]
            dists[(li, lj)] = kernel.call(di, dj)

    # ── Summary: within-family vs cross-family ────────────────────────────────
    within, cross = [], []
    for (li, lj), d in dists.items():
        fi, fj = family_of(li), family_of(lj)
        if fi == fj and fi != "other":
            within.append((d, li, lj, fi))
        else:
            cross.append((d, li, lj, fi, fj))

    within.sort()
    cross.sort()

    print("── Within-family pairs (should be LOW) ─────────────────────────────")
    for d, li, lj, fam in within:
        print(f"  [{fam:12s}]  {li} ↔ {lj:<8}  {d:.4f}")

    print(f"\n── Cross-family sample — 20 closest (should be HIGHER) ─────────────")
    for d, li, lj, fi, fj in cross[:20]:
        print(f"  {fi:12s} ↔ {fj:12s}  {li} ↔ {lj:<8}  {d:.4f}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    if within and cross:
        avg_w = sum(d for d,*_ in within) / len(within)
        avg_c = sum(d for d,*_ in cross)  / len(cross)
        print(f"\nAverage within-family distance : {avg_w:.4f}")
        print(f"Average cross-family distance  : {avg_c:.4f}")
        print(f"Separation ratio               : {avg_c/avg_w:.2f}×")


if __name__ == "__main__":
    main()
