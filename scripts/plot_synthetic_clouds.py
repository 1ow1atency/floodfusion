"""Figure to check synthetic clouds against real ones (floodfusion.clouds_synth).

Row 1: thick clouds, row 2: thin (cirrus-like) clouds, at coverages 0, 20, ..., 100% on
the 6 clearest available train chips (real cloud fraction from --cloud-csv). Row 3: real
train chips whose s2cloudless cloud fraction is closest to the same coverages.
Deterministic: cloudify(s2, coverage, seed + i).

    python scripts/plot_synthetic_clouds.py --root data/sen1floods11 --allow-missing
"""

from __future__ import annotations

import argparse

from floodfusion.chips import list_split_chips, load_raw
from floodfusion.clouds import read_cloud_csv
from floodfusion.clouds_synth import cloudify
from floodfusion.viz import plot_true_colour_grid

COVERAGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Sen1Floods11 root")
    p.add_argument("--cloud-csv", default="docs/cloud_cover.csv", help="real cloud fractions (compute_cloud_cover.py)")
    p.add_argument("--out", default="outputs/figures/synthetic_clouds.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-missing", action="store_true",
                   help="drop chips with missing files (local sample only; never on full data)")
    args = p.parse_args(argv)

    ids = set(list_split_chips(args.root, "train", args.allow_missing))
    real = sorted((r["cloud_fraction"], r["chip_id"]) for r in read_cloud_csv(args.cloud_csv)
                  if r["chip_id"] in ids and r["cloud_fraction"] == r["cloud_fraction"])  # skip NaN
    if len(real) < len(COVERAGES):
        raise SystemExit(f"need >= {len(COVERAGES)} train chips with a cloud fraction, found {len(real)}")
    clear = [cid for _, cid in real[:len(COVERAGES)]]

    rows = []
    for thin, label in ((False, "synthetic: thick"), (True, "synthetic: thin / cirrus")):
        panels = []
        for i, (cov, cid) in enumerate(zip(COVERAGES, clear)):
            s2 = cloudify(load_raw(args.root, cid, "s2"), cov, seed=args.seed + i, thin=thin)
            panels.append((f"{cid}\ntarget cover {cov:.0%}", s2))
        rows.append((label, panels))
    picked, panels = set(), []
    for cov in COVERAGES:
        frac, cid = min(((f, c) for f, c in real if c not in picked), key=lambda fc: abs(fc[0] - cov))
        picked.add(cid)
        panels.append((f"{cid}\nreal, s2cloudless {frac:.0%}", load_raw(args.root, cid, "s2")))
    rows.append(("real cloudy chips", panels))
    out = plot_true_colour_grid(rows, args.out, "Synthetic vs real clouds (train chips)")
    print(f"wrote {out}")
    print("synthetic rows use chips:", ", ".join(clear))


if __name__ == "__main__":
    main()
