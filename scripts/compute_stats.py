"""Compute train-split band stats (clip limits, mean, std) and descriptive label stats.

Writes <out> (band stats, the format Sen1Floods11(stats=...) expects) and
label_stats.json next to it.

    python scripts/compute_stats.py --root data/sen1floods11 --out outputs/band_stats.json
"""

from __future__ import annotations

import argparse

from floodfusion.stats import (
    compute_band_stats, compute_label_stats, format_band_table, format_label_table, save_json,
)
from floodfusion.utils import seed_everything


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Sen1Floods11 root (S1Hand/, S2Hand/, LabelHand/, splits/)")
    p.add_argument("--out", required=True, help="output JSON path for band stats")
    p.add_argument("--allow-missing", action="store_true",
                   help="drop chips with missing files (local sample only; never on full data)")
    p.add_argument("--seed", type=int, default=0, help="seed for the pass-1 pixel subsample")
    args = p.parse_args(argv)

    seed_everything(args.seed)
    band_stats = compute_band_stats(args.root, allow_missing=args.allow_missing, seed=args.seed)
    out = save_json(band_stats, args.out)
    label_stats = compute_label_stats(args.root, allow_missing=args.allow_missing)
    label_out = save_json(label_stats, out.with_name("label_stats.json"))

    print("\nBand stats (train split; clip, then normalise with mean/std):")
    print(format_band_table(band_stats))
    print("\nLabel stats (descriptive only):")
    print(format_label_table(label_stats))
    print(f"\nwrote {out}\nwrote {label_out}")


if __name__ == "__main__":
    main()
