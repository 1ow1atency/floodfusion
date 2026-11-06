"""Save per-chip panels (VV, VH, S2 true/false colour, label) and an S1 class histogram.

    python scripts/visualize_sample.py --root data/sen1floods11 --split train --index 0 --n 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from floodfusion.chips import SPLITS, list_split_chips
from floodfusion.viz import plot_chip, plot_s1_histograms, s1_class_histograms


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Sen1Floods11 root")
    p.add_argument("--split", default="train", choices=SPLITS)
    p.add_argument("--index", type=int, default=0, help="first chip (position in the split CSV)")
    p.add_argument("--n", type=int, default=1, help="number of chips to plot (0 = histogram only)")
    p.add_argument("--stats", default="configs/band_stats.json", help="band stats JSON (for clip limits)")
    p.add_argument("--out", default="outputs/figures/", help="output folder")
    p.add_argument("--allow-missing", action="store_true",
                   help="drop chips with missing files (local sample only)")
    p.add_argument("--no-hist", action="store_true", help="skip the split-wide S1 histogram")
    args = p.parse_args(argv)

    stats = json.loads(Path(args.stats).read_text())
    ids = list_split_chips(args.root, args.split, args.allow_missing)
    if args.n < 0 or (args.n == 0 and args.no_hist):
        raise SystemExit("--n must be >= 0, and --n 0 (histogram only) cannot be combined with --no-hist")
    if args.n > 0 and not 0 <= args.index < len(ids):
        raise SystemExit(f"--index must be in [0, {len(ids) - 1}] "
                         f"(split '{args.split}' has {len(ids)} usable chips)")
    chosen = ids[args.index : args.index + args.n]
    if len(chosen) < args.n:
        print(f"only {len(chosen)} chips available from index {args.index}")

    out = Path(args.out)
    for i, cid in enumerate(chosen, start=args.index):
        path = plot_chip(args.root, cid, stats, out / f"{args.split}_{i:03d}_{cid}.png")
        print(f"wrote {path}")

    if not args.no_hist:
        hist = s1_class_histograms(args.root, ids, progress=True)
        path = plot_s1_histograms(hist, stats, out / f"{args.split}_s1_hist.png", title=f"split={args.split}")
        print(f"wrote {path}  ({hist['n_chips']} chips)")


if __name__ == "__main__":
    main()
