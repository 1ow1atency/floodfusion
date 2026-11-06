"""Evaluate a checkpoint on full chips: metrics.json (global) + per_chip.csv.

    python scripts/evaluate.py --checkpoint <run>/best.pt --split test
Outputs go to <run>/eval_<split>/ unless --out is given.
"""

from __future__ import annotations

import argparse

from floodfusion.evaluation import EVAL_SPLITS, evaluate


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", required=True, choices=EVAL_SPLITS)
    p.add_argument("--root", help="override the data root stored in the checkpoint's config")
    p.add_argument("--out", help="output folder (default: <checkpoint folder>/eval_<split>)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--allow-missing", action="store_true", help="drop chips with missing files (local sample only)")
    args = p.parse_args(argv)
    evaluate(args.checkpoint, args.split, root=args.root, out_dir=args.out, batch_size=args.batch_size,
             allow_missing=args.allow_missing)


if __name__ == "__main__":
    main()
