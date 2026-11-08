"""Robustness evaluation of trained runs on test and bolivia (floodfusion.robustness).

FINAL REPORTING ONLY: never use these results to tune hyperparameters or select models.

Evaluates each run's best.pt under synthetic cloud coverage 0-100% (3 cloud seeds, identical
clouds for every model) for thick clouds (as trained, semi-transparent) and opaque clouds (same
masks, opacity 1: the stricter test), with S2 or S1 removed, and per real-cloud bin. Checks that
coverage 0 reproduces eval_<split>/metrics.json exactly and that opaque clouds at 0% equal clean.
Writes robustness.csv, summary.md and figures.

    python scripts/robustness_eval.py --runs /content/drive/MyDrive/floodfusion/runs \\
        --root /content/data/sen1floods11 --out /content/drive/MyDrive/floodfusion/results
"""

from __future__ import annotations

import argparse

from floodfusion.robustness import SPLITS, find_runs, run_robustness


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", help="folder of run folders; every <runs>/*/best.pt is evaluated")
    p.add_argument("--run", action="append", default=[], help="explicit run folder (repeatable; instead of --runs)")
    p.add_argument("--root", required=True, help="Sen1Floods11 root")
    p.add_argument("--out", default="outputs/results/", help="output folder")
    p.add_argument("--cloud-csv", default="docs/cloud_cover.csv", help="real cloud fractions per chip")
    p.add_argument("--splits", default=",".join(SPLITS), help="comma-separated, from test,bolivia")
    p.add_argument("--batch-size", type=int, default=4,
                   help="keep 4 (as evaluate.py) so coverage 0 reproduces eval_<split> exactly")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-dir", default=None, help="keep generated cloud layers here for later runs (local disk)")
    p.add_argument("--qual-split", default="test", help="split of the qualitative figure")
    p.add_argument("--allow-missing", action="store_true", help="local sample only; never on full data")
    p.add_argument("--allow-smoke", action="store_true", help="accept smoke-test checkpoints (pipeline check only)")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    runs = list(args.run) or (find_runs(args.runs) if args.runs else [])
    if not runs:
        p.error("no runs: give --runs <folder> (with */best.pt) or --run <run folder>")
    run_robustness(runs, args.root, args.out, cloud_csv=args.cloud_csv,
                   splits=[s.strip() for s in args.splits.split(",") if s.strip()], batch_size=args.batch_size,
                   num_workers=args.num_workers, cache_dir=args.cache_dir, allow_missing=args.allow_missing,
                   allow_smoke=args.allow_smoke, qual_split=args.qual_split, figures=not args.no_figures)


if __name__ == "__main__":
    main()
