"""Measure how much ground signal survives under the synthetic evaluation clouds
(floodfusion.cloud_leakage): opacity inside the cloud mask, clouded-vs-original correlation per
band, and NDWI / MNDWI water-vs-land separability, for thick (as trained) and opaque clouds.

Uses only chips with little real cloud (their ground is visible). Local sample:

    python scripts/cloud_leakage.py --root data/sen1floods11 --allow-missing --out docs/cloud_leakage.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from floodfusion.chips import SPLITS, list_split_chips
from floodfusion.cloud_leakage import measure, summary_markdown
from floodfusion.clouds import read_cloud_csv


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Sen1Floods11 root")
    p.add_argument("--cloud-csv", default="docs/cloud_cover.csv", help="real cloud fractions per chip")
    p.add_argument("--max-real-cloud", type=float, default=0.1, help="use chips with less real cloud than this")
    p.add_argument("--coverages", default="0.5,1.0")
    p.add_argument("--cloud-seeds", default="0,1,2")
    p.add_argument("--allow-missing", action="store_true", help="local sample only")
    p.add_argument("--out", default="outputs/cloud_leakage.md")
    args = p.parse_args(argv)

    frac = {r["chip_id"]: r["cloud_fraction"] for r in read_cloud_csv(args.cloud_csv)}
    ids = sorted({c for s in SPLITS for c in list_split_chips(args.root, s, allow_missing=args.allow_missing)
                  if c in frac and not math.isnan(frac[c]) and frac[c] < args.max_real_cloud})
    if not ids:
        p.error("no chips with a cloud fraction below --max-real-cloud")
    covs = [float(c) for c in args.coverages.split(",")]
    seeds = [int(s) for s in args.cloud_seeds.split(",")]
    print(f"{len(ids)} chips with < {args.max_real_cloud:.0%} real cloud; coverages {covs}; cloud seeds {seeds}")
    md = summary_markdown(measure(args.root, ids, covs, seeds), ids, seeds, args.max_real_cloud)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n")
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
