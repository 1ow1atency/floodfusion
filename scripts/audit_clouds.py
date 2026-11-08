"""Audit the s2cloudless cloud masks behind the real-cloud bins (floodfusion.cloud_audit).

For every chip of every split: cloud fraction at thresholds 0.4 (current) / 0.5 / 0.6, how the
hand labels (-1 / land / water) split between cloudy and clear pixels, and mask diagnostics.
Saves the raw probability maps and figures (true colour, probability, masks per threshold) of
the 6 cloudiest and 6 mid-cloud train chips, plus Paraguay_215904 / Paraguay_36146 if present.
Deterministic. Needs the `clouds` extra (s2cloudless); never imports torch. Does not change
docs/cloud_cover.csv.

    python scripts/audit_clouds.py --root data/sen1floods11 --out outputs/cloud_audit --allow-missing
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from floodfusion.chips import load_raw
from floodfusion.cloud_audit import (
    NAMED_CHIPS, THRESHOLDS, audit_all, masks_at, pick_audit_chips, probability_map, summary_markdown, write_rows,
)
from floodfusion.clouds import AVERAGE_OVER, DILATION_SIZE, make_detector, read_cloud_csv
from floodfusion.viz import plot_cloud_probability


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Sen1Floods11 root")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--thresholds", default=",".join(f"{t:g}" for t in THRESHOLDS))
    p.add_argument("--cloud-csv", default="docs/cloud_cover.csv",
                   help="existing cloud fractions, to check that the current threshold reproduces them")
    p.add_argument("--n-cloudy", type=int, default=6)
    p.add_argument("--n-mid", type=int, default=6)
    p.add_argument("--allow-missing", action="store_true", help="local sample only; never on full data")
    args = p.parse_args(argv)

    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = make_detector()  # fail fast if s2cloudless is not installed
    print(f"s2cloudless: thresholds {list(thresholds)}, average_over={AVERAGE_OVER}, dilation_size={DILATION_SIZE}")
    rows = audit_all(args.root, detector, thresholds, allow_missing=args.allow_missing)
    csv_path = write_rows(rows, out / "chip_audit.csv")

    ref = None
    if args.cloud_csv and Path(args.cloud_csv).is_file():
        ref = {r["chip_id"]: r["cloud_fraction"] for r in read_cloud_csv(args.cloud_csv)}
    note = ("**Local sample only** (chips with missing files dropped): indicative, not the full data."
            if args.allow_missing else "")
    md = summary_markdown(rows, thresholds, reference=ref, note=note)
    (out / "summary.md").write_text(md + "\n")

    cloudy, mid = pick_audit_chips(rows, args.n_cloudy, args.n_mid)
    missing = [c for c in NAMED_CHIPS if c not in {r["chip_id"] for r in rows}]
    if missing:
        print(f"not in this data: {missing}")
    by_id = {r["chip_id"]: r for r in rows}
    (out / "prob_maps").mkdir(exist_ok=True)
    for name, ids in (("cloudiest", cloudy), ("mid_cloud", mid)):
        panels = []
        for cid in ids:
            s2 = load_raw(args.root, cid, "s2")
            prob, valid = probability_map(detector, s2)
            np.savez_compressed(out / "prob_maps" / f"{cid}.npz", prob=prob.astype(np.float16), valid=valid)
            panels.append({"chip_id": cid, "split": by_id[cid]["split"], "s2": s2, "prob": prob,
                           "masks": masks_at(detector, prob, valid, thresholds),
                           "fractions": {t: by_id[cid][f"t{t:g}_cloud_fraction"] for t in thresholds}})
        if panels:
            what = (f"{len(cloudy)} cloudiest train chips" if name == "cloudiest" else
                    "train chips closest to 45% cloud" + (" + named chips" if any(c in NAMED_CHIPS for c in ids) else ""))
            fig = plot_cloud_probability(panels, out / f"prob_maps_{name}.png", title=f"s2cloudless audit: {what}")
            print(f"wrote {fig}")
    print(md)
    print(f"\nwrote {csv_path} ({len(rows)} chips), {out / 'summary.md'}, prob_maps/ ({len(cloudy) + len(mid)} chips)")
    if any(math.isnan(r[f"t{thresholds[0]:g}_cloud_fraction"]) for r in rows):
        print("note: chips without valid S2 have no cloud fraction (left out of the bins)")


if __name__ == "__main__":
    main()
