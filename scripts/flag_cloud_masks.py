"""Flag unreliable s2cloudless masks in cloud_cover.csv and write the 0.6-threshold variant.

From a full-data chip audit (scripts/audit_clouds.py -> chip_audit.csv) this
  1. adds the mask_unreliable column to --cloud-csv (floodfusion.cloud_audit.mask_unreliable:
     probability saturated over visible ground, see docs/cloud_audit.md), keeping the existing
     cloud fractions byte-identical;
  2. writes --t06-out with the cloud fractions at threshold 0.6 (sensitivity check) and the
     same flags.

Pure CSV-to-CSV: no s2cloudless, no torch, no data access.

    python scripts/flag_cloud_masks.py --audit docs/results/cloud_audit/chip_audit.csv
"""

from __future__ import annotations

import argparse
import math

from floodfusion.cloud_audit import UNRELIABLE_HIGH_T, cloud_cover_rows, read_rows, unreliable_chips
from floodfusion.clouds import CLOUD_THRESHOLD, read_cloud_csv, write_cloud_csv


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit", required=True, help="chip_audit.csv of the FULL-data audit")
    p.add_argument("--cloud-csv", default="docs/cloud_cover.csv", help="updated in place (flag column added)")
    p.add_argument("--t06-out", default="docs/cloud_cover_t06.csv")
    args = p.parse_args(argv)

    audit = read_rows(args.audit)
    existing = read_cloud_csv(args.cloud_csv)
    if {r["chip_id"] for r in audit} != {r["chip_id"] for r in existing}:
        raise SystemExit(f"{args.audit} and {args.cloud_csv} cover different chips: full-data audit needed")
    base = {r["chip_id"]: r for r in cloud_cover_rows(audit, CLOUD_THRESHOLD)}
    diffs = [abs(base[r["chip_id"]]["cloud_fraction"] - r["cloud_fraction"]) for r in existing
             if not math.isnan(r["cloud_fraction"])]
    print(f"audit reproduces {args.cloud_csv} at {CLOUD_THRESHOLD:g}: max abs difference {max(diffs):.2e}")
    if max(diffs) > 1e-4:
        raise SystemExit("difference too large: the audit does not match the cloud_cover.csv in use")

    for r in existing:  # keep the existing fractions; only add the flag
        r["mask_unreliable"] = base[r["chip_id"]]["mask_unreliable"]
    write_cloud_csv(existing, args.cloud_csv)
    t06 = cloud_cover_rows(audit, UNRELIABLE_HIGH_T)
    order = {r["chip_id"]: i for i, r in enumerate(existing)}
    t06.sort(key=lambda r: order[r["chip_id"]])
    write_cloud_csv(t06, args.t06_out)

    flagged = unreliable_chips(audit)
    print(f"flagged mask_unreliable: {len(flagged)} chip(s)")
    for r in flagged:
        print(f"  {r['split']:7s} {r['chip_id']:22s} cloud {r[f't{CLOUD_THRESHOLD:g}_cloud_fraction']:.4f} at "
              f"{CLOUD_THRESHOLD:g} / {r[f't{UNRELIABLE_HIGH_T:g}_cloud_fraction']:.4f} at {UNRELIABLE_HIGH_T:g}, "
              f"dim {r[f't{CLOUD_THRESHOLD:g}_mask_dim']:.0%}")
    print(f"wrote {args.cloud_csv} (flag column) and {args.t06_out} (threshold {UNRELIABLE_HIGH_T:g} fractions)")


if __name__ == "__main__":
    main()
