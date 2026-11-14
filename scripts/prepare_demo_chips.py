"""Select and pack the demo chips for the Gradio app (fixed rule, see floodfusion.demo).

    python scripts/prepare_demo_chips.py --root <dataset root with the selected chips>

Needs only the selected chips' GeoTIFFs under the root (S1Hand/, S2Hand/, LabelHand/);
run with no --root first to print which chips to copy from Google Drive.
"""

from __future__ import annotations

import argparse

from floodfusion.demo import DEMO_N_CHIPS, SELECTION_RULE, prepare_demo_chips, select_demo_chips


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="dataset root holding the selected chips; omit to only print the selection")
    p.add_argument("--out", default="app/chips", help="output folder for the packed .npz chips (default: app/chips)")
    p.add_argument("--cloud-csv", default="docs/cloud_cover.csv")
    p.add_argument("--audit-csv", default="docs/results/cloud_audit/chip_audit.csv")
    p.add_argument("--n", type=int, default=DEMO_N_CHIPS)
    args = p.parse_args(argv)

    print(f"Selection rule: {SELECTION_RULE}\n")
    if args.root:
        prepare_demo_chips(args.root, args.out, args.cloud_csv, args.audit_csv, n=args.n)
    else:
        for c in select_demo_chips(args.cloud_csv, args.audit_csv, n=args.n):
            print(f"  {c['chip_id']:<22s} water {c['water_fraction']:.1%}, real cloud {c['cloud_fraction']:.1%}")
        print("\nCopy these chips' S1Hand/S2Hand/LabelHand GeoTIFFs locally, then re-run with --root.")


if __name__ == "__main__":
    main()
