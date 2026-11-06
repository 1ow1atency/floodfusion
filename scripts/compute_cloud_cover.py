"""Cloud fraction of the S2 image of every chip (all splits) with s2cloudless.

Writes a CSV (chip_id, split, cloud_fraction, valid_s2_fraction), prints chip counts
per split in cloud bins, and saves a figure of the 6 cloudiest and 3 clearest train
chips with the cloud mask outlined. Deterministic (no randomness, so no seed).

    python scripts/compute_cloud_cover.py --root data/sen1floods11 --out outputs/cloud_cover.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from floodfusion.chips import load_raw
from floodfusion.clouds import (
    AVERAGE_OVER, CLOUD_THRESHOLD, DILATION_SIZE, cloud_mask, compute_cloud_cover, format_bin_table,
    make_detector, pick_examples, write_cloud_csv,
)
from floodfusion.viz import plot_cloud_examples


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Sen1Floods11 root")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--fig", default=None, help="example figure path (default: cloud_examples.png next to --out)")
    p.add_argument("--allow-missing", action="store_true",
                   help="drop chips with missing files (local sample only; never on full data)")
    args = p.parse_args(argv)

    detector = make_detector()  # fail fast if s2cloudless is not installed
    print(f"s2cloudless: threshold={CLOUD_THRESHOLD}, average_over={AVERAGE_OVER}, "
          f"dilation_size={DILATION_SIZE}, input = L1C DN / 10000, all 13 bands")
    rows = compute_cloud_cover(args.root, detector=detector, allow_missing=args.allow_missing)
    out = write_cloud_csv(rows, args.out)

    print("\nChips per split and cloud bin (cloud fraction of valid S2 pixels):")
    print(format_bin_table(rows))

    examples = pick_examples(rows, "train", n_cloudy=6, n_clear=3)
    fig_path = Path(args.fig) if args.fig else out.with_name("cloud_examples.png")
    if examples:
        pairs = [(r, cloud_mask(detector, load_raw(args.root, r["chip_id"], "s2"))[0]) for r in examples]
        plot_cloud_examples(args.root, pairs, fig_path,
                            title="train: 6 cloudiest (top two rows) and 3 clearest (bottom row)")
        print(f"\nwrote {fig_path}")
    else:
        print("\nno train chips with valid S2: example figure skipped")
    print(f"wrote {out} ({len(rows)} chips)")


if __name__ == "__main__":
    main()
