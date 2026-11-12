"""Diagnose chips where the s2cloudless probability saturates near 1 over the whole chip.

For each chip: median (p5-p95) raw DN of B1, B2, B8, B10, B11; the s2cloudless probability; and
the probability again with B10 (cirrus, 1375 nm) replaced by a normal clear-sky value. On the
affected Sen1Floods11 chips B10 is elevated chip-wide (median DN 300-1300 against 10-90 on
normal chips) - a high thin-cirrus veil, or a B10 artefact - and it alone pins the probability
at ~1 even where the other bands show plain ground (docs/cloud_audit.md). Needs the `clouds`
extra; never imports torch. Colab (CPU, after the notebook-05 clone + install):

    python scripts/diagnose_cloud_saturation.py --root /content/drive/MyDrive/floodfusion/data/sen1floods11

The default chips are the saturated ones plus genuinely overcast and normal chips to compare.
"""

from __future__ import annotations

import argparse

import numpy as np

from floodfusion.chips import S2_BANDS, chip_path, load_raw
from floodfusion.clouds import CLOUD_THRESHOLD, make_detector, s2_reflectance

# flagged mask_unreliable | genuinely overcast (also prob ~1, but opaque in every band) | normal
DEFAULT_CHIPS = ("Paraguay_215904", "Paraguay_36146", "Paraguay_148318", "Paraguay_225187",
                 "Paraguay_913449", "Paraguay_205585", "Paraguay_252217", "Paraguay_581976",
                 "Sri-Lanka_956740", "Sri-Lanka_1049830",
                 "Paraguay_126224", "Paraguay_12870", "Paraguay_36015", "Paraguay_198534", "Paraguay_40936",
                 "Paraguay_62897", "Paraguay_339807", "Ghana_103272", "Bolivia_233925")
SHOW = ("B1", "B2", "B8", "B10", "B11")
B10 = S2_BANDS.index("B10")
B10_CLEAR = 0.003  # typical clear-sky B10 TOA reflectance (30 DN)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Sen1Floods11 root")
    p.add_argument("--chips", default=",".join(DEFAULT_CHIPS), help="comma-separated chip ids")
    args = p.parse_args(argv)

    detector = make_detector()
    t = CLOUD_THRESHOLD
    print(f"{'chip':20s} " + " ".join(f"{b + ' DN':>16s}" for b in SHOW)
          + f" {'prob p50':>9s} {'>' + format(t, 'g'):>6s} {'p50 noB10':>10s} {'>' + format(t, 'g'):>6s}")
    for cid in args.chips.split(","):
        cid = cid.strip()
        if not chip_path(args.root, cid, "s2").is_file():
            print(f"{cid:20s} not in this data, skipped")
            continue
        s2 = load_raw(args.root, cid, "s2")
        refl, valid = s2_reflectance(s2)
        if not valid.any():
            print(f"{cid:20s} no valid S2")
            continue
        prob = detector.get_cloud_probability_maps(refl[None])[0]
        no_b10 = refl.copy()
        no_b10[..., B10] = B10_CLEAR
        prob2 = detector.get_cloud_probability_maps(no_b10[None])[0]
        cols = []
        for b in SHOW:
            v = s2[S2_BANDS.index(b)][valid]
            cols.append(f"{np.median(v):5.0f} ({np.percentile(v, 5):.0f}-{np.percentile(v, 95):.0f})".rjust(16))
        print(f"{cid:20s} " + " ".join(cols)
              + f" {np.median(prob[valid]):9.2f} {(prob[valid] > t).mean():6.2f}"
              + f" {np.median(prob2[valid]):10.2f} {(prob2[valid] > t).mean():6.2f}")
    print("\nReading: a saturated mask whose '>0.4' share collapses in the noB10 column is driven by the"
          "\nelevated B10 alone; normal chips have B10 medians of about 10-90 DN and are unaffected."
          "\n(The noB10 probability is diagnostic only: s2cloudless is always run with the real B10.)")


if __name__ == "__main__":
    main()
