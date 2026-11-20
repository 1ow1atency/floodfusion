"""Precompute every panel of the demo for the static page in static_demo/.

    python scripts/export_static_demo.py

Uses the same chips (app/chips), checkpoints (app/weights) and opaque cloud settings as
app/app.py, so each exported frame is pixel-identical to the Gradio app's. Writes
static_demo/img/**.{webp,png} and static_demo/data.json; index.html reads only data.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from floodfusion.demo import STATIC_COVERAGES, WEBP_QUALITY, WEIGHT_FILES, export_static_demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chips", default="app/chips", help="packed demo chips (default: app/chips)")
    p.add_argument("--weights", default="app/weights", help="folder with the seed-0 best.pt files")
    p.add_argument("--out", default="static_demo", help="output folder (default: static_demo)")
    p.add_argument("--step", type=int, default=10, help="cloud-cover step in %% (default: 10)")
    p.add_argument("--quality", type=int, default=WEBP_QUALITY, help="WebP quality of the photo panels")
    args = p.parse_args(argv)

    weights_dir = Path(args.weights)
    weight_paths = {name: weights_dir / fname for name, fname in WEIGHT_FILES.items()}
    missing = [str(p_) for p_ in weight_paths.values() if not p_.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint(s) not found: {missing}")
    coverages = list(range(0, 101, args.step)) if args.step != 10 else list(STATIC_COVERAGES)
    export_static_demo(args.chips, weight_paths, args.out, coverages=coverages, quality=args.quality)


if __name__ == "__main__":
    main()
