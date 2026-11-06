"""Train a U-Net baseline from a YAML config.

    python scripts/train.py --config configs/s1_only.yaml                 # Colab GPU
    python scripts/train.py --config configs/s1_only.yaml --resume        # continue after a disconnect
    python scripts/train.py --config configs/s1_only.yaml --smoke \\
        --root data/sen1floods11 --out outputs/runs                       # local CPU/MPS check
"""

from __future__ import annotations

import argparse

from floodfusion.training import load_config, resolve_config, run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config in configs/")
    p.add_argument("--root", help="override data.root")
    p.add_argument("--out", help="override output_dir")
    p.add_argument("--seed", type=int, help="override the config seed (saved in the run's config.yaml)")
    p.add_argument("--save-last-every", type=int, metavar="N",
                   help="write last.pt every N epochs and on the final epoch (default: config, else 5)")
    p.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases logging")
    p.add_argument("--resume", action="store_true", help="continue the run from its last.pt")
    p.add_argument("--smoke", action="store_true",
                   help="2 train + 2 val batches on CPU/MPS with the local sample (allow_missing); NOT a result")
    args = p.parse_args(argv)

    cfg = resolve_config(load_config(args.config), root=args.root, out=args.out, no_wandb=args.no_wandb,
                         seed=args.seed, save_last_every=args.save_last_every)
    run(cfg, resume=args.resume, smoke=args.smoke)


if __name__ == "__main__":
    main()
