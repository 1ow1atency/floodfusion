"""Interactive demo: does SAR+optical fusion survive clouds? (Gradio, CPU-only.)

Runs as a Hugging Face Space (this folder is the Space's content) or locally:

    python app/app.py          # needs app/chips/*.npz and the weights (local or HF)
    python app/app.py --dev    # synthetic chips + random weights, to try the UI only

Weights (never in git): app/weights/<file> if present, else downloaded at startup from
the Hugging Face model repo named by the FLOODFUSION_WEIGHTS_REPO environment variable
(a Space sets it under Settings -> Variables). All logic lives in floodfusion.demo;
this file is only the UI.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np

from floodfusion.chips import IGNORE_INDEX
from floodfusion.demo import (
    MANIFEST, WEIGHT_FILES, DemoModel, chip_water_iou, load_demo_model, load_packed_chip,
    make_dev_assets, opaque_cloud_s2, predict,
)
from floodfusion.viz import LABEL_COLOURS, NODATA_RGB, TRUE_COLOUR, TRUE_COLOUR_SCALE, composite

APP_DIR = Path(__file__).resolve().parent
WEIGHTS_REPO_ENV = "FLOODFUSION_WEIGHTS_REPO"
MODEL_TITLES = {"s2_only_robust": "Optical-only U-Net (robust)",
                "fusion_robust": "SAR+optical FusionNet (robust)"}
SLIDER_STEP = 5  # %; coarse steps keep the prediction cache small and effective

INTRO = """
# Flood mapping that survives clouds

Floods come with storms, and storms come with clouds — exactly when a flood map is needed,
the optical satellite image is often covered. This demo compares two flood-segmentation
models from my final-year project on a real flood scene from the **Sen1Floods11 test split**
(never used in training):

* **Optical-only U-Net**: sees only the Sentinel-2 optical image.
* **FusionNet**: fuses the same optical image with Sentinel-1 **radar** (SAR), which sees
  through clouds. Both were trained with the same cloud-robustness recipe.

Drag the slider to cover the optical image with **synthetic, fully opaque clouds**
(radar is unaffected — and the flood on the ground doesn't change, so the label stays
the truth). The optical-only model loses the water under the clouds; the fusion model
keeps finding it from radar. On the full test split (90 chips, seed 0), water IoU at
100% opaque cloud: optical-only **0.06** vs fusion **0.65** (both ~0.82 with no clouds).

*The clouds are synthetic and opaque (no ground signal underneath) — the stricter variant
of the project's robustness evaluation; real clouds are often semi-transparent. IoU below
is for the displayed chip only.*
"""


def _to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)


def sar_panel(s1: np.ndarray, stats: dict) -> np.ndarray:
    """VV in dB as grayscale with the train clip limits as display range; missing = magenta."""
    lo, hi = float(stats["s1"]["clip_min"][0]), float(stats["s1"]["clip_max"][0])
    vv = s1[0]
    g = np.clip((vv - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.repeat(np.nan_to_num(g)[..., None], 3, axis=-1)
    rgb[np.isnan(vv)] = NODATA_RGB
    return _to_uint8(rgb)


def label_panel(values: np.ndarray) -> np.ndarray:
    """Labels/predictions as RGB: water blue, not water white, ignored (-1) grey."""
    rgb = np.zeros((*values.shape, 3), np.uint8)
    for v, colour in LABEL_COLOURS.items():
        rgb[values == v] = [int(colour[i: i + 2], 16) for i in (1, 3, 5)]
    return rgb


class DemoAssets:
    """Loaded chips and models; one prediction pass per (chip, coverage), cached."""

    def __init__(self, chips_dir: Path, weight_paths: dict[str, Path],
                 band_stats_path: Path | None = None):
        manifest = json.loads((chips_dir / MANIFEST).read_text())
        self.rule = manifest["rule"]
        self.chips = {c["chip_id"]: {**c, "data": load_packed_chip(chips_dir / c["file"])}
                      for c in manifest["chips"]}
        if not self.chips:
            raise FileNotFoundError(f"no chips listed in {chips_dir / MANIFEST}")
        self.models: dict[str, DemoModel] = {name: load_demo_model(path) for name, path in weight_paths.items()}
        self.smoke = any(m.smoke for m in self.models.values())
        if band_stats_path and band_stats_path.is_file():  # sanity: checkpoints carry the train stats
            ref = json.loads(band_stats_path.read_text())
            for name, dm in self.models.items():
                for mod in dm.mods:
                    for k in ("mean", "std", "clip_min", "clip_max"):
                        if not np.allclose(dm.stats[mod][k], ref[mod][k]):
                            print(f"WARNING: {name} checkpoint {mod}.{k} differs from {band_stats_path}")
        self._cached = lru_cache(maxsize=48)(self._compute)

    def choice_label(self, cid: str) -> str:
        c = self.chips[cid]
        return f"{cid}  ({c['region']}, water {c['water_fraction']:.0%} of labelled pixels)"

    def _compute(self, cid: str, coverage_pct: int):
        chip = self.chips[cid]["data"]
        s2_cloudy = opaque_cloud_s2(chip["s2"], coverage_pct / 100.0, cid)
        optical = _to_uint8(composite(s2_cloudy, TRUE_COLOUR, TRUE_COLOUR_SCALE))
        results = {}
        for name, dm in self.models.items():
            pred = predict(dm, chip["s1"], s2_cloudy)
            shown = np.where(chip["label"] == IGNORE_INDEX, IGNORE_INDEX, pred.astype(np.int64))
            results[name] = (label_panel(shown), chip_water_iou(pred, chip["label"]))
        return optical, results

    def view(self, choice: str, coverage_pct: float):
        cid = choice.split()[0]
        optical, results = self._cached(cid, int(round(coverage_pct)))
        chip = self.chips[cid]["data"]
        stats = next(iter(self.models.values())).stats
        ious = []
        for name in WEIGHT_FILES:
            iou = results[name][1]
            ious.append(f"**{MODEL_TITLES[name]}** — water IoU on this chip: "
                        f"**{'n/a (no water in label or prediction)' if np.isnan(iou) else f'{iou:.3f}'}**")
        note = ("\n\n⚠️ **Random weights (dev/smoke mode): these predictions are meaningless.**"
                if self.smoke else "")
        return (optical, sar_panel(chip["s1"], stats), label_panel(chip["label"]),
                results["s2_only_robust"][0], results["fusion_robust"][0],
                "  ·  ".join(ious) + note)


def load_assets(dev: bool = False) -> DemoAssets:
    if dev:
        print("DEV MODE: synthetic chips and RANDOM weights (predictions are meaningless).")
        import tempfile

        chips_dir, weights = make_dev_assets(Path(tempfile.mkdtemp(prefix="floodfusion_dev_")), size=512)
        return DemoAssets(chips_dir, weights)

    chips_dir = APP_DIR / "chips"
    if not (chips_dir / MANIFEST).is_file():
        raise FileNotFoundError(f"{chips_dir / MANIFEST} not found. Pack the demo chips first "
                                "(scripts/prepare_demo_chips.py) or run with --dev.")
    weight_paths = {}
    for name, fname in WEIGHT_FILES.items():
        local = APP_DIR / "weights" / fname
        if local.is_file():
            weight_paths[name] = local
            continue
        repo = os.environ.get(WEIGHTS_REPO_ENV)
        if not repo:
            raise FileNotFoundError(
                f"{local} not found and {WEIGHTS_REPO_ENV} is not set. Either copy the seed-0 best.pt "
                f"checkpoints into app/weights/ or set {WEIGHTS_REPO_ENV}=<user>/floodfusion-weights.")
        from huggingface_hub import hf_hub_download

        print(f"downloading {fname} from {repo} ...")
        weight_paths[name] = Path(hf_hub_download(repo_id=repo, filename=fname))
    return DemoAssets(APP_DIR / "chips", weight_paths, band_stats_path=APP_DIR / "band_stats.json")


def build_demo(assets: DemoAssets) -> gr.Blocks:
    choices = [assets.choice_label(cid) for cid in assets.chips]
    with gr.Blocks(title="FloodFusion: cloud-robust flood mapping") as demo:
        gr.Markdown(INTRO)
        if assets.smoke:
            gr.Markdown("## ⚠️ DEV / SMOKE MODE — random weights, synthetic chips. "
                        "Predictions are meaningless.")
        with gr.Row():
            chip_dd = gr.Dropdown(choices=choices, value=choices[0], label="Test chip", scale=2)
            cover = gr.Slider(0, 100, value=0, step=SLIDER_STEP,
                              label="Synthetic opaque cloud cover of the optical image (%)", scale=3)
        with gr.Row():
            out_optical = gr.Image(label="Sentinel-2 optical (true colour) + synthetic clouds",
                                   interactive=False)
            out_sar = gr.Image(label="Sentinel-1 radar (VV, unaffected by clouds)", interactive=False)
            out_label = gr.Image(label="True label (blue = water, grey = no data)", interactive=False)
            out_s2only = gr.Image(label=MODEL_TITLES["s2_only_robust"], interactive=False)
            out_fusion = gr.Image(label=MODEL_TITLES["fusion_robust"], interactive=False)
        out_iou = gr.Markdown()
        gr.Markdown(f"*Chip selection rule: {assets.rule}.*")
        outputs = [out_optical, out_sar, out_label, out_s2only, out_fusion, out_iou]
        chip_dd.change(assets.view, [chip_dd, cover], outputs)
        cover.release(assets.view, [chip_dd, cover], outputs)
        demo.load(assets.view, [chip_dd, cover], outputs)
    return demo


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev", action="store_true", help="synthetic chips + random weights (UI check only)")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)
    demo = build_demo(load_assets(dev=args.dev))
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
