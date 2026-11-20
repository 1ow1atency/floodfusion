---
title: FloodFusion — cloud-robust flood mapping
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
---

# FloodFusion: does SAR+optical fusion survive clouds?

Static version of the demo for a final-year project on cloud-robust flood segmentation
(Sen1Floods11, hand-labelled subset). Pick a test chip, slide the synthetic opaque cloud
cover from 0 to 100%, and compare an optical-only U-Net against a two-branch SAR+optical
FusionNet — both trained with the same robustness recipe.

Everything on the page was **precomputed by the real trained models** (the seed-0
`best.pt` checkpoints) and exported as images by `scripts/export_static_demo.py`: no model
runs in the browser, and the IoUs are identical to the interactive Gradio version. The
clouds are **synthetic and fully opaque** — no ground signal survives under them — which is
the stricter variant of the project's robustness evaluation.

Chips come from the Sen1Floods11 **test split** (never used in training or model selection),
chosen by a fixed rule: mostly clear sky, visible water, a mix of regions.

Code and full results: https://github.com/1ow1atency/floodfusion
