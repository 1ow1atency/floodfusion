---
title: FloodFusion — cloud-robust flood mapping
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "6.32.0"
app_file: app.py
pinned: false
---

# FloodFusion: does SAR+optical fusion survive clouds?

Interactive demo of a final-year project on cloud-robust flood segmentation
(Sen1Floods11, hand-labeled subset). Pick a test chip, slide the synthetic opaque
cloud cover from 0 to 100%, and compare an optical-only U-Net against a two-branch
SAR+optical FusionNet — both trained with the same robustness recipe.

- The clouds are **synthetic and fully opaque**: no ground signal survives under them.
  Radar (Sentinel-1) is unaffected, so the fusion model keeps finding the water.
- Chips are from the Sen1Floods11 **test split** (never used in training or model
  selection), chosen by a fixed rule (clear sky, visible water, a mix of regions).
- Weights are downloaded at startup from the Hugging Face model repo named by the
  `FLOODFUSION_WEIGHTS_REPO` Space variable; inference runs on CPU.

Code: https://github.com/1ow1atency/floodfusion
