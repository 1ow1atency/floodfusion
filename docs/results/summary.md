# Robustness evaluation

**Final reporting only.** Nothing here may be used to tune hyperparameters or to select models.
Each model is its `best.pt`, selected on the real valid split. Metric: pooled water IoU (label `-1` ignored); precision and recall of the water class.

| model | run | best.pt epoch | modalities | robustness training |
|---|---|--:|---|---|
| s1_only | `s1_only_seed0` | 80 | s1 | no |
| s2_only | `s2_only_seed0` | 83 | s2 | no |
| s2_only_robust | `s2_only_robust_seed0` | 83 | s2 | yes |
| early_fusion | `early_fusion_seed0` | 74 | s1, s2 | no |
| early_fusion_robust | `early_fusion_robust_seed0` | 88 | s1, s2 | yes |
| fusion | `fusion_seed0` | 74 | s1, s2 | no |
| fusion_robust | `fusion_robust_seed0` | 95 | s1, s2 | yes |

Synthetic clouds, two variants of the same cloud layers (same masks, cloud seeds and coverage):
- **thick**: as used in robustness training (clouds_synth thick clouds, large or cumulus). They are semi-transparent: median opacity ~0.8 inside the mask, so water stays visible through them (docs/cloud_leakage.md).
- **opaque**: the same mask with opacity 1 inside it; the ground under the cloud is replaced entirely by the cloud spectrum. **The stricter test of cloud robustness.** Its coverage 0 is evaluated through the opaque path and checked equal to clean.
Each chip's cloud depends only on (chip id, coverage, cloud seed), so every model sees identical clouds; 3 cloud seeds per coverage, mean ± std over cloud seeds (sample std). Seed of every run: see the run folder.
Missing modality: the input is set to 0 (= train band mean after normalisation); FusionNet also gets the modality mask. S2 removed is the extreme beyond 100% cloud.

## test (90 chips)

### Opaque synthetic clouds (stricter test): water IoU

| model | 0% | 20% | 40% | 60% | 80% | 100% |
|---|--:|--:|--:|--:|--:|--:|
| s1_only † | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 |
| s2_only | 0.8276 ± 0.0000 | 0.5759 ± 0.0119 | 0.4001 ± 0.0031 | 0.2722 ± 0.0115 | 0.1407 ± 0.0065 | 0.0019 ± 0.0004 |
| s2_only_robust | 0.8212 ± 0.0000 | 0.6819 ± 0.0128 | 0.5256 ± 0.0162 | 0.3826 ± 0.0299 | 0.2150 ± 0.0065 | 0.0640 ± 0.0093 |
| early_fusion | 0.8202 ± 0.0000 | 0.7069 ± 0.0051 | 0.6059 ± 0.0099 | 0.5102 ± 0.0175 | 0.4454 ± 0.0078 | 0.3710 ± 0.0047 |
| early_fusion_robust | 0.8190 ± 0.0000 | 0.7771 ± 0.0029 | 0.7303 ± 0.0071 | 0.6875 ± 0.0073 | 0.6164 ± 0.0075 | 0.5302 ± 0.0075 |
| fusion | 0.8181 ± 0.0000 | 0.7379 ± 0.0063 | 0.6643 ± 0.0034 | 0.6066 ± 0.0172 | 0.5661 ± 0.0096 | 0.5461 ± 0.0130 |
| fusion_robust | 0.8130 ± 0.0000 | 0.7832 ± 0.0027 | 0.7496 ± 0.0040 | 0.7170 ± 0.0064 | 0.6870 ± 0.0035 | 0.6496 ± 0.0009 |

### Thick synthetic clouds (as trained, semi-transparent): water IoU

| model | 0% | 20% | 40% | 60% | 80% | 100% |
|---|--:|--:|--:|--:|--:|--:|
| s1_only † | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 | 0.6567 ± 0.0000 |
| s2_only | 0.8276 ± 0.0000 | 0.7262 ± 0.0071 | 0.6328 ± 0.0124 | 0.5454 ± 0.0332 | 0.4340 ± 0.0234 | 0.2105 ± 0.0533 |
| s2_only_robust | 0.8212 ± 0.0000 | 0.8113 ± 0.0012 | 0.8045 ± 0.0005 | 0.7959 ± 0.0022 | 0.7935 ± 0.0028 | 0.7906 ± 0.0038 |
| early_fusion | 0.8202 ± 0.0000 | 0.7707 ± 0.0028 | 0.7205 ± 0.0060 | 0.6645 ± 0.0090 | 0.6320 ± 0.0194 | 0.5462 ± 0.0071 |
| early_fusion_robust | 0.8190 ± 0.0000 | 0.8044 ± 0.0010 | 0.7877 ± 0.0015 | 0.7773 ± 0.0038 | 0.7622 ± 0.0019 | 0.7317 ± 0.0035 |
| fusion | 0.8181 ± 0.0000 | 0.7713 ± 0.0050 | 0.7219 ± 0.0050 | 0.6750 ± 0.0107 | 0.6377 ± 0.0072 | 0.6104 ± 0.0071 |
| fusion_robust | 0.8130 ± 0.0000 | 0.7995 ± 0.0019 | 0.7855 ± 0.0014 | 0.7710 ± 0.0043 | 0.7537 ± 0.0010 | 0.7297 ± 0.0035 |

### At 100% cloud: water IoU (precision / recall)

| model | clean | thick 100% | opaque 100% | S2 removed |
|---|--:|--:|--:|--:|
| s1_only † | 0.6567 (0.817 / 0.770) | 0.6567 ± 0.0000 (0.817 / 0.770) | 0.6567 ± 0.0000 (0.817 / 0.770) | 0.6567 (0.817 / 0.770) |
| s2_only | 0.8276 (0.927 / 0.885) | 0.2105 ± 0.0533 (0.919 / 0.215) | 0.0019 ± 0.0004 (0.100 / 0.002) | n/a |
| s2_only_robust | 0.8212 (0.919 / 0.885) | 0.7906 ± 0.0038 (0.915 / 0.853) | 0.0640 ± 0.0093 (0.150 / 0.100) | n/a |
| early_fusion | 0.8202 (0.934 / 0.871) | 0.5462 ± 0.0071 (0.918 / 0.574) | 0.3710 ± 0.0047 (0.877 / 0.392) | 0.0000 (nan / 0.000) |
| early_fusion_robust | 0.8190 (0.913 / 0.888) | 0.7317 ± 0.0035 (0.919 / 0.782) | 0.5302 ± 0.0075 (0.868 / 0.576) | 0.6348 (0.858 / 0.710) |
| fusion | 0.8181 (0.923 / 0.878) | 0.6104 ± 0.0071 (0.719 / 0.802) | 0.5461 ± 0.0130 (0.667 / 0.751) | 0.0002 (0.969 / 0.000) |
| fusion_robust | 0.8130 (0.916 / 0.879) | 0.7297 ± 0.0035 (0.883 / 0.807) | 0.6496 ± 0.0009 (0.829 / 0.751) | 0.6385 (0.883 / 0.698) |

### Missing modality: water IoU (precision / recall)

| model | clean | S2 removed | S1 removed |
|---|--:|--:|--:|
| s1_only | 0.6567 (0.817 / 0.770) | 0.6567 (0.817 / 0.770) † | n/a |
| s2_only | 0.8276 (0.927 / 0.885) | n/a | 0.8276 (0.927 / 0.885) † |
| s2_only_robust | 0.8212 (0.919 / 0.885) | n/a | 0.8212 (0.919 / 0.885) † |
| early_fusion | 0.8202 (0.934 / 0.871) | 0.0000 (nan / 0.000) | 0.6551 (0.960 / 0.674) |
| early_fusion_robust | 0.8190 (0.913 / 0.888) | 0.6348 (0.858 / 0.710) | 0.8127 (0.906 / 0.887) |
| fusion | 0.8181 (0.923 / 0.878) | 0.0002 (0.969 / 0.000) | 0.1558 (0.974 / 0.156) |
| fusion_robust | 0.8130 (0.916 / 0.879) | 0.6385 (0.883 / 0.698) | 0.8044 (0.906 / 0.878) |

### Real clouds (s2cloudless bins): water IoU (chips)

| model | 0-10% | 10-30% | 30-60% | 60-100% |
|---|--:|--:|--:|--:|
| s1_only | 0.7133 (54) | 0.7742 (11) | 0.4351 (13) | 0.2584 (11) |
| s2_only | 0.8776 (54) | 0.9024 (11) | 0.6787 (13) | 0.3468 (11) |
| s2_only_robust | 0.8736 (54) | 0.8987 (11) | 0.6617 (13) | 0.2544 (11) |
| early_fusion | 0.8814 (54) | 0.9074 (11) | 0.6043 (13) | 0.4208 (11) |
| early_fusion_robust | 0.8645 (54) | 0.8915 (11) | 0.6868 (13) | 0.3791 (11) |
| fusion | 0.8777 (54) | 0.9053 (11) | 0.6096 (13) | 0.4469 (11) |
| fusion_robust | 0.8671 (54) | 0.8965 (11) | 0.6323 (13) | 0.4091 (11) |

⚠ = unreliable: fewer than 10 chips. Chips without a cloud fraction (no valid S2): 1.
_Real-cloud bins use s2cloudless masks; the mask audit (docs/ROADMAP.md, Stage 5 TODO) is still pending, so treat these bins as indicative._

## bolivia (15 chips)

### Opaque synthetic clouds (stricter test): water IoU

| model | 0% | 20% | 40% | 60% | 80% | 100% |
|---|--:|--:|--:|--:|--:|--:|
| s1_only † | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 |
| s2_only | 0.7908 ± 0.0000 | 0.5300 ± 0.0260 | 0.3811 ± 0.0324 | 0.2528 ± 0.0243 | 0.1118 ± 0.0097 | 0.0027 ± 0.0006 |
| s2_only_robust | 0.7886 ± 0.0000 | 0.6708 ± 0.0043 | 0.5592 ± 0.0121 | 0.4034 ± 0.0348 | 0.2796 ± 0.0661 | 0.0505 ± 0.0155 |
| early_fusion | 0.7922 ± 0.0000 | 0.6274 ± 0.0495 | 0.5035 ± 0.0261 | 0.3958 ± 0.0191 | 0.2943 ± 0.0337 | 0.2530 ± 0.0300 |
| early_fusion_robust | 0.7818 ± 0.0000 | 0.7393 ± 0.0078 | 0.6861 ± 0.0095 | 0.6124 ± 0.0289 | 0.5810 ± 0.0463 | 0.4106 ± 0.0053 |
| fusion | 0.7971 ± 0.0000 | 0.7137 ± 0.0336 | 0.6683 ± 0.0121 | 0.6642 ± 0.0101 | 0.6252 ± 0.0385 | 0.6867 ± 0.0072 |
| fusion_robust | 0.7916 ± 0.0000 | 0.7667 ± 0.0074 | 0.7407 ± 0.0047 | 0.6955 ± 0.0121 | 0.6768 ± 0.0154 | 0.6176 ± 0.0008 |

### Thick synthetic clouds (as trained, semi-transparent): water IoU

| model | 0% | 20% | 40% | 60% | 80% | 100% |
|---|--:|--:|--:|--:|--:|--:|
| s1_only † | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 | 0.6253 ± 0.0000 |
| s2_only | 0.7908 ± 0.0000 | 0.6355 ± 0.0373 | 0.5161 ± 0.0384 | 0.3961 ± 0.0179 | 0.2501 ± 0.0307 | 0.1013 ± 0.0810 |
| s2_only_robust | 0.7886 ± 0.0000 | 0.7879 ± 0.0054 | 0.7813 ± 0.0040 | 0.7849 ± 0.0048 | 0.7789 ± 0.0022 | 0.7826 ± 0.0063 |
| early_fusion | 0.7922 ± 0.0000 | 0.7243 ± 0.0267 | 0.6683 ± 0.0213 | 0.6405 ± 0.0049 | 0.5903 ± 0.0135 | 0.6116 ± 0.0309 |
| early_fusion_robust | 0.7818 ± 0.0000 | 0.7725 ± 0.0063 | 0.7561 ± 0.0111 | 0.7459 ± 0.0041 | 0.7373 ± 0.0070 | 0.7325 ± 0.0094 |
| fusion | 0.7971 ± 0.0000 | 0.7391 ± 0.0286 | 0.7043 ± 0.0053 | 0.6935 ± 0.0087 | 0.6490 ± 0.0294 | 0.6898 ± 0.0058 |
| fusion_robust | 0.7916 ± 0.0000 | 0.7858 ± 0.0038 | 0.7767 ± 0.0003 | 0.7602 ± 0.0068 | 0.7494 ± 0.0086 | 0.7278 ± 0.0021 |

### At 100% cloud: water IoU (precision / recall)

| model | clean | thick 100% | opaque 100% | S2 removed |
|---|--:|--:|--:|--:|
| s1_only † | 0.6253 (0.834 / 0.714) | 0.6253 ± 0.0000 (0.834 / 0.714) | 0.6253 ± 0.0000 (0.834 / 0.714) | 0.6253 (0.834 / 0.714) |
| s2_only | 0.7908 (0.896 / 0.871) | 0.1013 ± 0.0810 (0.769 / 0.105) | 0.0027 ± 0.0006 (0.167 / 0.003) | n/a |
| s2_only_robust | 0.7886 (0.855 / 0.911) | 0.7826 ± 0.0063 (0.889 / 0.868) | 0.0505 ± 0.0155 (0.129 / 0.078) | n/a |
| early_fusion | 0.7922 (0.929 / 0.843) | 0.6116 ± 0.0309 (0.930 / 0.642) | 0.2530 ± 0.0300 (0.980 / 0.254) | 0.0000 (nan / 0.000) |
| early_fusion_robust | 0.7818 (0.850 / 0.907) | 0.7325 ± 0.0094 (0.908 / 0.791) | 0.4106 ± 0.0053 (0.939 / 0.422) | 0.5258 (0.924 / 0.550) |
| fusion | 0.7971 (0.904 / 0.871) | 0.6898 ± 0.0058 (0.765 / 0.876) | 0.6867 ± 0.0072 (0.805 / 0.824) | 0.0000 (nan / 0.000) |
| fusion_robust | 0.7916 (0.855 / 0.914) | 0.7278 ± 0.0021 (0.888 / 0.801) | 0.6176 ± 0.0008 (0.878 / 0.676) | 0.4014 (0.895 / 0.421) |

### Missing modality: water IoU (precision / recall)

| model | clean | S2 removed | S1 removed |
|---|--:|--:|--:|
| s1_only | 0.6253 (0.834 / 0.714) | 0.6253 (0.834 / 0.714) † | n/a |
| s2_only | 0.7908 (0.896 / 0.871) | n/a | 0.7908 (0.896 / 0.871) † |
| s2_only_robust | 0.7886 (0.855 / 0.911) | n/a | 0.7886 (0.855 / 0.911) † |
| early_fusion | 0.7922 (0.929 / 0.843) | 0.0000 (nan / 0.000) | 0.0398 (0.937 / 0.040) |
| early_fusion_robust | 0.7818 (0.850 / 0.907) | 0.5258 (0.924 / 0.550) | 0.7641 (0.830 / 0.906) |
| fusion | 0.7971 (0.904 / 0.871) | 0.0000 (nan / 0.000) | 0.0090 (0.999 / 0.009) |
| fusion_robust | 0.7916 (0.855 / 0.914) | 0.4014 (0.895 / 0.421) | 0.7637 (0.816 / 0.922) |

### Real clouds (s2cloudless bins): water IoU (chips)

| model | 0-10% | 10-30% | 30-60% | 60-100% |
|---|--:|--:|--:|--:|
| s1_only | 0.7080 (6) ⚠ | 0.5639 (4) ⚠ | 0.5355 (5) ⚠ | – (0) |
| s2_only | 0.8752 (6) ⚠ | 0.7096 (4) ⚠ | 0.7368 (5) ⚠ | – (0) |
| s2_only_robust | 0.8685 (6) ⚠ | 0.7047 (4) ⚠ | 0.7521 (5) ⚠ | – (0) |
| early_fusion | 0.8702 (6) ⚠ | 0.7096 (4) ⚠ | 0.7480 (5) ⚠ | – (0) |
| early_fusion_robust | 0.8549 (6) ⚠ | 0.7007 (4) ⚠ | 0.7568 (5) ⚠ | – (0) |
| fusion | 0.8770 (6) ⚠ | 0.7094 (4) ⚠ | 0.7614 (5) ⚠ | – (0) |
| fusion_robust | 0.8584 (6) ⚠ | 0.7236 (4) ⚠ | 0.7550 (5) ⚠ | – (0) |

⚠ = unreliable: fewer than 10 chips. Chips without a cloud fraction (no valid S2): 0.
_Real-cloud bins use s2cloudless masks; the mask audit (docs/ROADMAP.md, Stage 5 TODO) is still pending, so treat these bins as indicative._

† = unaffected by this setting (model lacks the changed modality): its clean result is shown.
n/a = the setting removes the model's only input (for s2_only, S2 removed = beyond 100% cloud).

Clouds generated in this run (main process): 0.

