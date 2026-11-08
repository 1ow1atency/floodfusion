# s2cloudless mask audit (real-cloud bins)

Status (2026-11-08): **local sample only**. The full-data run (`notebooks/05_cloud_audit.ipynb`,
CPU, ~35–50 min) is still to do. `docs/cloud_cover.csv` is unchanged (threshold 0.4).

The local sample has 25 chips: 20 Ghana train chips and 5 Bolivia chips. It has no test chips
and no Paraguay chips, so Paraguay_215904 and Paraguay_36146 (from the ROADMAP TODO) could not
be inspected here. They are in the figure list of the full-data run.

At threshold 0.4, the audit reproduces `docs/cloud_cover.csv` for all 25 chips (max difference
5e-7). The masks below are therefore the ones behind the current bins.

Reproduce:

```
python scripts/audit_clouds.py --root data/sen1floods11 --out outputs/cloud_audit --allow-missing
```

It writes `chip_audit.csv`, `summary.md` (all tables), probability-map figures and `prob_maps/*.npz`.

## 1. Labels under cloud: annotators skipped what they could not see

Pooled over valid S2 pixels, at threshold 0.4 (shares are label -1 / land / water):

| group | chips | cloudy share of valid S2 | cloudy pixels: -1 / land / water | clear pixels: -1 / land / water | scored pixels under cloud |
|---|--:|--:|--:|--:|--:|
| train | 20 | 39.7% | 57.7% / 41.8% / 0.5% | 1.8% / 95.6% / 2.6% | 22.1% |
| bolivia | 5 | 13.5% | 50.3% / 49.5% / 0.2% | 3.0% / 74.8% / 22.2% | 7.4% |
| train, 60–100% bin | 8 | 85.1% | 70.2% / 29.8% / 0.0% | 14.2% / 85.8% / 0.0% | 66.5% |
| train, 30–60% bin | 2 | 55.4% | 1.0% / 95.9% / 3.2% | 0.0% / 73.0% / 26.9% | 55.2% |

"Scored pixels under cloud" is the share of the pixels IoU is computed on (labels 0 and 1)
that are cloudy.

- **Most cloudy pixels are labelled -1:** 50–58% of cloudy pixels are -1, against 2–3% of clear
  pixels. At thresholds 0.5 and 0.6 this rises to 66% and 75% (train).
- **Almost no water is labelled under cloud:** 0.5% of cloudy pixels in train vs 2.6% of clear
  pixels; 0.2% vs 22% in Bolivia.
- **What the real-cloud bins actually score:**
  - water in the clear parts of cloudy chips;
  - false water alarms on land that is labelled under thin cloud or haze.

  They do **not** test whether a model finds water under cloud, because that water is
  unlabelled (-1) and ignored. Only the synthetic opaque-cloud setting tests that.
- **Report the bins with this caveat,** next to the opaque results. A better threshold does not
  change this: it is a property of the hand labels.

Local caveat: the 8 train chips in the 60–100% bin contain no labelled water at all, so their
bin IoU rests on false positives only. Check the per-bin water counts on the full data.

## 2. Mask audit: 0.4 over-flags haze, not clear gaps

Figures: `outputs/cloud_audit/prob_maps_{cloudiest,mid_cloud}.png`. The mid-cloud one is saved
as [figures/cloud_audit_mid_cloud_local.png](figures/cloud_audit_mid_cloud_local.png). Each
chip shows true colour, the raw s2cloudless probability, and the mask outlined at 0.4 (red,
solid), 0.5 (yellow, dashed) and 0.6 (cyan, dotted).

- **Cloudiest chips (Ghana_161233, _154838, _144050, _168875, _264787, _134751): correct.**
  These are real cloud decks and bright cumulus, so the masks are right at every threshold. The
  clear gaps between cumulus (Ghana_264787, also Ghana_146222 and Ghana_97516) are left out
  cleanly.
  - Smoothing and dilation add only a thin rim. About 16% of mask pixels (median over chips)
    have their own probability below the threshold.
  - So the mask does **not** over-flag clear gaps in any way that matters.
- **Hazy chips: 0.4 flags haze through which the ground is plainly visible.**
  - Ghana_103272: uniform haze with fields visible. Cloud fraction is 56% at 0.4, 25% at 0.5
    and 7% at 0.6.
  - Ghana_11745: light haze. 26% / 15% / 6%; 97% of its mask pixels are dim (mean B2–B4 TOA
    < 0.15).
  - Ghana_141910: 55% / 37% / 21%.
  - Ghana_128663 (altocumulus or haze, ground faint): 71% / 49% / 32%. This one is genuinely
    borderline.
  - The labels agree that these are over-flagged. On the two hazy chips in the 30–60% bin, 96%
    of the pixels flagged cloudy at 0.4 were labelled as land by the annotators, who could see
    the ground.

Mask agreement with the annotators' no-data (label -1), **train chips only**, so the choice
never looks at test or Bolivia labels:

| threshold | mean cloud fraction | dim mask pixels (median) | precision vs -1 | recall vs -1 | F1 |
|--:|--:|--:|--:|--:|--:|
| 0.4 (current) | 37.1% | 29.9% | 57.7% | 95.5% | 72.0% |
| 0.5 | 31.4% | 18.9% | 66.3% | 91.7% | 76.9% |
| 0.6 | 26.0% | 11.4% | 75.1% | 85.5% | **80.0%** |
| 0.7 (extra check) | 21.2% | 2.3% | 82.6% | 77.4% | 79.9% |
| 0.8 (extra check) | 16.7% | 0.3% | 91.1% | 67.9% | 77.8% |

Label -1 is only a proxy for "the annotator could not see the ground"; it is not ground truth.

**Bin counts.** Locally only Bolivia can be checked, and its bins do not change between 0.4,
0.5 and 0.6 (4 / 0 / 1 / 0 chips in 0–10 / 10–30 / 30–60 / 60–100%). Across all 25 local
chips, 2 change bin at 0.5 and 5 at 0.6, all moving to a lower bin:
- the four hazy or borderline chips above (Ghana_103272, _141910, _11745, _128663);
- Ghana_146222, which goes from 60–100% to 30–60% at 0.6 because the haze between its cumulus
  is no longer counted.

Ghana_161233 stays in 60–100% but drops from 98% to 76%, losing a bright, thin layer on its
left side. The test-split bin counts need the full-data run.

## Recommendation: threshold 0.6 (provisional until the full-data run)

- **What 0.6 removes:** the haze-only flagging (Ghana_103272 and Ghana_11745 drop below 10%),
  while thick decks and bright cumulus stay masked (Ghana_154838 88%, Ghana_144050 77%).
- **Agreement with the annotators:** it gives the best agreement with where they could not
  see. F1 is 80% vs 72% at 0.4, and the share of masked pixels the annotators labelled anyway
  falls from 42% to 25%.
- **Why not higher:** 0.7 is equal on F1 but has lower recall, and beyond that the mask starts
  dropping genuine cloud. Within the tie, 0.6 is the conservative choice: more pixels counted
  as cloud.
- **For the report:** this departs from s2cloudless's default of 0.4 for 10 m data, so it
  needs this justification. The bins measure "cloud the optical model cannot see through", and
  0.4 also counts haze it can see through.

Confirm on the full data before changing `cloud_cover.csv`:
1. F1 against -1 on all train chips still peaks around 0.6.
2. Paraguay_215904 and Paraguay_36146 leave the 60–100% bin, or their probability maps show
   why not.
3. The test and Bolivia bins at 0.6 still have enough chips (fewer than 10 is flagged
   unreliable in the robustness evaluation).
