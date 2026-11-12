# s2cloudless mask audit (real-cloud bins)

Status (2026-11-13): **full data, final** (446 chips, notebook 05 on Colab CPU). Sources:
[results/cloud_audit/](results/cloud_audit/) (`summary.md`, `chip_audit.csv`, both
probability-map figures). This replaces the provisional local-sample conclusions (git history
has them). Decisions taken:

1. **Keep threshold 0.4 for the reported bins** (`docs/cloud_cover.csv`, fractions unchanged);
   0.6 is a sensitivity check (`docs/cloud_cover_t06.csv`).
2. **10 chips get `mask_unreliable = 1`** in `docs/cloud_cover.csv` (s2cloudless saturated over
   visible ground, below); `robustness_eval.py` reports the real-cloud bins with and without them.
3. **The real-cloud bins are supporting evidence only.** Water under real cloud is mostly
   unlabelled, so they cannot test what the project asks. The opaque synthetic sweep is the
   main experiment.

At 0.4 the audit reproduces `docs/cloud_cover.csv` for all 445 chips with valid S2 (max
difference 5e-7), so the audited masks are the ones behind the current bins.

## 1. Labels under real cloud: the bins cannot score water under cloud

At threshold 0.4, pooled over valid S2 pixels (shares are label -1 / land / water):

| group | cloudy pixels | clear pixels |
|---|--:|--:|
| test (90 chips) | **47.9%** / 48.8% / **3.3%** | 1.8% / 85.3% / **12.9%** |
| bolivia (15 chips) | **66.9%** / 32.7% / **0.4%** | 6.0% / 77.7% / 16.3% |
| train (252 chips) | 43.2% / 54.3% / 2.5% | 1.5% / 88.2% / 10.3% |

- **Annotators skipped what they could not see:** about half of the cloudy pixels are -1
  (two thirds on Bolivia), against ~2% of clear pixels.
- **Almost no water is labelled under cloud:** 3.3% of cloudy test pixels vs 12.9% of clear
  ones; 0.4% vs 16.3% on Bolivia.
- **The 60–100% test bin is the worst case:** 83.4% of its scored (0/1) pixels are under
  cloud, but only 0.7% of its cloudy pixels are labelled water. Its IoU therefore rests on
  false alarms and the few visible patches, not on finding water under cloud.
- No threshold changes this; it is a property of the hand labels. **Report the bins next to
  the opaque results as supporting evidence only.** Only the opaque synthetic setting tests
  water under cloud against a trustworthy label.

## 2. Threshold: keep 0.4, report 0.6 as sensitivity

Agreement of the mask with the annotators' -1 (train chips only, so the choice never looks at
test or Bolivia labels): F1 **58.4% / 60.7% / 62.3%** at 0.4 / 0.5 / 0.6. The full-data gain
from 0.6 is real but small (the 25-chip local sample had overstated it), and 0.4 is the
s2cloudless default for 10 m data.

- `docs/cloud_cover.csv` keeps the 0.4 fractions, byte-identical; only the
  `mask_unreliable` column was added.
- `docs/cloud_cover_t06.csv` holds the 0.6 fractions (`scripts/flag_cloud_masks.py` builds
  both from `chip_audit.csv`). Pass it to `robustness_eval.py --cloud-csv` to redo the bins
  at 0.6.
- What 0.6 changes: fewer false alarms on haze and bright bare ground
  (Pakistan_401863: 46% cloud at 0.4 → 10% at 0.6), but the test 60–100% bin shrinks from
  11 to 7 chips. Bolivia moves one chip (6/4/5/0 → 7/4/4/0); its 60–100% bin stays empty.

## 3. Saturated masks over visible ground (the "Paraguay problem"): flagged, not fixed

**Symptom.** On a group of chips, nearly all from the Paraguay flood event, the s2cloudless
probability is ~1.0 over the whole chip at every
threshold (`prob_maps_cloudiest.png`: a solid dark-blue map). On several of them the ground is
plainly visible — crisp fields, roads, rivers (Paraguay_215904, _36146, _148318, _225187,
_913449, valid chips _205585, _252217, _581976, and Sri-Lanka_956740).

**Cause: the cirrus band B10.** Verified locally on 22 chips fetched from the public
Sen1Floods11 bucket (`scripts/diagnose_cloud_saturation.py` reproduces this on Colab):

- The saturated chips have B10 elevated chip-wide: median 260–1340 DN, against 10–90 DN on
  normal chips. B1, B2, B8 and B11 are in normal ranges, and the audit reproduces
  `cloud_cover.csv` exactly, so **the inputs are scaled correctly (DN/10000)** — the anomaly
  is in the source data's B10 itself.
- Replacing B10 with a normal clear-sky value drops the probability below 0.4 for most of the
  visible-ground chips (Paraguay_913449: share above 0.4 falls from 1.00 to 0.37), while the
  genuinely overcast chips stay cloudy (Paraguay_126224, _12870, _36015, _198534, _40936:
  share stays 1.00). So B10 alone pins the probability at 1 on the false chips.
- Physically this is a high thin-cirrus veil (B10, 1375 nm, sees it; the other bands see
  through it) or a B10 artefact of these acquisitions; either way s2cloudless weighs B10
  heavily and the mask stops meaning "cloud the optical model cannot see through".

**Why it is not fixed.** s2cloudless is trained with B10 as a feature; feeding a doctored B10
is off-distribution and would silently change every mask. Switching cloud detectors is out of
scope. Instead the affected chips are flagged.

**Flag rule** (`floodfusion.cloud_audit.mask_unreliable`, computed from `chip_audit.csv`
columns): `mask_unreliable = 1` iff

- cloud fraction ≥ 0.94 **at threshold 0.6** — near-uniform saturation that survives raising
  the threshold (genuine broken cloud and haze collapse there: Pakistan_712873 falls to 0.71,
  Ghana_161233 to 0.76), **and**
- dim share ≥ 10% at 0.4 (share of masked pixels with mean B2–B4 TOA < 0.15) — dark ground
  visible inside the "cloud"; genuinely overcast chips are bright everywhere (dim 0%).

The cutoffs sit between visually checked chips: every false mask has cloud ≥ 0.945 at 0.6 and
dim ≥ 14%; every genuine one has cloud < 0.92 at 0.6 or dim ≤ 5%.

**Flagged chips (10):**

| split | chips |
|---|---|
| train | Paraguay_148318, Paraguay_215904, Paraguay_225187, Paraguay_36146, Sri-Lanka_956740 |
| valid | Paraguay_205585, Paraguay_252217, Paraguay_581976 |
| test | Paraguay_913449, Sri-Lanka_1049830 |
| bolivia | none |

- **Paraguay_913449 is flagged** (dim 98%: fields visible under a probability-1.0 map);
  **Paraguay_40936 is not** (dim 0%: a genuinely thick veil, and its mask survives the
  B10 substitution).
- All 10 sit in the 60–100% bin at 0.4, so excluding them moves the test 60–100% bin from
  11 to 9 chips. Sri-Lanka_1049830 is the most borderline flag: a real semi-transparent veil,
  but ground shows through and its saturation carries the B10 signature.
- The flag never changes training or the synthetic-cloud settings; it only annotates the
  real-cloud bins, which are reported with and without the flagged chips.

## Reproduce

```
python scripts/audit_clouds.py --root <data> --out <dir>             # full audit (clouds extra)
python scripts/flag_cloud_masks.py --audit <dir>/chip_audit.csv      # flags + cloud_cover_t06.csv
python scripts/diagnose_cloud_saturation.py --root <data>            # B10 diagnosis per chip
```
