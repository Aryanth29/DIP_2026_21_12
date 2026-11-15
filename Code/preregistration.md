# Pre-registered rules: items 1-4 and E vs S (written 2026-11-14, BEFORE any of them was evaluated)

The current rule was frozen first: `classification_rule_frozen_73galaxies.json` (CA rule, 73 galaxies, 26 groups,
LOGO-CV 64.4 %). Every rule below is fixed here before its effect on accuracy is measured. Each item is reported
**separately and combined**, with leave-one-group-out (LOGO) CV, for three tasks: 3-class (E/S/Irr),
early vs late (E vs S+Irr), and E vs S. Where a choice between alternatives is made, it is made by **nested**
LOGO-CV (inner LOGO on the training groups only). All results are reported, including those that do not help.

Honesty caveat: all 73 galaxies have already been inspected (development stage-14 error analysis), so none of them is a
blind test. The protection is (a) these rules are fixed before evaluation and are justified independently of
which galaxies they fix, (b) nested CV for every selection, (c) the frozen rule for future, newly downloaded
galaxies.

## Item 1 — steadier threshold fitting (lower variance)
Three threshold-fitting procedures for the same rule families (no new features):
- **grid** (current): exhaustive search maximising balanced accuracy (then accuracy, then median of ties).
- **midpoint**: no search. Each threshold = midpoint between the two class MEDIANS it separates:
  `C_E = (med C[E] + med C[S]) / 2`, `A_E = (med A[E] + med A[S]) / 2`, `A_hi = (med A[S] + med A[Irr]) / 2`
  (for 2-class tasks: the two classes of the task, with S+Irr pooled as "late" for early vs late).
- **bagged**: the grid fit repeated on 101 bootstrap resamples of GROUPS (resample groups with replacement);
  each threshold = median over the resamples.
Selection among the three by nested LOGO-CV.

## Item 2 — filter homogenisation (label-free F606W -> F814W calibration)
- Morphology of multi-band fields is already measured on the reddest band (F814W). The problem is the
  single-band F606W fields (HCG 16 ACS, 16 WFPC2, 18, 40, 56, 88, 93, 95).
- New engine option `measure_bands: ["F606W"]`: in a multi-band job, every object is ALSO measured on the
  F606W linear image with the SAME segmentation (columns `C_F606W`, `A_F606W`, `S_F606W`, `measurable_F606W`).
- Calibration sample: every object (labelled or not; **no labels are used**) in the multi-band ACS jobs that is
  measurable in both bands, with r_petro >= 2" and S/N per pixel >= 5 in both.
- Relation: Theil-Sen straight line `X_814 = a * X_606 + b` for X in {C, A}.
- Applied to every measurement whose Filter is exactly `F606W` (both ACS and WFPC2); other bands unchanged.

## Item 3 — label cleaning (independent cross-check with HyperLEDA)
- For every labelled Hickson member in the sample, HyperLEDA `meandata` (t, e_t) is queried; match = nearest
  HyperLEDA galaxy within max(5", 0.25 x Hickson diameter).
- HyperLEDA class ranges (same mapping as the labels): E: t <= -1; S: 1 <= t <= 7; Irr: t >= 8.
- **Disputed** if the whole interval [t - e_t, t + e_t] lies inside the range of a class DIFFERENT from the Hickson
  label. Intervals touching the Hickson class, or the S0/a (-1,1) or Sd/Sdm (7,8) gaps, are not disputed.
  No HyperLEDA type -> not disputed ("unverified"). Stephan's Quintet labels (NED/GZ2) and HyperLEDA-only labels
  are not re-checked.
- Accuracy is reported on ALL labels (headline) and on the CLEAN subset (disputed removed), never only the latter.

## Item 4 — frame-edge / chip-gap truncation (label-free simulation)
- Sample: every object (labels unused) in the measurement caches that is measurable, has r_petro >= 3" and no
  missing aperture (unrecoverable_frac = 0, aperture_clipped_frac = 0, neighbour_frac <= 0.05).
- Each is re-measured after masking a straight-edged half-plane (random orientation; offset chosen to cover a
  target fraction f = 0.05 ... 0.50 of the 1.5 r_p aperture) as no-data, with the pipeline's own code
  (symmetric filling included). Recorded: unrecoverable_frac, dC, dA.
- Tolerance: 75th percentile of |dC| <= 0.2 and of |dA| <= 0.03 (about a quarter of the E-S separation in
  median C (0.87) and A (0.13)).
- New `max_missing_frac` = the largest unrecoverable_frac bin within tolerance (rounded down to 0.05). If that
  is >= the current 0.25, nothing changes. Galaxies above it become "unmeasurable" (reported, not scored).

## E vs S — new feature tested
- **A_hp** (high-pass asymmetry): asymmetry of the unsharp-masked image H = I - boxcar(I, 0.25 r_p) inside
  1.5 r_p, normalised by sum|I|, minus the same quantity for the nearest empty sky box. Rationale: neighbour
  halos and tidal lopsidedness are LARGE-scale (removed by the high-pass); spiral arms, HII regions and dust lanes
  are small-scale and not point-symmetric at small scale.
- Tested in the E-vs-S rule `E if C > C_E and A_hp < X_E` and in place of A for the E condition in the 3-class rule
  (Irr still decided by A), against A and RFF, by nested LOGO-CV.

## Amendment A1 (item 4) — written after the label-free simulation, BEFORE evaluating accuracy
The simulation (`Output/baseline_output/truncation_simulation.csv`, 30 galaxies x 10 cuts) shows that a
straight-edge cut is always fully recovered by symmetric filling as far as `unrecoverable_frac` is concerned
(it stays 0 for every cut up to half the aperture), so the pre-registered rule on `unrecoverable_frac` can never
change anything. The bias instead grows with the CLIPPED share of the aperture (`aperture_clipped_frac`):
75th-percentile |dA| = 0.020 at a median clipped share of 0.165, 0.040 at 0.21 (tolerance 0.03); |dC| stays
within 0.2 up to 0.43. Same tolerance, applied to the quantity that tracks the bias:
**a galaxy with aperture_clipped_frac > 0.15 is flagged unmeasurable** (0.165 rounded down to 0.05).
No labels were looked at for this choice.

## Item 3 outcome (recorded before evaluating accuracy)
HyperLEDA cross-check (`catalogs/hickson_leda_crosscheck.csv`): 69 consistent, 3 unverified (no type), 1 disputed:
HCG 95c (Hickson Sm -> Irr; HyperLEDA t = 4.4 +- 1.4 -> S).

# Pre-registered rules: aperture recovery, disc detection, A_in — written 2026-11-14 BEFORE evaluation (baseline = `Output/final_output`, 3-class 77.8 %, E vs S 86.3 %)

Adoption rule for everything below: a change enters the production configuration only if it does not lower
the nested-LOGO accuracy of EITHER the 3-class task or E vs S (on the measurable sample), and, for idea 1, only if
its label-free simulation passes. All results are reported.

## Idea 1 — aperture recovery for truncated galaxies (instead of dropping them)
- For an object whose 1.5 r_p aperture is clipped by more than `max_clipped_frac` (0.15), A is re-measured
  inside the largest concentric circle R_A <= 1.5 r_p whose clipped share is <= 0.15, with the floor
  R_A >= 0.5 r_p. The asymmetry centre, C, S, Gini and M20 (all measured on the symmetric-filled image) are
  unchanged. If no R_A >= 0.5 r_p satisfies the limit, the object stays unmeasurable.
- Label-free validation: the truncation simulation (same 30 galaxies, same cuts) is repeated with recovery on.
  Pass = 75th-percentile |dA| <= 0.03 AND |dC| <= 0.2 for every cut bin whose full-aperture clipped share is
  <= 0.50. If it passes only up to some bin, recovery is allowed only up to that full-aperture clipped share
  (`max_clipped_frac_recover`), rounded down to 0.05.
- A_in (0.5 r_p) is measured inside min(0.5 r_p, R_A).

## Idea 2 — disc detection for bulge-dominated spirals (E vs S)
- Isophote fit (photutils.isophote.Ellipse, fixed centre = asymmetry centre) on the symmetric-filled image with
  other objects masked, semi-major axes from 2 px to 1.5 r_p. Features:
  `e_in` = median ellipticity over sma in [0.2, 0.5] r_p; `e_out` = median over [0.8, 1.5] r_p;
  `de` = e_out − e_in; `pa_twist` = circular std of the PA over [0.5, 1.5] r_p (degrees).
  Fallback when the isophote fit fails: flux-weighted second-moment ellipticity in the same annuli.
- Rationale (fixed in advance): an inclined disc is flatter than its bulge, so ellipticity RISES outward with
  constant PA; in an elliptical it is roughly constant (PA twists are common). Rules tested, all with grid fitting:
  3-class: `CA`, `C-Ain`, `CA+e_out` (E also requires e_out < e_E), `C-Ain+e_out`, `CA+de` (E also requires
  de < de_E), `C-Ain+de`; 2-class analogues `E2:A`, `E2:A_in`, `E2:A+e_out`, `E2:A_in+e_out`, `E2:A+de`,
  `E2:A_in+de`. Selection by nested LOGO over this set; the single-family LOGO scores are reported as reference.

## Idea 3 — A_in on the cleaned sample
- `C-Ain` / `E2:A_in` are part of the option set above (A_in was better than A at separating E from S on the
  final configuration sample: rank separation 0.85 vs 0.80, descriptive only).

## Idea 1 outcome (recorded before evaluating ideas 2-3)
FAILED the pre-registered simulation (`Output/baseline_output/truncation_simulation_recover.csv`): with recovery the
75th-percentile |dA| is 0.036 at 16.5 % clipped (dropped: 0.020), 0.049 at 21 %, 0.068 at 24.5 %, 0.078 at 29 %.
A inside a smaller circle is a different quantity from A inside 1.5 r_p (the thresholds' scale), and the offset
exceeds the truncation bias it avoids. The rule allows recovery only up to the last passing bin (13 %, rounded
down to 0.10) -- below max_clipped_frac, so recovery never applies. Not adopted; `recover_aperture` stays False.

# Pre-registered rule: bulge gate on the Irr branch — written 2026-11-14 BEFORE evaluation (baseline = `Output/final_output`: CA rule, 3-class LOGO-CV 77.8 %, 49/63)

## Idea: a bulge gate on the Irr branch
- Rationale: Sm/Im galaxies have no bulge, and in the CAS plane (Conselice 2003) irregulars sit at low C. A galaxy
  that is both concentrated and asymmetric is more likely a disturbed spiral or a tidally distorted early type than
  an irregular.
- How the idea arose (NOT blind): from the final configuration error table. 6 of the 7 galaxies wrongly called Irr have
  C = 3.77-4.70, while the true Irr have median C = 3.05. Only the error list and per-class C/A summaries were
  looked at; no accuracy of the new rule was computed before this entry.
- Family `CA+Cirr`: **Irr if A >= A_hi and C < C_I; E if C > C_E and A < A_E; else S.** C_I is fitted by the same
  grid as every other threshold (candidates = up to 14 midpoints of the training C values, plus "off" = +inf, which
  makes the rule exactly CA), same objective and tie-breaking (balanced accuracy, then accuracy, then the tie
  nearest the median; an optional term stays off if most ties have it off). Nothing is set by hand.
- Variant `CA+Cirr+Chi`: additionally **E if C > C_hi regardless of A** (tidally disturbed ellipticals); C_hi fitted
  the same way, with an "off" option. Offered only inside nested selection.
- Scope: only the 3-class rule changes. The 2-class tasks (E vs S, early vs late) use their own rules (`E2:*`),
  which have no Irr branch, so they are unchanged by construction and are not re-evaluated.

## Adoption rule (all must hold)
1. Nested LOGO-CV on the 63 measurable galaxies, family chosen inside every outer fold (inner LOGO on the training
   groups) from {CA, CA+Cirr}: at least 50/63, i.e. strictly better than CA's 49/63.
2. Robustness on other measurements of the same galaxies: the LOGO-CV accuracy of CA+Cirr must not be lower than
   CA's on (a) the baseline run baseline (73 galaxies, truncated ones included, `Output/baseline_output`) and (b) the
   macOS neighbour-model run measurement (`C_v12`, `A_v12` in `Output/neighbour_model_output/neighbour_model_evaluation_galaxies.csv`), on the
   same 63 galaxies.
3. The variant is adopted instead only if nested selection from {CA, CA+Cirr, CA+Cirr+Chi} beats the nested number
   of rule 1 by at least one galaxy AND the variant passes rule 2.
- Reported whatever the outcome: per-family LOGO-CV, nested LOGO-CV, galaxies fixed/broken, thresholds on all data.
- If adopted, the pipeline gets the family (`fit_rule` / `predict_rule`); `classification_rule_frozen_73galaxies.json`
  (CA, fitted before final configuration) stays the frozen rule for the next genuine hold-out test.

## Bulge gate outcome — NOT adopted (`Output/final_output/bulge_gate_evaluation.csv`)
| LOGO-CV, 3 classes | CA | CA+Cirr | CA+Cirr+Chi |
|---|---|---|---|
| final configuration, adopted measurement (63) | 77.8 % (49) | 73.0 % (46) | 71.4 % (45) |
| baseline run baseline, truncated included (73) | 64.4 % (47) | 63.0 % (46) | 65.8 % (48) |
| neighbour-model run macOS measurement, same 63 | 77.8 % (49) | 73.0 % (46) | 73.0 % (46) |

- Rule 1 fails: nested LOGO over {CA, CA+Cirr} = 76.2 % (48/63), needs >= 50/63 (the gate was chosen in 1 of 23 folds).
- Rule 2 fails on both other measurements (63.0 < 64.4; 73.0 < 77.8). Rule 3 fails: nested over all three = 74.6 % (47/63).
- Why: the premise does not hold in this sample. 3 of the 12 Irr are concentrated (HCG 16c C = 3.91, 95c 4.10,
  31c 4.43; interacting / starburst systems), inside the C range of the asymmetric S/E the gate was meant to rescue
  (3.73-4.70). Any gate below 4.43 removes true Irr, so the fits put C_I at 4.4-4.6, where it only moves HCG 40b/79c
  from Irr to S (still wrong). When HCG 16, 31, 87 or 95 is held out, the training fit does place a low gate
  (C_I = 3.2-4.1) and re-balances A_hi, which breaks HCG 16c, 16d, 31c and 87c. Fixed: HCG 16a only.
- Side result: the plain CA rule on the macOS measurement of the same 63 galaxies also scores 77.8 % (49/63), so the
  final configuration number is not a Windows-measurement artefact.

# Pre-registered rule: resolution and S/N homogenisation of A (written 2026-11-14 BEFORE any computation)

Baseline: `Output/final_output`, CA rule, 3-class LOGO-CV 77.8 % (49/63), E vs S 86.3 % (44/51).

## Motivation
All images are resampled to a common work scale (0.18-0.26"/px), but the galaxies span r_p = 10-290 work pixels and
S/N per pixel 5-200. A big, near or bright galaxy shows small-scale structure (HII knots) and faint outer features
that a small, far or faint one cannot, which inflates its A. Goal: measure A for every galaxy at the same number of
resolution elements per r_p and the same surface-brightness S/N.

## Measurement A_hom (from the baseline run measurement cache)
Same cutout, neighbour masking, symmetric filling, asymmetry centre and 1.5 r_p aperture as A; only the image changes.
1. Resolution. Native resolution element = 2 work pixels for every image. Target FWHM_T = r_p / N_T. The cutout is
   convolved (normalised convolution over valid pixels) with a Gaussian of FWHM_k = sqrt(FWHM_T^2 - 2^2); a galaxy
   with r_p / 2 <= N_T is not smoothed.
2. N_T (label-free) = 10th percentile of r_p / 2 (work pixels) over the 73 catalogue galaxies of the baseline run baseline
   sample, rounded down to an integer.
3. Noise terms. The 5 sky boxes nearest the galaxy (plane removed) are smoothed with the same kernel (periodic
   boundaries); per-pixel std sigma_s and asymmetry term a_s = mean |B_s - B_s,180| as in the pipeline. For
   FWHM_k > 8 px (a quarter of the 32-px box) both are extrapolated from their value at 8 px as proportional to
   1 / FWHM_k (white-noise scaling).
4. S/N. eta = mu_p / sigma_s, mu_p = mean surface brightness in the Petrosian annulus [0.8, 1.25] r_p. Target eta_T
   (label-free) = 10th percentile of eta over the same 73 galaxies at N_T. For a galaxy with eta > eta_T, white
   Gaussian noise is added to the cutout before smoothing, scaled so that the smoothed noise reaches mu_p / eta_T;
   the background term becomes sqrt(a_s^2 + (2 sigma_add,s / sqrt(pi))^2). A_hom = median over 5 noise realisations
   (fixed seeds). A galaxy with eta <= eta_T gets no noise.
5. A_R (step 1 only, no noise) is computed as a diagnostic, not evaluated against labels.

## Label-free validation (must pass before any label is used)
- Objects: measurable in the baseline run baseline, aperture_clipped_frac = 0, unrecoverable_frac = 0,
  neighbour_frac <= 0.05, r_p >= 3" (the truncation-simulation selection), up to 6 per job, largest r_p first.
- Each frame is degraded to mimic the same field seen b = 2, 3, 4 times farther away: b x b block mean, Gaussian PSF
  of FWHM sqrt(2^2 - (2/b)^2) px, white noise restoring the native per-pixel sky sigma; segmentation subsampled,
  no-data block-OR, sky boxes re-found. A degraded copy is used only if its r_p / 2 >= N_T (so that the
  homogenisation applies in full to both versions).
- Compared: A, A_R and A_hom of each degraded copy against the same quantity on the original.
- PASS for A_hom: 75th-percentile |dA_hom| <= 0.03 at every b AND below plain A's 75th-percentile |dA| at every b.
- If plain A already has 75th-percentile |dA| <= 0.03 at every b, the premise (A depends on resolution in our data)
  is false and the idea is dropped without label evaluation.

## Label evaluation (only if the validation passes), 63 measurable galaxies
- 3-class families: CA; `CA[hom]` = A_hom in place of A everywhere; `CA[Irr:hom]` = A_hom only in the Irr test
  (Irr if A_hom >= A_hi; E if C > C_E and A < A_E; else S), because knots matter most at the S/Irr boundary.
- E vs S: E2:A versus E2:A_hom.
- Adoption (all must hold):
  1. nested LOGO-CV, 3 classes, over {CA, CA[hom], CA[Irr:hom]} >= 50/63;
  2. if the family adopted uses A_hom in the E test: nested LOGO-CV, E vs S, over {E2:A, E2:A_hom} >= 44/51;
  3. the family selected on all 63 is not worse than CA by LOGO-CV on the baseline run baseline sample (73 galaxies,
     truncated ones included). (A macOS check is impossible: there is no Mac measurement cache.)

## Amendment A2 (homogenisation) — written after the first simulation run, BEFORE any label is used
- Label-free targets (`Output/baseline_output/homogenisation/targets.json`): N_T = 17 (P10 of r_p/2 = 17.2;
  median 39.2), eta_T = 8.02 (P10 of eta; median 34.5). Noise is therefore added to about 90 % of the galaxies.
- The pre-registered selection gave only 11 usable degraded copies (9 at b = 2, 2 at b = 3, 0 at b = 4): with
  N_T = 17 a copy is usable only if the original has r_p >= 68 px (b = 2), 102 px (b = 3), 136 px (b = 4), and no
  clean galaxy has r_p >= 136 px. "At every b" cannot be tested as written.
- First-run numbers, reported as they are: b = 2 (n = 9): 75th-percentile |dA| = 0.038, |dA_R| = 0.017,
  |dA_hom| = 0.009; b = 3 (n = 2): 0.012 / 0.010 / 0.007.
- Amendment (sample size only; criterion unchanged): selection widened to neighbour_frac <= 0.30 (other conditions
  unchanged), no per-job cap. The pass criterion is applied at b = 2 (needs >= 10 usable copies); b = 3 is reported
  but has too few copies to decide; b = 4 is impossible. The premise check (plain A 75th-percentile |dA| > 0.03) is
  also applied at b = 2. Copies whose degraded r_p/2 falls below N_T are reported separately: they show the residual
  bias for the 10 % of galaxies that are under-resolved relative to the target.

## Homogenisation outcome — validation PASSED, classification NOT improved -> NOT adopted
Validation (`Output/baseline_output/homogenisation/homogenisation_simulation.csv`, amended sample, 42 galaxies):
| b | usable copies | 75th pct. abs(dA), plain A | A_R | A_hom |
|---|---|---|---|---|
| 2 | 16 | 0.055 | 0.021 | **0.015** |
| 3 | 4 | 0.048 | 0.035 | 0.022 |
| 4 | 1 | 0.155 | 0.113 | 0.080 |
Premise holds (plain A changes by > 0.03 when the same galaxy is seen at half the resolution) and A_hom passes at
b = 2. Copies that fall BELOW N_T after degradation keep a bias (median dA_hom = -0.04 to -0.05, 75th pct. 0.09-0.10):
galaxies with r_p/2 < N_T (the smallest 10 %) cannot be homogenised upwards.

Label evaluation (`Output/final_output/homogenisation_evaluation.csv`):
| LOGO-CV | CA / E2:A | CA[hom] / E2:A_hom | CA[Irr:hom] | nested |
|---|---|---|---|---|
| 3 classes (63) | 77.8 % (49) | 76.2 % (48) | 76.2 % (48) | 77.8 % (49) |
| E vs S (51) | 86.3 % (44) | 82.4 % (42) | - | 84.3 % (43) |
| early vs late (63) | 88.9 % (56) | 88.9 % (56) | - | 88.9 % (56) |
| 3 classes, baseline run baseline (73) | 64.4 % (47) | 72.6 % (53) | 74.0 % (54) | 65.8 % (48) |
- Rule 1 fails (nested 49/63, needs >= 50). The family selected on all 63 is CA, so rules 2-3 do not arise.
- A_hom does what it was designed to do: spirals lose more asymmetry (median -0.038) than irregulars or
  ellipticals (-0.017), F606W galaxies more than others (-0.033 vs -0.018); the targets drop (HCG 16a 0.289 -> 0.190,
  40c 0.309 -> 0.233, 40d 0.324 -> 0.246). But the refitted Irr threshold drops with them (0.317 -> 0.232), so 40c/40d
  stay above it, and new errors appear: HCG 16d (Irr, r_p = 10 px, the smallest galaxy: no smoothing possible, noise
  added) falls below, and two faint spirals (HCG 7d, 87c; eta 14 and 7) RISE (0.315 -> 0.370, 0.285 -> 0.338), which
  points to an under-estimated noise term after smoothing for low-S/N galaxies (not covered by the simulation, whose
  galaxies are large and bright). E vs S loses 2 galaxies: smoothing removes arm signal.
- Post-hoc observation (NOT part of the adoption rule): on the 73-galaxy baseline the single-family LOGO gains are
  large (+6/+7, none of them truncated galaxies), but nested selection keeps only +1 (48/73): unstable thresholds.
