# Morphological Classification of Galaxies — Results Report

Generated: 2026-11-03 15:45

Targets this run: Stephan's Quintet (HCG 92), HCG 90, HCG 16

Pipeline: classical, deterministic, rule-based DIP — no ML, no trained classifiers, the same reusable pipeline looped once per target.

---

## Stephan's Quintet (HCG 92)

- Data: JWST NIRCam (F090W/F150W/F200W) + HST ACS/WFC
- Name cross-matching: catalog-based (WCS + topology)
- Background-relative detection: auto-selected k=3.20σ, **63** raw cores → **33** after area filter
- Star–galaxy separation: **20 stars**, **13 galaxy candidates**
- Cutouts extracted: **5**
- Measured PSF FWHM: **4.20 px** (std=3.32px, n=20)

![Stephan's Quintet (HCG 92) — final segmentation overlay](../Output/week_5_output/sq/sq_segmentation_overlay.png)

*Stephan's Quintet (HCG 92) — final segmentation overlay*

![Stephan's Quintet (HCG 92) — extracted galaxy cutouts](../Output/week_5_output/sq/sq_cutout_gallery.png)

*Stephan's Quintet (HCG 92) — extracted galaxy cutouts*

| Galaxy | C | A | S | S_gran | eccentricity | ellipticity | area_px |
|---|---|---|---|---|---|---|---|
| NGC7317 | 1.7024 | 0.2290 | 0.0000 | 0.0155 | 0.6596 | 0.2484 | 191 |
| NGC7318A | 1.4892 | 0.0000 | 0.0000 | 0.0011 | 0.3267 | 0.0549 | 576 |
| NGC7318B | 1.5793 | 0.0782 | 0.0000 | 0.0090 | 0.4498 | 0.1069 | 347 |
| NGC7319 | 1.6534 | 0.0176 | 0.0000 | 0.0117 | 0.7900 | 0.3869 | 1130 |
| NGC7320 | 2.3274 | 0.0246 | 0.0000 | 0.0158 | 0.9495 | 0.6862 | 349 |

`S` is the Gaussian-residual clumpiness (background-corrected; can clip to exactly 0). `S_gran` is the granulometric clumpiness (Mathematical Morphology; no background term, so it cannot clip the same way) — use it as the primary clumpiness value wherever `S` reads 0.

![Stephan's Quintet (HCG 92) — CAS + granulometric parameter distributions](../Output/week_5_output/sq/sq_cas_distributions.png)

*Stephan's Quintet (HCG 92) — CAS + granulometric parameter distributions*


---

## HCG 90

- Data: HST (single filter)
- Name cross-matching: none — generic object labels (no name classifier)
- Background-relative detection: auto-selected k=10.00σ, **722** raw cores → **288** after area filter
- Star–galaxy separation: **253 stars**, **35 galaxy candidates**
- Cutouts extracted: **35**
- Measured PSF FWHM: **1.71 px** (std=1.42px, n=252)

![HCG 90 — final segmentation overlay](../Output/week_5_output/hcg90/hcg90_segmentation_overlay.png)

*HCG 90 — final segmentation overlay*

![HCG 90 — extracted galaxy cutouts](../Output/week_5_output/hcg90/hcg90_cutout_gallery.png)

*HCG 90 — extracted galaxy cutouts*

| Galaxy | C | A | S | S_gran | eccentricity | ellipticity | area_px |
|---|---|---|---|---|---|---|---|
| hcg90_Obj1 | 1.8428 | 0.1188 | 0.1161 | 0.0027 | 0.4586 | 0.1114 | 10454 |
| hcg90_Obj2 | 1.8736 | 0.1212 | 0.1043 | 0.0039 | 0.4097 | 0.0878 | 8924 |
| hcg90_Obj3 | 2.1617 | 0.1469 | 0.1038 | 0.0025 | 0.4972 | 0.1324 | 4583 |
| hcg90_Obj4 | 2.2321 | 0.0969 | 0.0973 | 0.0030 | 0.5239 | 0.1482 | 4312 |
| hcg90_Obj5 | 2.2047 | 0.1478 | 0.0674 | 0.0037 | 0.6597 | 0.2484 | 4104 |
| hcg90_Obj6 | 2.1905 | 0.1122 | 0.0992 | 0.0039 | 0.5704 | 0.1786 | 3756 |
| hcg90_Obj7 | 1.7516 | 0.1567 | 0.0090 | 0.0032 | 0.8046 | 0.4061 | 3324 |
| hcg90_Obj8 | 2.2954 | 0.1184 | 0.0943 | 0.0030 | 0.4165 | 0.0908 | 2014 |
| hcg90_Obj9 | 1.6472 | 0.1490 | 0.0051 | 0.0004 | 0.5298 | 0.1519 | 1738 |
| hcg90_Obj10 | 2.1774 | 0.2039 | 0.0990 | 0.0029 | 0.5410 | 0.1590 | 1603 |
| hcg90_Obj11 | 1.8033 | 0.0000 | 0.0459 | 0.0050 | 0.8160 | 0.4220 | 1418 |
| hcg90_Obj12 | 1.6200 | 0.0112 | 0.0269 | 0.0074 | 0.6095 | 0.2072 | 1067 |
| hcg90_Obj13 | 1.5687 | 0.0704 | 0.1241 | 0.0126 | 0.7737 | 0.3665 | 840 |
| hcg90_Obj14 | 1.5582 | 0.0000 | 0.0000 | 0.0015 | 0.4905 | 0.1286 | 835 |
| hcg90_Obj15 | 1.6544 | 0.0000 | 0.0572 | 0.0109 | 0.6015 | 0.2011 | 799 |
| hcg90_Obj16 | 2.3733 | 0.2815 | 0.0516 | 0.0016 | 0.9482 | 0.6825 | 730 |
| hcg90_Obj17 | 1.7455 | 0.0000 | 0.0619 | 0.0077 | 0.7773 | 0.3709 | 693 |
| hcg90_Obj18 | 1.7201 | 0.0531 | 0.0704 | 0.0042 | 0.2489 | 0.0315 | 584 |
| hcg90_Obj19 | 2.6808 | 0.3343 | 0.0878 | 0.0048 | 0.9744 | 0.7750 | 513 |
| hcg90_Obj20 | 1.5708 | 0.1697 | 0.0085 | 0.0006 | 0.6172 | 0.2132 | 435 |
| hcg90_Obj21 | 2.1513 | 0.1945 | 0.0561 | 0.0029 | 0.9198 | 0.6076 | 405 |
| hcg90_Obj22 | 1.5779 | 0.3034 | 0.0674 | 0.0121 | 0.7915 | 0.3889 | 385 |
| hcg90_Obj23 | 1.7488 | 0.0000 | 0.1524 | 0.0285 | 0.8959 | 0.5558 | 379 |
| hcg90_Obj24 | 2.4315 | 0.1119 | 0.0744 | 0.0051 | 0.9086 | 0.5823 | 375 |
| hcg90_Obj25 | 2.0594 | 0.0000 | 0.0659 | 0.0102 | 0.8649 | 0.4981 | 361 |
| hcg90_Obj26 | 1.9659 | 0.4672 | 0.0903 | 0.0121 | 0.8647 | 0.4977 | 359 |
| hcg90_Obj27 | 1.9405 | 0.2269 | 0.1329 | 0.0053 | 0.2414 | 0.0296 | 348 |
| hcg90_Obj28 | 2.0110 | 0.2513 | 0.1332 | 0.0111 | 0.8903 | 0.5447 | 347 |
| hcg90_Obj29 | 1.8124 | 0.3026 | 0.1079 | 0.0046 | 0.6511 | 0.2410 | 336 |
| hcg90_Obj30 | 1.7372 | 0.1575 | 0.0069 | 0.0022 | 0.7904 | 0.3874 | 334 |
| hcg90_Obj31 | 1.6369 | 0.4900 | 0.0471 | 0.0083 | 0.7594 | 0.3493 | 308 |
| hcg90_Obj32 | 1.9009 | 0.3247 | 0.0545 | 0.0047 | 0.8746 | 0.5152 | 246 |
| hcg90_Obj33 | 1.8991 | 0.1958 | 0.0688 | 0.0111 | 0.7972 | 0.3962 | 198 |
| hcg90_Obj34 | 1.5319 | 0.2532 | 0.0078 | 0.0017 | 0.6779 | 0.2648 | 190 |
| hcg90_Obj35 | 2.1825 | 0.2174 | 0.0100 | 0.0012 | 0.9379 | 0.6532 | 182 |

`S` is the Gaussian-residual clumpiness (background-corrected; can clip to exactly 0). `S_gran` is the granulometric clumpiness (Mathematical Morphology; no background term, so it cannot clip the same way) — use it as the primary clumpiness value wherever `S` reads 0.

![HCG 90 — CAS + granulometric parameter distributions](../Output/week_5_output/hcg90/hcg90_cas_distributions.png)

*HCG 90 — CAS + granulometric parameter distributions*


---

## HCG 16

- Data: HST (single filter)
- Name cross-matching: none — generic object labels (no name classifier)
- Background-relative detection: auto-selected k=12.80σ, **73** raw cores → **45** after area filter
- Star–galaxy separation: **20 stars**, **25 galaxy candidates**
- Cutouts extracted: **25**
- Measured PSF FWHM: **4.26 px** (std=1.95px, n=20)

![HCG 16 — final segmentation overlay](../Output/week_5_output/hcg16/hcg16_segmentation_overlay.png)

*HCG 16 — final segmentation overlay*

![HCG 16 — extracted galaxy cutouts](../Output/week_5_output/hcg16/hcg16_cutout_gallery.png)

*HCG 16 — extracted galaxy cutouts*

| Galaxy | C | A | S | S_gran | eccentricity | ellipticity | area_px |
|---|---|---|---|---|---|---|---|
| hcg16_Obj1 | 1.5185 | 0.0141 | 0.0000 | 0.0001 | 0.4757 | 0.1204 | 2976 |
| hcg16_Obj2 | 1.5458 | 0.0000 | 0.0247 | 0.0032 | 0.6965 | 0.2824 | 1962 |
| hcg16_Obj3 | 2.0415 | 0.0000 | 0.0710 | 0.0085 | 0.8111 | 0.4151 | 1828 |
| hcg16_Obj4 | 2.2616 | 0.0000 | 0.0602 | 0.0180 | 0.9386 | 0.6549 | 1645 |
| hcg16_Obj5 | 1.6211 | 0.2253 | 0.0170 | 0.0021 | 0.4724 | 0.1186 | 1578 |
| hcg16_Obj6 | 2.0964 | 0.0000 | 0.0966 | 0.0163 | 0.8410 | 0.4589 | 1288 |
| hcg16_Obj7 | 1.7332 | 0.1962 | 0.0517 | 0.0029 | 0.7394 | 0.3267 | 1228 |
| hcg16_Obj8 | 2.3275 | 0.0000 | 0.0590 | 0.0030 | 0.9534 | 0.6984 | 1018 |
| hcg16_Obj9 | 2.2406 | 0.4135 | 0.0800 | 0.0256 | 0.9068 | 0.5784 | 844 |
| hcg16_Obj10 | 1.9904 | 0.0646 | 0.0218 | 0.0052 | 0.8640 | 0.4965 | 660 |
| hcg16_Obj11 | 1.4778 | 0.0000 | 0.0817 | 0.0229 | 0.2574 | 0.0337 | 502 |
| hcg16_Obj12 | 1.9082 | 0.0000 | 0.0937 | 0.0141 | 0.8778 | 0.5210 | 471 |
| hcg16_Obj13 | 1.6569 | 0.1311 | 0.0202 | 0.0113 | 0.6996 | 0.2854 | 432 |
| hcg16_Obj14 | 3.1444 | 0.0692 | 0.0206 | 0.0506 | 0.9962 | 0.9126 | 364 |
| hcg16_Obj15 | 1.9584 | 0.1764 | 0.0178 | 0.0047 | 0.8698 | 0.5065 | 345 |
| hcg16_Obj16 | 1.9591 | 0.2361 | 0.0650 | 0.0089 | 0.8915 | 0.5470 | 333 |
| hcg16_Obj17 | 1.6115 | 0.0280 | 0.0252 | 0.0013 | 0.7079 | 0.2937 | 329 |
| hcg16_Obj18 | 2.1610 | 0.1331 | 0.0795 | 0.0188 | 0.9209 | 0.6101 | 323 |
| hcg16_Obj19 | 3.0237 | 0.1383 | 0.0182 | 0.0676 | 0.9935 | 0.8857 | 295 |
| hcg16_Obj20 | 1.9622 | 0.0889 | 0.0279 | 0.0116 | 0.8540 | 0.4798 | 263 |
| hcg16_Obj21 | 1.7484 | 0.3444 | 0.0433 | 0.0099 | 0.6133 | 0.2102 | 256 |
| hcg16_Obj22 | 1.6829 | 0.3159 | 0.0888 | 0.0082 | 0.4865 | 0.1263 | 245 |
| hcg16_Obj23 | 1.6379 | 0.2590 | 0.0183 | 0.0114 | 0.7643 | 0.3551 | 228 |
| hcg16_Obj24 | 1.5071 | 0.1408 | 0.0342 | 0.0028 | 0.4204 | 0.0927 | 198 |
| hcg16_Obj25 | 1.6466 | 0.3104 | 0.0000 | 0.0036 | 0.7184 | 0.3043 | 174 |

`S` is the Gaussian-residual clumpiness (background-corrected; can clip to exactly 0). `S_gran` is the granulometric clumpiness (Mathematical Morphology; no background term, so it cannot clip the same way) — use it as the primary clumpiness value wherever `S` reads 0.

![HCG 16 — CAS + granulometric parameter distributions](../Output/week_5_output/hcg16/hcg16_cas_distributions.png)

*HCG 16 — CAS + granulometric parameter distributions*


---

## Combined Cross-Target Table

Full diagnostic table for every target this run (incl. A_raw/A_bg_local/S_raw/S_bg_local/Hu moments) saved to: `../Output/week_5_output/cas_features_all_targets.csv`
