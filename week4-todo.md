# Week 4 To-Do — Segmentation & CAS Feature Extraction

Pure project work (no ECE501 deck exercises this week). Applied to both primary targets:
**Stephan's Quintet (HCG 92)** and **Pillars of Creation (M16)**. Builds directly on the
restoration/enhancement pipeline finalized in Week 3 (Wiener deconvolution → bilateral
denoise → Asinh stretch → JWST RGB composite).

Scope for this week: **Segmentation + Source Isolation (Track E) and CAS Feature Extraction
(Track F) only.** Rule-based classification (Elliptical/Spiral/Irregular) and Galaxy Zoo
benchmarking are deferred to Week 5.

---

## Track E — Segmentation & Source Isolation

### E1. Adaptive Thresholding
- [x] Apply global Otsu thresholding to the Week 3 final JWST composite (Stephan's Quintet) as a baseline
- [x] Apply adaptive/local thresholding (local median or `skimage.filters.threshold_local`) and compare against global Otsu
- [x] Repeat both thresholding approaches on the Pillars of Creation composite
- [x] Visually compare: which threshold method better separates faint galaxy/nebula structure from background sky

### E2. Morphological Cleanup
- [x] Apply binary morphological opening to remove small noise specks from the thresholded mask
- [x] Apply binary morphological closing to fill small gaps within galaxy/pillar regions
- [x] Apply hole filling (`scipy.ndimage.binary_fill_holes`) to close interior gaps in extended objects
- [x] Document structuring element size/shape chosen and why

### E3. Star–Galaxy Separation (Stephan's Quintet only — Pillars has no comparable point-source confusion)
- [x] Compute compactness/concentration for each detected blob (e.g., area vs. bounding-box ratio, or peak-to-total-flux ratio)
- [x] Flag point-source candidates (foreground stars, including those with diffraction spikes) using a compactness threshold
- [x] Remove flagged point sources from the galaxy mask
- [x] Before/after comparison: mask with stars included vs. stars removed

### E4. Connected Component Labeling & Cutout Extraction
- [x] Run connected component labeling (`cv2.connectedComponents` or `skimage.measure.label`) on the cleaned mask
- [x] For Stephan's Quintet: extract individual bounding-box cutouts for NGC 7317, NGC 7318A, NGC 7318B, NGC 7319, NGC 7320 (cross-reference against known coordinates/WCS to assign correct labels)
- [x] For Pillars of Creation: extract bounding-box cutouts for each of the three main pillar structures
- [x] Save each cutout as an individual image file, named by object ID
- [x] Discard components below a minimum area threshold (spurious noise blobs)

### E5. Segmentation Quality Check
- [x] Overlay final segmentation mask / bounding boxes on the original composite for visual verification
- [x] Note any merged objects (two galaxies labeled as one component) or split objects (one galaxy broken into multiple components) and how handled
- [x] Document final segmentation parameters (threshold method, structuring element, min area, compactness threshold) as the "recommended parameter set" going forward

---

## Track F — CAS Feature Extraction

Computed per segmented object (Stephan's Quintet galaxies; Pillars treated as extended
structures where applicable — note CAS was designed for galaxies, so flag any metrics that
don't transfer cleanly to the pillar cutouts).

### F1. Concentration (C)
- [x] For each cutout, compute the curve of growth (cumulative flux vs. radius from photometric center)
- [x] Determine r20 and r80 (radii enclosing 20% and 80% of total flux)
- [x] Compute $C = 5 \log_{10}(r_{80}/r_{20})$
- [x] Tabulate C for all Stephan's Quintet galaxies

### F2. Asymmetry (A)
- [x] Determine photometric center for each cutout
- [x] Rotate each cutout 180° about its photometric center
- [x] Compute residual flux: |original − rotated|, summed
- [x] Repeat the rotation/residual procedure on a background sky patch to estimate noise asymmetry
- [x] Compute $A$ = (galaxy residual − background residual) / original flux
- [x] Tabulate A for all Stephan's Quintet galaxies

### F3. Smoothness / Clumpiness (S)
- [x] Apply a boxcar (uniform) smoothing filter to each cutout at a chosen smoothing scale
- [x] Compute residual: original − smoothed
- [x] Repeat on a background patch to estimate noise contribution
- [x] Compute $S$ = (galaxy residual − background residual) / original flux
- [x] Tabulate S for all Stephan's Quintet galaxies

### F4. Geometric Shape Moments
- [x] Compute image moments (`cv2.moments` or `skimage.measure.moments`) for each cutout
- [x] Derive centroid, area, perimeter from the moments/mask
- [x] Compute eccentricity and ellipticity
- [x] Compute Hu's moment invariants
- [x] Tabulate all shape moments alongside C, A, S in a single results table

### F5. CAS Validation Checks
- [x] Sanity-check C, A, S ranges against published literature values for similar galaxy types (rough order-of-magnitude check only — no classification yet)
- [x] Flag any object whose CAS values look anomalous (e.g., due to incomplete segmentation or residual artifacts) for re-inspection
- [x] Save final CAS + shape moment table as CSV for use in Week 5 classification

---

## Documentation

### G1. Week 4 Report Section
- [x] Write a short technical section summarizing segmentation methodology (thresholding, morphology, star-galaxy separation, connected components) and CAS extraction methodology
- [x] Include figures: segmentation mask overlay, example cutouts, CAS results table
- [x] State final segmentation + CAS parameter set carried forward into Week 5 classification

---

## What's Still Deliberately Excluded
No rule-based classification (Elliptical/Spiral/Irregular decision tree) and no Galaxy Zoo
benchmarking this week — both deferred to Week 5, per the project roadmap in context.md.

---

## Notes
- Segmentation and CAS extraction run on the JWST NIRCam composite as primary dataset (per Week 3 decision); HST not re-processed unless a specific comparison is requested.
- Pillars of Creation is architecturally different from Stephan's Quintet (nebular pillars, not discrete galaxies) — CAS metrics may need reinterpretation or may only be reported for Stephan's Quintet depending on how cleanly segmentation works on the pillars.
