# Week 2 To-Do — ECE501 Exercises + Galaxy Classification Project

Two tracks this week: (1) the generic ECE501 skills exercises from the session deck, and
(2) the actual project's dataset acquisition + tool setup. Track 1 builds the muscle;
Track 2 is where it gets applied to real Hubble/JWST data.

---

## Track A — ECE501 Session Deck Exercises

### A1. Sampling / Downsampling
- [ ] Pick one test image to use for all of A1–A2
- [ ] Downsample it by 1/2, 1/4, and 1/8
- [ ] Upscale each downsampled version back to original display size (no interpolation smoothing — just resize for viewing)
- [ ] Visually compare information loss across the three levels
- [ ] Before running: write a one-line prediction of what will happen at 1/8
- [ ] Compare actual result against your prediction

### A2. Interpolation
- [ ] Take the 1/4 or 1/8 downsampled image from A1
- [ ] Resize back to original size using nearest-neighbor
- [ ] Resize back to original size using bilinear
- [ ] Resize back to original size using bicubic
- [ ] Compare the three outputs side by side
- [ ] Write one line confirming: interpolation estimates missing pixels, it does not recover lost information

### A3. Aliasing / Stripe Experiment
- [ ] Generate or obtain a repeating stripe pattern image
- [ ] Sample it at multiple intervals, including the period-8 pattern sampled every 7 pixels
- [ ] Observe and note the false/wider pattern produced by undersampling
- [ ] Record why this happens (relate to sampling rate vs. pattern frequency)

### A4. Quantization
- [ ] Take a grayscale image at 256 levels
- [ ] Requantize to 32 levels
- [ ] Requantize to 8 levels
- [ ] Requantize to 4 levels
- [ ] Requantize to 2 levels
- [ ] Note the level at which visible banding/posterization begins
- [ ] Write one line distinguishing: quantization = intensity precision, sampling = spatial resolution

### A5. Pixel-Neighborhood Task
- [ ] For a chosen pixel, identify its N4 neighbors
- [ ] Identify its ND (diagonal) neighbors
- [ ] Identify its N8 neighbors
- [ ] Write definitions of 4-adjacency, 8-adjacency, and m-adjacency in your own words

### A6. Distance-Measure Hand Calculations
- [ ] Pick two pixel coordinates (x1,y1), (x2,y2)
- [ ] Calculate Euclidean distance by hand: D_E = sqrt((x1-x2)² + (y1-y2)²)
- [ ] Calculate City-block distance by hand: D4 = |x1-x2| + |y1-y2|
- [ ] Calculate Chessboard distance by hand: D8 = max(|x1-x2|, |y1-y2|)
- [ ] Double check all three by hand (deck explicitly requires this, not just code output)

### A7. Main Mini-Challenge — Affine Transformation
- [ ] Pick your own image
- [ ] Choose a rotation angle and translation offset
- [ ] Build the combined 2×3 affine transformation matrix by hand
- [ ] Pick a recognizable landmark point in the image
- [ ] Calculate the landmark's transformed position by hand
- [ ] Apply the transformation using `cv2.warpAffine`
- [ ] Compare actual transformed landmark position vs. your hand calculation
- [ ] Note any discrepancy and why (rounding, interpolation, matrix convention)

### A8. Bonus — Add Scaling
- [ ] Extend the affine matrix with a scale factor, sweeping 1.0 → 1.3
- [ ] Predict how the landmark point will move under scaling
- [ ] Run the transformation and verify your prediction

---

## Track B — Galaxy Classification Project: Dataset Acquisition & Setup

### B1. Environment Setup
- [ ] Set up Python environment (venv or conda) for the project
- [ ] Install astropy
- [ ] Install astroquery
- [ ] Install photutils
- [ ] Install scikit-image
- [ ] Install OpenCV (opencv-python)
- [ ] Install numpy, scipy, matplotlib
- [ ] Verify Jupyter notebook environment works for interactive FITS viewing

### B2. Data Gathering — MAST / Astroquery
- [ ] Confirm `astroquery.mast` query syntax for both Hubble and JWST products
- [ ] Identify 2–3 target objects imaged by both telescopes (e.g., Pillars of Creation, Southern Ring Nebula, Stephan's Quintet, Carina Nebula)
- [ ] Query MAST for Hubble FITS files of chosen target(s)
- [ ] Query MAST for JWST FITS files of the same target(s)
- [ ] Download and locally store both sets of FITS files
- [ ] Record proposal/observation IDs for each downloaded file (for report traceability)

### B3. Data Processing (Initial)
- [ ] Load a Hubble FITS file and inspect header metadata (filter, exposure time, resolution)
- [ ] Load a JWST FITS file and inspect header metadata
- [ ] Extract pixel data array from each FITS file
- [ ] Normalize/scale pixel data for basic display (e.g., zscale or log stretch)
- [ ] Note visible differences in noise, resolution, and dynamic range between the two sources

### B4. Visualization
- [ ] Display raw Hubble image using matplotlib
- [ ] Display raw JWST image using matplotlib
- [ ] Create a side-by-side comparison plot of the same target from both telescopes
- [ ] Save comparison plots for use in the Week 2 report/deliverable

### B5. Galaxy Zoo (Validation Data — Preliminary)
- [ ] Confirm current access method for Galaxy Zoo labeled data
- [ ] Identify whether labels are available for the specific targets chosen in B2
- [ ] Note any gaps (if no overlap with Galaxy Zoo targets, flag for problem-statement adjustment)

---

## Notes
- Track A exercises can be done on any convenient test image (does not need to be astronomical) — the point is building the underlying DIP skills.
- Track B is where those skills (sampling, quantization, pixel neighborhoods, distance measures, affine transforms) will later be applied to real Hubble/JWST FITS data — no need to force the connection yet this week.
- Mathematical/algorithmic method selection (deconvolution, denoising, CAS parameters, etc.) is deferred to later in Week 2 per the project plan — B1–B5 above is data/tooling only.
