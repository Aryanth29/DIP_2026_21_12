# Week 3 To-Do — Restoration & Enhancement Pipeline R&D

These extend your existing restoration/enhancement pipeline with deeper experimentation and 
quantitative rigor. No segmentation, CAS parameters, or morphology work included — pure DIP R&D.

---

## Track D — Restoration & Enhancement Pipeline R&D (Pure DIP, No Classification)

### D1. Deconvolution Experiments
- [x] Implement Richardson-Lucy deconvolution on the JWST/HST Stephan's Quintet images
- [x] Obtain or approximate the PSF (point spread function) — use `astropy`/`webbpsf` if available, or approximate with a Gaussian/Airy disk kernel if not
- [x] Run deconvolution with varying iteration counts (5, 10, 20, 50) and compare sharpness vs. noise amplification at each
- [x] Implement Wiener deconvolution as a second method, compare against Richardson-Lucy
- [x] Quantify results: measure edge sharpness (e.g., variance of Laplacian) before vs. after deconvolution for both methods
- [x] Document: which method gives better results on this data, and why (noise amplification tradeoff, ringing artifacts, etc.)

### D2. Denoising Method Comparison
- [x] Apply and compare multiple denoising techniques on the same preprocessed image:
  - Non-local means (`cv2.fastNlMeansDenoising`)
  - Gaussian filtering (`scipy.ndimage.gaussian_filter`)
  - Median filtering (`scipy.ndimage.median_filter`)
  - Wavelet shrinkage (`skimage.restoration.denoise_wavelet`)
  - Bilateral filtering (`cv2.bilateralFilter`)
- [x] Quantitatively compare: compute SNR or noise variance in a flat background region before/after each method
- [x] Visually compare: does denoising blur out faint galaxy structure? Pick the best tradeoff method
- [x] Document findings: which method preserves galaxy detail best while reducing background noise

### D3. Contrast Enhancement Deep-Dive
- [x] Compare multiple contrast/stretch methods on the same image:
  - Linear stretch
  - Log stretch
  - Asinh stretch (`astropy.visualization.AsinhStretch`)
  - CLAHE (`skimage.exposure.equalize_adapthist`)
  - Histogram equalization (global)
- [x] Generate a comparison grid (5 panels) showing each method's output side by side
- [x] Discuss which method best reveals faint structure (e.g., tidal tails, faint arms) without saturating bright cores
- [x] Pick a final recommended stretch method for the rest of the project and justify the choice

### D4. Multi-Filter RGB Composite Refinement
- [x] If not already done: complete full RGB composite for Stephan's Quintet using 3 JWST filters (F090W, F150W, F200W)
- [x] Experiment with different filter-to-channel mappings (e.g., swap which filter maps to R/G/B) and observe how color balance changes
- [x] Apply per-channel background subtraction before compositing (if not already done) and compare to compositing without it
- [x] Try combining more than 3 filters using a weighted average approach for a "true-color-like" composite (optional, advanced)
- [x] Save final composite with different normalization choices (ZScale vs. percentile vs. Asinh) and pick the best for visual clarity

### D5. HST vs JWST Image Quality — Quantitative Comparison (NOT classification)
*This is comparative image quality analysis only — no galaxy detection or classification involved.*

- [x] Measure and compare basic image quality metrics between HST and JWST versions of the same target:
  - Signal-to-noise ratio (SNR) in a flat background patch
  - Sharpness (variance of Laplacian) on a common bright feature
  - Dynamic range (max/min usable pixel values after calibration)
  - Spatial resolution comparison (if pixel scale metadata available in FITS headers)
- [x] Create a summary table: HST vs JWST across these metrics
- [x] Visual comparison: crop the same small region from both telescopes' images at matched scale and compare detail visibility side by side
- [x] Write a short paragraph on which telescope's raw data is better suited for downstream processing, purely from an image-quality standpoint (not classification accuracy)

### D6. Artifact Handling
- [x] Identify and document any visible artifacts in your current images:
  - Cosmic ray hits (sharp, isolated bright pixels)
  - Hot/dead pixels
  - Diffraction spikes from bright stars
  - Detector seams or striping (if any remain)
- [x] Implement a simple cosmic-ray rejection method (e.g., median filtering with threshold, or `astropy`'s `astroscrappy`/`ccdproc` cosmic ray detection if available)
- [x] Before/after comparison: show the image with artifacts removed vs. original
- [x] Note any remaining artifacts that couldn't be cleanly removed and why

### D7. Parameter Sensitivity / Robustness Check
- [x] For your background subtraction step (`photutils.Background2D`), test different box sizes (e.g., 25, 50, 100) and observe effect on background estimate quality
- [x] For denoising, test different strength parameters (e.g., `h` value in NLM) and find the point where detail starts getting lost
- [x] Document a "recommended parameter set" based on these experiments, to be used consistently going forward in the project

### D8. Documentation & Intermediate Report Section
- [x] Write a short technical section (1 page) titled "Restoration and Enhancement Pipeline: Method Comparison and Parameter Selection" summarizing D1–D7
- [x] Include 2–3 key comparison figures (deconvolution methods, denoising methods, HST vs JWST quality table)
- [x] State the final pipeline configuration chosen (which deconvolution, which denoising, which stretch method, which parameters) that will be carried forward into segmentation/classification work later

---

## Why These Tasks Matter
This set turns your pipeline from "ran one method and moved on" into a **justified, comparative methodology** — which is exactly what a DIP course evaluator wants to see at a mid-project checkpoint: evidence that you tested alternatives, measured results quantitatively, and made deliberate choices rather than arbitrary ones. It also directly feeds your final report's Methodology section, since D1–D7 produce the comparison figures and justification text you'll need there anyway.

---

## What's Still Deliberately Excluded
No segmentation, no CAS parameters, no morphology classification, no Galaxy Zoo work — all of that remains blocked until you're ready to move into that phase, per your instruction.

---

## Notes
- As deferred from Week 2, Track D resolves the mathematical and algorithmic method selections through comparative experimentation before moving into detection/classification.
- **Final Pipeline Decisions Selected in Week 3:**
  - **Deconvolution:** Unsupervised Wiener deconvolution (`skimage.restoration.unsupervised_wiener`) with $7 \times 7$ Gaussian PSF ($\sigma=1.5$), providing optimal noise-sharpness trade-off without the severe ringing and noise amplification seen in 50-iteration Richardson-Lucy.
  - **Denoising:** Bilateral filter (`cv2.bilateralFilter`, $d=9, \sigma_{\text{color}}=75, \sigma_{\text{space}}=75$), uniquely preserving boundary edges while flattening flat-sky background variance.
  - **Stretch:** Asinh Stretch (`astropy.visualization.AsinhStretch(a=0.05)`), preserving high dynamic range from bright AGN nuclei to faint tidal tails.
  - **Composite:** Physical mapping (R=F200W, G=F150W, B=F090W) with per-channel 99.5th percentile clipping and Asinh stretch.
  - **Dataset Selection:** JWST NIRCam chosen as primary instrument over HST ACS/WFC due to superior SNR and spatial resolution.
  - **Artifact Rejection:** $5 \times 5$ median filtering with $3\sigma_{\text{diff}}$ thresholding and local median substitution.
  - **Background Subtraction:** `photutils.Background2D` with box size 50 (balanced between over-fitting at 25 and under-fitting gradients at 100).
- Handover to Week 4: Image restoration/enhancement foundation is complete; Week 4 proceeds with segmentation (adaptive thresholding, star-galaxy separation), CAS feature extraction, and morphological classification.
