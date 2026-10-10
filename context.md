# Project Context: Morphological Classification of Galaxies in HST and JWST Images Using Classical DIP

---

## 1. Executive Summary & Overview

This project tackles the restoration, segmentation, quantitative feature extraction, and morphological classification of galaxies from raw **Hubble Space Telescope (HST)** and **James Webb Space Telescope (JWST)** astronomical imagery using **classical, deterministic Digital Image Processing (DIP)** algorithms. 

While modern computer vision approaches often resort to deep neural networks, this project intentionally imposes a strict constraint: **no machine learning, deep learning, or trained statistical classifiers**, and **no astronomical/cosmological spectral modeling**. Instead, the goal is to demonstrate the power, mathematical rigor, and physical interpretability of classical DIP techniques—including PSF modeling and deconvolution, adaptive spatial filtering, dynamic range compression, connected-component analysis, and the classical **Concentration-Asymmetry-Smoothness (CAS)** morphological framework.

---

## 2. Project Metadata

| Item | Details |
| :--- | :--- |
| **Course** | **ECE 501 — Digital Image Processing** |
| **Project Title** | *Morphological Classification of Galaxies in HST and JWST Images Using Classical DIP* |
| **Team Members** | **Aryan Thakkar**, **Nikkesh Parekh**, **Hitaksh Thaker**, **Maitri Parekh** |
| **Course Faculty** | **Prof. Mehul Raval** |
| **Teaching Assistants** | **Apoorva Mehta**, **Jyoti Triklani**, **Sagar Chavda** |
| **Primary Target Objects** | **Stephan's Quintet (HCG 92)**, **Pillars of Creation (M16 / Eagle Nebula)** |
| **Data Repositories** | **NASA Mikulski Archive for Space Telescopes (MAST)**, **Galaxy Zoo** |

---

## 3. Problem Statement & Motivation

Spaceborne observatories such as the Hubble Space Telescope (operating primarily in optical/UV) and the James Webb Space Telescope (operating in the infrared) capture rich, deeply detailed views of distant galaxies. However, raw space telescope data presents severe signal processing challenges:

1. **Extreme Dynamic Range:** Bright active galactic nuclei (AGN) and dense bulges are orders of magnitude brighter than the faint, diffuse outer galactic discs, tidal tails, and interacting debris. Standard display stretches either saturate the core or lose the outer structure entirely.
2. **Optical Blurring & Diffraction:** Images suffer from instrument-specific Point Spread Function (PSF) convolution, causing blurring and prominent diffraction spikes around bright point sources (stars).
3. **Complex Noise & Artifacts:** Images are degraded by photon (Poisson) noise, detector read noise, cosmic-ray hits (isolated sharp high-intensity spikes), and non-uniform background gradients.
4. **Field Crowding & Object Separability:** Galaxy fields lack pre-defined segmentation boundaries; interacting galaxies (such as NGC 7318A and NGC 7318B in Stephan's Quintet) have overlapping light distributions, while foreground stars mimic galactic nuclei.
5. **The Black-Box Limitation of Modern ML:** State-of-the-art machine learning models require tens of thousands of manually labeled images, obscure the physical mechanisms of decision boundaries, and fail to explain *why* an object is categorized into a given morphological class.

### Core Philosophy & Strict Methodological Constraints
- **Strictly Classical DIP:** All filtering, restoration, segmentation, measurement, and classification must be deterministic and rule-based.
- **No Machine Learning:** No CNNs, random forests, SVMs, or trained embeddings.
- **Pure Image Domain Analysis:** No astrophysics domain modeling (no cosmological redshift corrections, stellar population synthesis, or spectroscopic SED fitting). The challenge is treated purely as a multi-stage 2D image processing problem.

---

## 4. What We Are Trying to Achieve (Core Objectives)

1. **End-to-End Image Restoration & Enhancement Pipeline:**
   - Model the optical degradation and reverse PSF blur using classical deconvolution without amplifying noise.
   - Suppress high-frequency noise and detector variance while rigorously preserving steep galaxy intensity edges.
   - Compress the extreme dynamic range using astronomical non-linear stretch functions to reveal both faint halos and nuclear detail.
   - Clean detector seams, hot pixels, and cosmic ray spikes deterministically.
   - Construct calibrated, morphologically informative multi-band RGB composites.

2. **Empirical Instrument Comparison (HST vs. JWST):**
   - Provide an objective, quantitative comparison of identical target crops from Hubble (ACS/WFC) and JWST (NIRCam) across Signal-to-Noise Ratio (SNR), sharpness (Laplacian variance $\sigma^2_{\nabla^2}$), usable dynamic range, and spatial resolution.
   - Formally select the superior primary instrument for downstream morphological feature extraction.

3. **Deterministic Galaxy Segmentation & Source Isolation:**
   - Differentiate point-source stars (with characteristic diffraction spikes) from extended galaxies using compactness and concentration criteria.
   - Perform adaptive local thresholding, morphological opening/closing, and connected component labeling to extract clean binary masks for individual galaxies.

4. **Quantitative Morphological Feature Extraction (The CAS System):**
   - Implement the astronomically established **CAS framework** using classical pixel operations:
     - **Concentration ($C$):** Logarithmic ratio of radii enclosing $80\%$ and $20\%$ of total galactic curve of growth ($C = 5 \log_{10}(r_{80} / r_{20})$).
     - **Asymmetry ($A$):** Residual flux computed by rotating the galaxy $180^\circ$ about its photometric center, normalized by original flux and subtracted by background noise asymmetry.
     - **Smoothness / Clumpiness ($S$):** High-frequency spatial variance obtained by subtracting a boxcar-smoothed version from the original galaxy.
     - **Geometric Shape Moments:** Moments of inertia, eccentricity, ellipticity, and Hu's moment invariants.

5. **Interpretable Rule-Based Galaxy Classification:**
   - Categorize detected galaxies into standard Hubble morphological types:
     - **Elliptical (E):** High concentration ($C$), low asymmetry ($A$), low clumpiness ($S$).
     - **Spiral (S):** Moderate concentration, higher asymmetry and clumpiness due to star-forming arms.
     - **Irregular / Interacting (Irr):** Low concentration, high asymmetry ($A$), irregular structural borders.

6. **Validation Against Ground Truth:**
   - Benchmark the classical rule-based classifications against consensus classifications from the **Galaxy Zoo** citizen science catalog and published astronomical literature (e.g., Hickson Compact Group 92 catalogs).

---

## 5. End-to-End Technical Pipeline

```
+---------------------------------------------------------------------------------+
|                                 1. INPUT DATA                                   |
|   - MAST FITS Files: JWST NIRCam (F090W, F150W, F200W) & HST ACS/WFC (F814W)   |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                      2. PREPROCESSING & ARTIFACT REJECTION                      |
|   - FITS HDU extraction, pixel calibration, bad pixel masking                   |
|   - Cosmic-ray rejection via 5x5 median filter & 3-sigma difference threshold   |
|   - photutils Background2D estimation (50x50 mesh) for sky background flattening |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                    3. RESTORATION & DYNAMIC RANGE ENHANCEMENT                   |
|   - Deconvolution: Unsupervised Wiener deconvolution with Gaussian PSF          |
|   - Denoising: Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)             |
|   - Dynamic Range Stretch: Asinh Stretch (a = 0.05) with 99.5% percentile clip  |
|   - Multi-channel RGB compositing (R=F200W, G=F150W, B=F090W physical mapping)   |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                    4. QUANTITATIVE INSTRUMENT EVALUATION                        |
|   - Compute background SNR, Laplacian sharpness variance, dynamic range (dB)    |
|   - Justify JWST NIRCam as the primary dataset over HST                         |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                        5. SEGMENTATION & SOURCE ISOLATION                       |
|   - Adaptive multi-level thresholding (Otsu / local median)                     |
|   - Morphological cleanup (erosion, dilation, hole filling)                     |
|   - Star-galaxy separation via concentration/compactness thresholding           |
|   - Connected component labeling & bounding-box / cutout extraction             |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                     6. QUANTITATIVE FEATURE EXTRACTION (CAS)                    |
|   - Concentration (C): Growth curve flux ratios (r80 / r20)                     |
|   - Asymmetry (A): 180-degree rotational residual minus background noise         |
|   - Smoothness (S): Residual after spatial low-pass filter subtraction           |
|   - Geometric moments: Ellipticity, area, perimeter, centroid                   |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                     7. RULE-BASED MORPHOLOGICAL CLASSIFICATION                  |
|   - Decision tree boundaries in CAS space:                                      |
|     * If C > C_thresh and A < A_thresh and S < S_thresh  --> Elliptical (E)     |
|     * If C <= C_thresh and A >= A_thresh                 --> Spiral (S)         |
|     * Otherwise (high A, structural disruption)          --> Irregular (Irr)    |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                            8. GROUND TRUTH BENCHMARKING                         |
|   - Comparison against Galaxy Zoo consensus morphological labels                |
|   - Calculation of confusion matrix, precision, recall, and failure case analysis|
+---------------------------------------------------------------------------------+
```

---

## 6. Project Chronology & Milestone Progress

### Week 1 — Problem Statement & Scoping
- Established the core DIP-only philosophy and formal non-ML boundary conditions.
- Defined target observables (Hubble vs. JWST) and identified NASA MAST as the FITS data provider and Galaxy Zoo for validation.
- Formulated the project roadmap and literature survey strategy.
- Delivered: [DIP_Report_week_1.tex]

### Week 2 — Environment Setup, EDA & Core DIP Exercises
- Configured Python environment (`astropy`, `astroquery`, `photutils`, `scipy`, `skimage`, `opencv-python`).
- Executed foundational ECE 501 laboratory exercises in sampling, interpolation, aliasing, quantization, neighborhood metrics, distance hand calculations, and affine coordinate transformations.
- Queried and downloaded calibrated FITS datasets from MAST for **Stephan's Quintet (HCG 92)** and **Pillars of Creation (M16)**.
- Performed initial Exploratory Data Analysis (EDA) on FITS header metadata and dynamic ranges.
- Delivered: [week2-todo.md][DIP_Report_week_2.tex]

### Week 3 — Restoration & Enhancement Pipeline R&D (Completed)
- Executed rigorous quantitative comparative experiments across classical DIP restoration techniques:
  - **Deconvolution (D1):** Richardson-Lucy (5, 10, 20, 50 iterations) vs. Unsupervised Wiener deconvolution. Selected **Wiener deconvolution** due to bounded noise amplification and avoidance of ringing around stellar cores.
  - **Denoising (D2):** Compared NLM, Gaussian, Median, Wavelet shrinkage, and Bilateral filtering. Selected **Bilateral filter** ($d=9, \sigma_c=75, \sigma_s=75$) for flattening sky variance while maintaining sharp galactic edge gradients.
  - **Contrast Stretch (D3):** Compared Linear, Log, CLAHE, Histogram Equalization, and Asinh. Selected **Asinh Stretch** ($a=0.05$) for optimal simultaneous visibility of bright galactic cores and faint tidal arms.
  - **RGB Compositing (D4):** Combined JWST NIRCam filters (F090W, F150W, F200W) using standard physical mapping (Red=F200W, Green=F150W, Blue=F090W).
  - **HST vs. JWST Comparison (D5):** Measured SNR, Laplacian variance sharpness, and dynamic range on matched crops. Confirmed **JWST NIRCam** as the superior primary dataset.
  - **Artifact Rejection (D6):** Implemented $5 \times 5$ median filtering with $3\sigma$ difference map thresholding for cosmic-ray cleaning.
  - **Parameter Tuning (D7):** Optimized `Background2D` box size to $50$ px and NLM smoothing to $h=10$.

### Week 4 & Beyond — Segmentation, CAS Extraction & Classification (Upcoming)
- Implement adaptive thresholding and morphological operators to isolate galaxy segments.
- Implement star-galaxy separation using concentration and compactness.
- Compute CAS indices ($C, A, S$) and shape moments for each segmented object.
- Apply rule-based decision trees to classify galaxies into Elliptical, Spiral, and Irregular.
- Benchmark results against Galaxy Zoo reference classifications for Stephan's Quintet (NGC 7317, NGC 7318A, NGC 7318B, NGC 7319, NGC 7320) and report overall classification accuracy and failure modes.

---

## 7. Workspace Architecture & Directory Guide

```
DIP Project/
├── context.md                          # This overarching project context document
├── week2-todo.md                       # Week 2 sprint tasks and ECE501 exercises
├── week3-todo.md                       # Week 3 finished R&D sprint tasks and decisions
│
├── Code/
│   └── ECE501.ipynb                    # Primary notebook (Tracks A-F: EDA, processing, segmentation, CAS)
│
├── Weekly Reports/
│   ├── DIP_Report_week_1.tex / .pdf    # Week 1 Progress Report
│   ├── DIP_Report_week_2.tex / .pdf    # Week 2 Progress Report
│   ├── DIP_Report_week_3.tex / .pdf    # Week 3 Progress Report
│   └── DIP_Week3_Report_v2.docx        # Word format submission for Week 3
│
├── Output/
│   └── week_3_output/                  # Figures and evaluation outputs
│       ├── classification_map_jwst.png
│       ├── classification_map_hst.png
│       ├── galaxy_classifications_jwst.csv
│       ├── galaxy_classifications_hst.csv
│       ├── comparison_hubble_vs_jwst_stephans_quintet.png
│       ├── hst_vs_jwst_comparison.png
│       ├── stephans_quintet_jwst_rgb.png
│       ├── robustness_sweep.png
│       └── pillars_of_creation_m16_hubble.png
│
└── mastDownload/                       # Raw and calibrated FITS science files
    ├── JWST/                           # NIRCam F090W, F150W, F200W calibrated FITS
    └── HST/                            # ACS/WFC calibrated FITS
```

---

## 8. Software Environment & Dependencies

The project is built on Python 3 with the following core toolchain:

| Package | Purpose in Project |
| :--- | :--- |
| **`astropy`** | FITS file I/O, World Coordinate System (WCS), astronomical stretch functions (`AsinhStretch`, `LogStretch`, `ZScaleInterval`) |
| **`astroquery`** | Automated programmatic querying and acquisition from NASA MAST |
| **`photutils`** | 2D background modeling (`Background2D`), source detection, and aperture photometry |
| **`opencv-python`** (`cv2`) | Bilateral filtering, morphological transformations, affine transforms, connected components |
| **`scikit-image`** (`skimage`)| Richardson-Lucy and Wiener deconvolution, CLAHE, wavelet shrinkage, region measurements |
| **`scipy`** | Multi-dimensional Gaussian/median spatial filtering, numerical minimization |
| **`numpy`** | N-dimensional array processing, matrix algebra, hand-calculated image metrics |
| **`matplotlib`** | Multi-panel comparison figures, RGB visualization, parameter histograms |
| **`pandas`** | Tabular collation of extracted CAS parameters and classification catalogs |
