# Morphological Classification of Galaxies in HST and JWST Images Using Classical DIP

**ECE 501: Digital Image Processing Course Project**  
**Team:** Pixel Pioneers  

---

## 1. Project Overview & Context

This project investigates the restoration, segmentation, quantitative feature extraction, and morphological classification of distant galaxies from raw spaceborne astronomical imagery captured by the **Hubble Space Telescope (HST)** and the **James Webb Space Telescope (JWST)**.

Modern astronomical computer vision frequently deploys deep neural networks (CNNs, Vision Transformers). However, deep learning models operate as "black boxes" requiring massive labeled training sets, obscures the physical mechanisms of decision boundaries, and does not provide mathematical interpretability. 

This project intentionally adopts a **strictly classical Digital Image Processing (DIP)** paradigm to evaluate how far deterministic, physics-informed, and mathematically rigorous signal processing techniques can go in analyzing deep-sky imaging.

### Core Methodological Boundaries
* **Strictly Classical DIP:** All filtering, deconvolution, segmentation, morphological operators, feature extractors, and classification rules are deterministic.
* **No Machine Learning:** No convolutional networks, random forests, SVMs, or trained embeddings.
* **Pure Image Domain Processing:** No astrophysical/cosmological modeling (no redshift corrections, spectral energy distribution [SED] fitting, or stellar population synthesis). The imagery is treated purely as a multi-stage 2D signal processing challenge.

---

## 2. Project Metadata

| Attribute | Details |
| :--- | :--- |
| **Course** | **ECE 501 — Digital Image Processing** |
| **Academic Term** | Autumn 2026 |
| **Team Name** | **Pixel Pioneers** |
| **Team Members** | • **Aryan Thakkar** (AU2440248)<br>• **Nikkesh Parekh** (AU2440214)<br>• **Hitaksh Thaker** (AU2440300)<br>• **Maitri Parekh** (AU2440008) |
| **Course Faculty** | **Prof. Mehul Raval** |
| **Teaching Assistants** | **Apoorva Mehta**, **Jyoti Triklani**, **Sagar Chavda** |
| **Primary Target Observables** | • **Stephan's Quintet (HCG 92):** Compact interacting group (`NGC 7317`, `NGC 7318A`, `NGC 7318B`, `NGC 7319`, `NGC 7320`)<br>• **Pillars of Creation (M16 / Eagle Nebula):** Extended gas and dust pillar structures |
| **Primary Data Repositories** | • **NASA MAST** (Mikulski Archive for Space Telescopes): Calibrated FITS mosaics<br>• **Galaxy Zoo 2 (GZ2)** & **NASA/IPAC NED**: Consensus ground truth |

---

## 3. Astronomical Imaging Challenges

Space telescope observations present unique signal processing difficulties that the classical DIP pipeline is designed to resolve:

1. **Extreme Dynamic Range:** Luminous galactic nuclei and active cores (AGN) can be orders of magnitude brighter than diffuse outer spiral arms, tidal tails, and interacting bridges. Standard linear display stretches either clip the core to pure white or suppress the outer structure into the background noise floor.
2. **Optical Point Spread Function (PSF) & Diffraction:** Optical systems induce diffraction patterns (e.g., the characteristic hexagonal spikes of JWST's segmented beryllium primary mirror), blurring fine structural details.
3. **Complex Noise & Artifacts:** Observations contain mixed Poisson photon shot noise, detector read noise, cosmic-ray spikes (isolated single-pixel or sub-cluster energy deposits), and non-uniform background gradients across detector mosaics.
4. **Field Crowding & Blended Sources:** Interacting galaxy pairs (e.g., `NGC 7318A` and `NGC 7318B`) exhibit overlapping isophotes, while foreground Milky Way stars can mimic compact galactic cores.

---

## 4. End-to-End Pipeline Architecture

The overall methodology follows a modular, deterministic DIP pipeline:

```
+─────────────────────────────────────────────────────────────+
|               1. INGESTION & EXPLORATORY ANALYSIS           |
|  - NASA MAST FITS mosaic retrieval via astroquery           |
|  - Header extraction, WCS alignment, dynamic range audits   |
+──────────────────────────────┬──────────────────────────────+
                               │
                               ▼
+─────────────────────────────────────────────────────────────+
|            2. CLASSICAL RESTORATION & ENHANCEMENT           |
|  - Deconvolution (Unsupervised Wiener filtering)            |
|  - Edge-preserving denoising (Bilateral filtering)          |
|  - Non-linear dynamic range compression (Asinh stretch)     |
|  - Chromatic composite generation (JWST NIRCam RGB)         |
+──────────────────────────────┬──────────────────────────────+
                               │
                               ▼
+─────────────────────────────────────────────────────────────+
|             3. SEGMENTATION & SOURCE ISOLATION              |
|  - Background modeling & local sky estimation               |
|  - Foreground footprinting & distance-transform watershed   |
|  - Morphological cleanup (opening, closing, hole-filling)   |
|  - Star-galaxy separation via flux concentration criteria   |
|  - Connected-component cutout extraction per source         |
+──────────────────────────────┬──────────────────────────────+
                               │
                               ▼
+─────────────────────────────────────────────────────────────+
|             4. QUANTITATIVE FEATURE EXTRACTION (CAS)        |
|  - Concentration (C): Growth curve ratio 5*log10(r80 / r20) |
|  - Asymmetry (A): 180-deg rotational residual minus sky     |
|  - Smoothness (S): High-frequency clumpiness residual       |
|  - Moment invariants: Hu moments, ellipticity, eccentricity |
+──────────────────────────────┬──────────────────────────────+
                               │
                               ▼
+─────────────────────────────────────────────────────────────+
|              5. RULE-BASED MORPHOLOGICAL CLASSIFICATION     |
|  - Deterministic CAS space boundaries                       |
|    * Elliptical (E) : High C, Low A, Low S                  |
|    * Spiral (S)     : Moderate C, Moderate/High A           |
|    * Irregular (Irr): Elevated A / S, disturbed morphology  |
|  - Benchmarking against Galaxy Zoo 2 / NED consensus labels |
+─────────────────────────────────────────────────────────────+
```

---

## 5. Summary of Current Progress & Milestones

The project has advanced through several major developmental phases:

### Phase 1: Foundational DIP Benchmarking & Data Ingestion
* Implemented baseline digital image processing operations from scratch to establish ground-truth algorithmic intuition (spatial sampling, aliasing/Moiré patterns, uniform intensity quantization, distance transforms, and affine transformations).
* Built automated ingestion scripts using `astroquery.mast` to retrieve calibrated FITS science mosaics for **Stephan's Quintet** (JWST NIRCam: `F090W`, `F150W`, `F200W`; HST ACS: `F814W`) and the **Pillars of Creation** (HST WFPC2/ACS).
* Performed exploratory data analysis (EDA) quantifying background sky non-uniformities, noise standard deviations, and dynamic ranges across spectral bands.

### Phase 2: Restoration & Enhancement Pipeline R&D
* **Deconvolution:** Compared iterative Richardson-Lucy against Unsupervised Wiener deconvolution. Selected **Wiener deconvolution** due to its controlled noise amplification in low-signal regimes and absence of ringing artifacts around high-contrast stellar cores.
* **Edge-Preserving Denoising:** Evaluated Gaussian, Median, Non-Local Means (NLM), Wavelet shrinkage, and Bilateral filtering. Selected **Bilateral filtering** ($d=9, \sigma_c=75, \sigma_s=75$) for its ability to suppress Poisson sky variance while maintaining sharp outer isophotal boundaries.
* **Dynamic Range Compression:** Tested Linear, Logarithmic, Histogram Equalization, CLAHE, and Inverse Hyperbolic Sine (`Asinh`) stretches. Selected **Asinh Stretch** ($a=0.05$), preserving faint tidal features and outer discs without saturating high-intensity galactic cores.
* **Quantitative HST vs. JWST Comparison:** Verified that JWST NIRCam provides significantly superior SNR (85.42 vs. 28.14) and sharpness ($\sigma^2_{\nabla^2} = 312.58$ vs. $104.22$) over HST ACS, establishing JWST as the primary instrument for downstream feature extraction.

### Phase 3: Segmentation & CAS Feature Extraction (In Progress)
* **Segmentation Refinement:** Transitioned from single global/adaptive thresholding to an Otsu foreground footprint combined with distance-transform watershed segmentation to prevent interacting cores (such as `NGC 7318A/B`) from erroneously merging into single macro-components.
* **Morphological Filtering:** Employed elliptical structuring elements for morphological closing, speckle removal via opening, and binary hole filling applied per segmented region.
* **Star–Galaxy Separation:** Established compactness and circular aperture concentration thresholds to filter out point-source foreground stars with diffraction spikes.
* **CAS Parameter Computation:** Formulated deterministic routines for curve-of-growth Concentration ($C$), 180° rotational residual Asymmetry ($A$), spatial low-pass filter Smoothness ($S$), and Hu moment invariants across extracted galaxy cutouts.

---

## 6. Setup & Execution Guide

### Prerequisites
* Python 3.10 or 3.11
* Virtual environment tool (`venv` or `conda`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aryanth29/DIP_2026_21_12.git
   cd DIP_2026_21_12
   ```

2. **Create and activate a virtual environment:**
   * **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```
     *(Alternatively, run the automated setup script: `setup_environment.bat`)*

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Register the Jupyter Kernel and launch:**
   ```bash
   python -m ipykernel install --user --name dip_env --display-name "Python (DIP ECE501)"
   jupyter notebook
   ```
   Open [Code/ECE501.ipynb](file:///Users/aryanthakkar/Desktop/DIP_2026_21_12/Code/ECE501.ipynb) in Jupyter to inspect or execute the pipeline.

---

## 7. Primary Toolchain

| Package | Purpose |
| :--- | :--- |
| **`astropy`** | FITS file I/O, astronomical WCS coordinate handling, and non-linear stretch functions (`AsinhStretch`, `LogStretch`, `ZScaleInterval`) |
| **`astroquery`** | Programmatic querying and retrieval of science products from the NASA MAST archive |
| **`photutils`** | 2D background estimation (`Background2D`) and aperture photometry |
| **`opencv-python`** (`cv2`) | Bilateral filtering, morphological transformations, affine transforms, and connected components |
| **`scikit-image`** (`skimage`)| Richardson-Lucy and Wiener deconvolution, local thresholding, and region properties |
| **`scipy`** | Distance transforms, watershed processing, spatial filters, and numerical routines |
| **`numpy`** | N-dimensional array processing and matrix mathematics |
| **`matplotlib`** | Multi-band composite rendering, feature distribution plots, and diagnostic figures |
| **`pandas`** | Catalog compilation and tabular analysis of extracted morphological parameters |
