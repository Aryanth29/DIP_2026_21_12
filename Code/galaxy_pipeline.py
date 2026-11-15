"""
galaxy_pipeline.py -- generalised classical-DIP galaxy morphology pipeline
==========================================================================

Runs the project's segmentation -> star/galaxy separation -> CAS +
granulometry -> rule-based E/S/Irr classification chain on ANY number of
HST / JWST FITS images, with nothing hard-coded to a particular target.

100 % classical image processing: sigma-clipped background modelling,
thresholding, watershed, mathematical morphology, moments, order
statistics and fixed numeric decision rules.  No machine learning, no deep
learning, no trained model of any kind anywhere in this file.

How generality is achieved
--------------------------
* Discovery  -- every *.fits under the data directory is inspected; the
  telescope, instrument, detector, filter, product type, WCS footprint and
  pixel scale are read from the file's OWN headers (no obs-id look-up
  tables, no per-target path maps).
* Fields     -- images are grouped into sky "fields" by WCS proximity, so
  several images of the same galaxy group (e.g. HST + 3 JWST frames of
  Stephan's Quintet) are recognised as one field automatically.
* Bands      -- images of one field taken with the same instrument on the
  same pixel grid (e.g. HAP drizzled F439W/F555W/F814W products) are
  stacked automatically: detection on the reddest band, colour composite
  for display.
* Parameters -- frame-dependent parameters (background box, no-data buffer,
  cosmic-ray rejection) are derived from the data / product type.
  Instrument-specific overrides live in a JSON config, keyed by
  INSTRUMENT/DETECTOR, not by target name.
* Names      -- detections are given generic labels; any catalogue CSV in
  the catalogue directory (galaxy_id, ra, dec[, morph_gt]) whose members
  fall inside an image is cross-matched by WCS, and its morph_gt column
  becomes ground truth for the accuracy benchmark -- for any target.

The algorithms themselves (background_relative_watershed,
clean_labeled_mask, separate_stars_galaxies, compute_* CAS functions,
derive_thresholds / classify_cas_row_v2) are the notebook's validated
earlier stage implementations, moved here unchanged except where noted.

development stage (audit against Hickson 1989 morphologies: earlier stage scored 7/30)
---------------------------------------------------------------------
* every image is block-averaged to a common 0.2"/px scale (section 7b);
* morphology (C, A, S, Gini, M20) is measured on LINEAR sky-subtracted flux
  with standard Petrosian-aperture definitions (measure_morphology) instead
  of the 8-bit display image; the asymmetry background term is normalised by
  the galaxy flux;
* detection on linear data: fixed significance + two-tier deblending with
  nucleus-based fragment re-attachment (detect_linear), core-ratio star test
  (point_source_test), chip-gap bridging by Telea inpainting, flat sky for
  drizzled products, L.A.Cosmic for single-exposure drizzles, WCS
  registration by offset voting (refine_wcs_shift);
* classification thresholds are fitted on catalogue galaxies with published
  types and reported with leave-one-group-out cross-validation
  (classify_all); the earlier stage median calibration is kept as
  classify_all_median_calibration. work_scale_arcsec=None restores the native-resolution
  earlier stage detection path.
"""
from __future__ import annotations

import fnmatch
import glob
import json
import math
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import rotate as ndimage_rotate
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import Background2D, MedianBackground

warnings.simplefilter('ignore', FITSFixedWarning)

# ════════════════════════════════════════════════════════════════════════
# 0. DEFAULT PARAMETERS (all overridable from pipeline_config.json)
# ════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    # background / detection (earlier stage values)
    'bg_box': 150,                 # Background2D box size (px), capped for small frames
    'bg_min_boxes_per_axis': 8,    # cap: box <= min(H, W) / this
    'bg_box_min': 32,
    'smooth_sigma': 2.5,
    'min_distance': 20,
    'h_sigma_mult': 3.0,
    'fg_frac_lo': 0.003,
    'fg_frac_hi': 0.02,
    'k_hi': 10.0,
    'k_lo': 1.0,
    'k_step': 0.2,
    'k_max': 80.0,
    # morphology / star-galaxy separation
    'morph_kernel': 5,
    'min_area_px': 25,
    'psf_radius_px': 5,
    'conc_thresh': 0.5,
    'area_max_px': 1500,
    # preprocessing
    'nlm_h': 10,
    'cr_reject': 'auto',           # auto = on for single, non-CR-combined exposures
    'weight_frac_thresh': 0.3,     # WHT < frac * modal weight -> no-data
    'weight_mask_max_frac': 0.70,  # safety ceiling for the weight mask
    'nodata_buffer_px': 'auto',    # auto = bg_box // 3 when no-data covers > trigger
    'nodata_buffer_trigger_frac': 0.05,
    'nodata_buffer_max_frac': 0.90,
    'reject_edge_objects': True,   # drop galaxies truncated by frame/no-data edge
    'edge_flux_frac': 0.05,        # ... i.e. > this share of its linear flux within 10 px of an edge
    # large-galaxy support
    'sky_model': 'auto',           # 'auto' | 'local' | 'coarse' (see select_sky_model)
    'coarse_box_frac': 0.25,       # coarse sky box = this fraction of the frame's short side
    'sky_select_tol': 0.05,        # coarse wins unless sky scatter is > 5 % worse
    'grow_segments': True,         # hysteresis region growing of galaxy seeds
    'grow_k': 3.0,                 # low threshold, in sky sigmas of the smoothed map
    'grow_smooth_sigma': 5.0,      # px, smoothing before growing (2x detection smoothing)
    'merge_contrast': 0.40,        # dynamics merge threshold (HCG 87: fragments <= 0.30, distinct >= 0.56)
    # development stage: common working resolution + physical sizes
    'work_scale_arcsec': 0.2,      # block-average every image to ~this pixel scale (None = native)
    'flat_select_tol': 0.25,       # drizzled products: flat sky unless > 25 % worse than coarse
    'nodata_buffer_arcsec': 2.5,   # = the earlier stage 50 px at 0.05"/px
    'edge_px_arcsec': 0.5,         # = the earlier stage 10 px at 0.05"/px
    'sky_box_px': 32,              # empty-sky box for the A / S background terms
    'wcs_search_arcsec': 30.0,     # registration: catalogue member -> nearest galaxy nucleus
    'wcs_agree_arcsec': 3.0,       # all member offsets within this of their median
    'wcs_min_shift_arcsec': 1.5,   # smaller shifts are left alone
    'wcs_min_area_arcsec2': 20.0,
    'wcs_peak_frac': 0.25,         # a nucleus is a peak >= this fraction of its segment's maximum  # only galaxies at least this large are used as anchors
    'cr_method': 'lacosmic',       # 'lacosmic' (van Dokkum 2001) | 'median' (earlier stage crude 5x5 clip)
    'k_detect': 3.0,               # linear detection: k x sigma of the smoothed sky (fixed significance)
    'detect_smooth_px': 1.0,       # Gaussian sigma of the detection filter (working-grid px)
    'deblend_nlevels': 32,
    'deblend_contrast': 0.005,     # SExtractor DEBLEND_MINCONT default (small blends)
    'deblend_contrast_big': 0.05,  # for parents >= deblend_big_arcsec2 (resolved galaxies)
    'deblend_big_arcsec2': 40.0,
    'deblend_big_smooth_arcsec': 1.5,  # large parents: deblend on this Gaussian-smoothed image
    'max_missing_frac': 0.25,      # > this share of the 1.5 r_p aperture unrecoverably off the data -> unmeasurable
    # final configuration: > this share of the 1.5 r_p aperture CUT OFF by the frame / no-data edge -> unmeasurable,
    # even when symmetric filling recovers it (None = off). 0.15 from a label-free simulation: beyond it the
    # 75th-percentile bias in A exceeds 0.03 (preregistration.md, amendment A1)
    'max_clipped_frac': None,
    # aperture/disc experiments: instead of dropping a galaxy whose aperture is clipped > max_clipped_frac, measure A inside the
    # largest concentric circle (>= recover_min_rp x r_p) clipped by <= max_clipped_frac; objects whose FULL
    # aperture is clipped by more than max_clipped_frac_recover are still unmeasurable (None = no cap)
    'recover_aperture': False,
    'recover_min_rp': 0.5,
    'max_clipped_frac_recover': None,
    'min_sn_per_pixel': 2.5,       # Lotz et al. (2004) reliability limit for G / M20 / CAS
    'star_core_ratio': 0.7,        # F(<psf_radius) / F(<star_outer_px) above this -> point source (HCG 42: stars 0.80-0.91, galaxies <= 0.58)
    'star_outer_px': 5,            # 1" at 0.2"/px
    'gap_bridge_arcsec': 6.0,      # no-data thinner than ~2x this (chip gaps) is bridged for detection
    'gap_object_max_frac': 0.5,    # objects with more of their area on the bridged gap are artefacts
    'gap_merge_frac': 0.5,         # deblended neighbours whose boundary is mostly on the gap are re-joined
    'star_max_area_arcsec2': 400.0,
    'grow_smooth_arcsec': 0.25,    # = grow_smooth_sigma 5 px at 0.05"/px   (used when binned)
    'smooth_arcsec': 0.125,        # = smooth_sigma 2.5 px
    'min_distance_arcsec': 1.0,    # = min_distance 20 px
    'psf_radius_arcsec': 0.25,     # = psf_radius_px 5 px
    'morph_kernel_arcsec': 0.25,   # = morph_kernel 5 px
    # neighbour-model run: subtract smooth symmetric models of all OTHER objects before
    # measuring each galaxy (build_neighbour_models); measure_legacy also
    # stores the development stage measurement (no model subtraction) as *_v12 columns
    'neighbour_model': True,
    'nb_model_extent': 4.0,        # model out to this many segment semi-major axes
    'measure_legacy': False,
    # baseline run: save the inputs of measure_morphology (linear image, segmentation,
    # no-data mask, sky terms, catalogue matches) to <job out_dir>/_measure_cache.npz,
    # so new morphology features can be tested on the labelled galaxies without
    # re-running detection / segmentation (~0.5 GB for all 42 jobs)
    'save_measure_cache': False,
    # final configuration: in a multi-band job, ALSO measure every object on these bands' own linear
    # images with the same segmentation (columns C_<band>, A_<band>, S_<band>, measurable_<band>),
    # e.g. ["F606W"] for the label-free F606W -> F814W calibration (preregistration.md)
    'measure_bands': [],
    # cutouts / CAS
    'cutout_pad': 30,
    'cas_smooth_sigma_frac': 0.25,
    'assumed_fwhm_px': 2.355 * 1.5,
    # catalogue matching
    'catalog_match_radius_arcsec': 10.0,
    # field grouping
    'field_link_arcmin': 4.0,
    'band_align_tol_px': 1.0,      # frame-centre agreement; corners also allow 0.1 % plate-scale drift
    # diagnostics / output
    'print_obj_diagnostics': False,
    'show_figures': False,
    'gallery_max_cols': 6,
    'gallery_max_objects': 48,
}

MIN_N_FOR_AUTO_CALIBRATION = 20   # H1 (earlier stage) -- unchanged

COMBINED_PRODUCTS = {'drz', 'drc', 'i2d'}           # multi-exposure, CR-combined
SINGLE_EXPOSURE_PRODUCTS = {'flt', 'flc', 'c0m', 'c0f', 'cal', 'raw'}  # may still hold CRs


def _r(v, nd):
    """round() that passes NaN / None through."""
    try:
        return round(float(v), nd) if v is not None and np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _slug(text):
    s = re.sub(r'[^0-9a-zA-Z]+', '_', str(text)).strip('_').lower()
    return s or 'unnamed'


def _norm(text):
    return re.sub(r'[^0-9a-z]', '', str(text).lower())


# ════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ════════════════════════════════════════════════════════════════════════
def load_config(path=None):
    """Read pipeline_config.json (optional). Keys that are not present fall
    back to DEFAULTS. Returns a plain dict."""
    cfg = {'defaults': dict(DEFAULTS), 'exclude': [], 'field_aliases': [],
           'instrument_overrides': {}, 'field_overrides': {},
           'data_dir': 'mastDownload', 'output_dir': None, 'catalog_dir': 'catalogs'}
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            user = json.load(f)
        user = {k: v for k, v in user.items() if not k.startswith('_')}
        cfg['defaults'].update(user.pop('defaults', {}) or {})
        cfg.update(user)
        base = os.path.dirname(os.path.abspath(path))
        for key in ('data_dir', 'output_dir', 'catalog_dir'):
            if cfg.get(key) and not os.path.isabs(cfg[key]):
                cfg[key] = os.path.normpath(os.path.join(base, cfg[key]))
    return cfg


def params_for(job, cfg):
    """Effective parameters for one image job: DEFAULTS <- config defaults
    <- instrument overrides (matched on 'INSTRUME' or 'INSTRUME/DETECTOR')
    <- field overrides (matched on field name)."""
    p = dict(cfg['defaults'])
    inst_keys = [job.instrument, f'{job.instrument}/{job.detector}']
    for pattern, ov in (cfg.get('instrument_overrides') or {}).items():
        if any(fnmatch.fnmatch(_norm(k), _norm(pattern).replace('_', '*')) or
               _norm(k) == _norm(pattern) for k in inst_keys if k):
            p.update({k: v for k, v in ov.items() if not k.startswith('_')})
    for pattern, ov in (cfg.get('field_overrides') or {}).items():
        if fnmatch.fnmatch(job.field_name.lower(), pattern.lower()):
            p.update({k: v for k, v in ov.items() if not k.startswith('_')})
    return p


# ════════════════════════════════════════════════════════════════════════
# 2. DISCOVERY -- everything read from the FITS headers themselves
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ImageRecord:
    path: str
    telescope: str
    instrument: str
    detector: str
    filter: str
    wavelength_um: float
    targname: str
    product: str
    sci_ext: int
    shape: tuple
    n_sci: int
    has_dq: bool
    has_wht: bool
    ra_c: float
    dec_c: float
    pix_scale_arcsec: float
    corners: np.ndarray = field(repr=False, default=None)
    field_name: str = ''
    status: str = 'ok'
    n_exp: int = 0          # development stage: distinct input exposures (HDRTAB rootnames; 0 = unknown)
    exptime: float = 0.0


def _product_type(path):
    m = re.search(r'_([a-z0-9]+)\.fits(\.gz)?$', os.path.basename(path).lower())
    return m.group(1) if m else 'unknown'


def _pick_filter(h0, h1):
    pat = re.compile(r'^F\d{2,4}[A-Z]{1,3}$')
    for h in (h0, h1):
        for key in ('FILTER', 'FILTNAM1', 'FILTNAM2', 'FILTER1', 'FILTER2', 'PUPIL'):
            v = h.get(key)
            if isinstance(v, str) and pat.match(v.strip().upper()):
                return v.strip().upper()
    return 'UNKNOWN'


def _filter_wavelength_um(filt, telescope, detector):
    """Pivot-ish wavelength from the filter name -- only used to ORDER bands
    (reddest = detection band, RGB channel order). HST UVIS/optical names
    are in nm (F606W = 606 nm); HST IR and JWST names are in 10 nm units
    (F160W = 1.60 um, F200W = 2.00 um)."""
    m = re.match(r'^F(\d{2,4})', filt or '')
    if not m:
        return float('nan')
    n = int(m.group(1))
    tel = (telescope or '').upper()
    det = (detector or '').upper()
    if tel == 'JWST' or det in ('IR', 'NIC1', 'NIC2', 'NIC3'):
        return n / 100.0
    return n / 1000.0


def inspect_fits(path):
    """Return an ImageRecord (or None if the file has no 2-D science image
    with a celestial WCS)."""
    try:
        with fits.open(path, memmap=True) as hdul:
            h0 = hdul[0].header
            sci_idx, n_sci = None, 0
            for i, hdu in enumerate(hdul):
                if hdu.header.get('NAXIS', 0) == 2 and hdu.header.get('XTENSION', 'IMAGE') in ('IMAGE', None) \
                        and (hdu.header.get('EXTNAME', 'SCI') or 'SCI').upper() in ('SCI', 'PRIMARY'):
                    n_sci += 1
                    if sci_idx is None:
                        sci_idx = i
            if sci_idx is None:
                return None
            hs = hdul[sci_idx].header
            names = {(h.header.get('EXTNAME') or '').upper() for h in hdul}
            ny, nx = int(hs['NAXIS2']), int(hs['NAXIS1'])
            w = WCS(hs, fobj=hdul).celestial
            if not w.has_celestial:
                return None
            ra_c, dec_c = (float(v) for v in w.all_pix2world(nx / 2.0, ny / 2.0, 0))
            corners = np.array([w.all_pix2world(px, py, 0)
                                for px, py in [(0, 0), (nx, 0), (nx, ny), (0, ny)]], dtype=float)
            scale = float(np.mean(proj_plane_pixel_scales(w)) * 3600.0)   # deg/px -> arcsec/px
            telescope = str(h0.get('TELESCOP', hs.get('TELESCOP', '?'))).strip()
            instrument = str(h0.get('INSTRUME', hs.get('INSTRUME', '?'))).strip()
            detector = str(h0.get('DETECTOR', hs.get('DETECTOR', ''))).strip()
            filt = _pick_filter(h0, hs)
            targ = str(h0.get('TARGNAME') or h0.get('OBJECT') or hs.get('TARGNAME') or '').strip()
            n_exp = 0
            if 'HDRTAB' in names:
                try:
                    tab = hdul['HDRTAB'].data
                    col = next((c for c in ('ROOTNAME', 'rootname') if c in tab.columns.names), None)
                    n_exp = len(set(tab[col])) if col else len(tab)
                except Exception:
                    n_exp = 0
            if not n_exp and h0.get('NCOMBINE'):
                n_exp = int(h0.get('NCOMBINE'))
            exptime = float(h0.get('EXPTIME', hs.get('EXPTIME', 0.0)) or 0.0)
            return ImageRecord(
                path=os.path.abspath(path), telescope=telescope, instrument=instrument,
                detector=detector, filter=filt,
                wavelength_um=_filter_wavelength_um(filt, telescope, detector),
                targname=targ, product=_product_type(path), sci_ext=sci_idx,
                shape=(ny, nx), n_sci=n_sci, has_dq='DQ' in names, has_wht='WHT' in names,
                ra_c=ra_c, dec_c=dec_c, pix_scale_arcsec=scale, corners=corners,
                n_exp=n_exp, exptime=exptime)
    except Exception as e:
        warnings.warn(f'Could not inspect {path}: {type(e).__name__}: {e}')
        return None


def _matches_any(rec, patterns):
    keys = [os.path.basename(rec.path), rec.path, rec.targname, rec.field_name]
    for pat in patterns or []:
        for k in keys:
            if k and (fnmatch.fnmatch(k.lower(), pat.lower()) or fnmatch.fnmatch(_norm(k), _norm(pat))):
                return True
    return False


def _sep_arcmin(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 60.0


def _footprint_overlap_fraction(corners_a, corners_b):
    """Bounding-box RA/Dec overlap of two footprints as a fraction of A
    (notebook cell 27, unchanged)."""
    def bbox(c):
        return c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()
    ra0a, ra1a, dec0a, dec1a = bbox(corners_a)
    ra0b, ra1b, dec0b, dec1b = bbox(corners_b)
    cos_dec = np.cos(np.radians((dec0a + dec1a) / 2))
    ra_overlap = max(0.0, min(ra1a, ra1b) - max(ra0a, ra0b)) * cos_dec
    dec_overlap = max(0.0, min(dec1a, dec1b) - max(dec0a, dec0b))
    area_a = (ra1a - ra0a) * cos_dec * (dec1a - dec0a)
    return (ra_overlap * dec_overlap) / area_a if area_a > 0 else 0.0


def load_catalogs(catalog_dir):
    """Every *.csv in catalog_dir with columns galaxy_id, ra, dec (optional:
    group, morph_gt, ned_type). Returns one concatenated DataFrame."""
    frames = []
    if catalog_dir and os.path.isdir(catalog_dir):
        for p in sorted(glob.glob(os.path.join(catalog_dir, '*.csv'))):
            df = pd.read_csv(p, comment='#')
            if not {'galaxy_id', 'ra', 'dec'} <= set(df.columns):
                warnings.warn(f'catalogue {p} lacks galaxy_id/ra/dec columns -- ignored')
                continue
            if 'group' not in df.columns:
                df['group'] = os.path.splitext(os.path.basename(p))[0]
            df['catalog_file'] = os.path.basename(p)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['group', 'galaxy_id', 'ra', 'dec', 'morph_gt'])
    cat = pd.concat(frames, ignore_index=True)
    if 'morph_gt' not in cat.columns:
        cat['morph_gt'] = np.nan
    return cat


def _catalog_rows_in_footprint(cat, corners):
    if cat is None or len(cat) == 0:
        return cat.iloc[0:0] if cat is not None else None
    ra0, ra1 = corners[:, 0].min(), corners[:, 0].max()
    dec0, dec1 = corners[:, 1].min(), corners[:, 1].max()
    return cat[(cat['ra'] >= ra0) & (cat['ra'] <= ra1) & (cat['dec'] >= dec0) & (cat['dec'] <= dec1)]


def assign_fields(records, cfg, catalog=None):
    """Union-find grouping: two images belong to the same field when their
    frame centres are within field_link_arcmin or their footprints overlap.
    Field name: config alias (pattern on TARGNAME/filename) > catalogue
    group whose members fall inside the field > most common TARGNAME."""
    n = len(records)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    link = cfg['defaults'].get('field_link_arcmin', DEFAULTS['field_link_arcmin'])
    for i in range(n):
        for j in range(i + 1, n):
            a, b = records[i], records[j]
            if (_sep_arcmin(a.ra_c, a.dec_c, b.ra_c, b.dec_c) <= link or
                    _footprint_overlap_fraction(a.corners, b.corners) > 0 and
                    _footprint_overlap_fraction(b.corners, a.corners) > 0):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(records[i])

    aliases = cfg.get('field_aliases') or []
    used = Counter()
    for members in groups.values():
        name = None
        for rule in aliases:
            pats = rule.get('match', [])
            pats = [pats] if isinstance(pats, str) else pats
            if any(_matches_any(r, pats) for r in members):
                name = rule['name']
                break
        if name is None and catalog is not None and len(catalog) > 0:
            votes = Counter()
            for r in members:
                inside = _catalog_rows_in_footprint(catalog, r.corners)
                votes.update(inside['group'].tolist())
            if votes:
                name = votes.most_common(1)[0][0]
        if name is None:
            tn = Counter(r.targname for r in members if r.targname)
            name = tn.most_common(1)[0][0] if tn else os.path.basename(members[0].path)
        used[name] += 1
        if used[name] > 1:
            name = f'{name} ({used[name]})'
        for r in members:
            r.field_name = name
    return records


def discover_images(data_dir, cfg, catalog=None, verbose=True):
    """Find + inspect every FITS image under data_dir, assign fields, apply
    the exclude list. Returns (included_records, excluded_records)."""
    paths = sorted(set(glob.glob(os.path.join(data_dir, '**', '*.fits'), recursive=True) +
                       glob.glob(os.path.join(data_dir, '**', '*.fits.gz'), recursive=True)))
    recs = []
    for p in paths:
        r = inspect_fits(p)
        if r is None:
            if verbose:
                print(f'  skip (no 2-D celestial science image): {os.path.relpath(p, data_dir)}')
            continue
        recs.append(r)
    assign_fields(recs, cfg, catalog)
    included, excluded = [], []
    for r in recs:
        if _matches_any(r, cfg.get('exclude')):
            r.status = 'excluded (config)'
            excluded.append(r)
        else:
            included.append(r)
    return included, excluded


def manifest_frame(records):
    rows = []
    for r in records:
        rows.append({'field': r.field_name, 'file': os.path.basename(r.path), 'telescope': r.telescope,
                     'instrument': r.instrument, 'detector': r.detector, 'filter': r.filter,
                     'wavelength_um': r.wavelength_um, 'product': r.product, 'targname': r.targname,
                     'shape': f'{r.shape[0]}x{r.shape[1]}', 'pix_scale_arcsec': round(r.pix_scale_arcsec, 4),
                     'ra_center': round(r.ra_c, 5), 'dec_center': round(r.dec_c, 5),
                     'has_dq': r.has_dq, 'has_wht': r.has_wht, 'status': r.status, 'path': r.path})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# 3. JOBS -- one job = one detection image (+ optional aligned colour bands)
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ImageJob:
    field_name: str
    bands: list            # list[ImageRecord], sorted blue -> red
    detect: ImageRecord    # reddest band
    is_primary: bool = False

    @property
    def instrument(self):
        return self.detect.instrument

    @property
    def detector(self):
        return self.detect.detector

    @property
    def label(self):
        filt = '+'.join(b.filter for b in self.bands)
        tag = f'{self.detect.instrument}_{self.detect.detector}_{filt}' if self.detect.detector \
            else f'{self.detect.instrument}_{filt}'
        return f'{tag}_{_slug(os.path.splitext(os.path.basename(self.detect.path))[0])[:24]}'

    @property
    def sky_area_arcmin2(self):
        h, w = self.detect.shape
        return h * w * (self.detect.pix_scale_arcsec / 60.0) ** 2


def _same_grid(a, b, tol_px):
    if a.shape != b.shape or a.instrument != b.instrument:
        return False
    try:
        with fits.open(a.path) as ha, fits.open(b.path) as hb:
            wa = WCS(ha[a.sci_ext].header, fobj=ha).celestial
            wb = WCS(hb[b.sci_ext].header, fobj=hb).celestial
        ny, nx = a.shape
        # centre must agree to tol_px; corners may additionally drift by a
        # 0.1 % plate-scale difference (per-filter distortion solutions)
        corner_tol = tol_px + 1e-3 * 0.5 * math.hypot(nx, ny)
        for (px, py), tol in [((nx / 2, ny / 2), tol_px), ((0, 0), corner_tol),
                              ((nx - 1, ny - 1), corner_tol), ((0, ny - 1), corner_tol)]:
            ra, dec = wa.all_pix2world(px, py, 0)
            qx, qy = wb.all_world2pix(ra, dec, 0)
            if math.hypot(qx - px, qy - py) > tol:
                return False
        return True
    except Exception:
        return False


def build_jobs(records, cfg):
    """Group each field's images into jobs. Images on the same pixel grid
    (same instrument/shape, WCS agreeing to < band_align_tol_px) with
    different filters become ONE multi-band job; everything else is its own
    single-band job. The primary job of a field (used for threshold
    calibration so the same galaxy isn't counted once per image) is the one
    covering the largest sky area; ties -> CR-combined product -> redder."""
    tol = cfg['defaults'].get('band_align_tol_px', DEFAULTS['band_align_tol_px'])
    by_field = {}
    for r in records:
        by_field.setdefault(r.field_name, []).append(r)
    jobs = []
    for fname, recs in by_field.items():
        # reddest first: each stack is seeded by its detection band
        remaining = sorted(recs, key=lambda r: (r.instrument, -np.nan_to_num(r.wavelength_um, nan=0.0)))
        fjobs = []
        while remaining:
            seed = remaining.pop(0)
            stack = [seed]
            for other in list(remaining):
                if other.filter not in {s.filter for s in stack} and _same_grid(seed, other, tol):
                    stack.append(other)
                    remaining.remove(other)
            stack.sort(key=lambda r: (np.nan_to_num(r.wavelength_um, nan=0.0)))
            fjobs.append(ImageJob(field_name=fname, bands=stack, detect=stack[-1]))
        best = max(fjobs, key=lambda j: (round(j.sky_area_arcmin2, 2),
                                          j.detect.product in COMBINED_PRODUCTS,
                                          np.nan_to_num(j.detect.wavelength_um, nan=0.0)))
        best.is_primary = True
        jobs.extend(fjobs)
    return jobs


# ════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING (notebook cell 25 -- unchanged logic, box size passed in)
# ════════════════════════════════════════════════════════════════════════
def load_sci(rec):
    with fits.open(rec.path) as hdul:
        return np.asarray(hdul[rec.sci_ext].data, dtype=np.float32)


def load_ext(rec, extname):
    """First extension called extname (DQ / WHT) whose shape matches SCI."""
    with fits.open(rec.path) as hdul:
        for hdu in hdul[rec.sci_ext:]:
            if (hdu.header.get('EXTNAME') or '').upper() == extname and hdu.data is not None \
                    and hdu.data.shape == tuple(rec.shape):
                return np.asarray(hdu.data)
    return None


def lacosmic_mask(data, exptime=0.0, sigclip=4.5, sigfrac=0.3, objlim=4.0, niter=2):
    """Cosmic-ray detection by Laplacian edge detection (van Dokkum 2001,
    'L.A.Cosmic'), classical and single-image:
      L+  = positive part of the Laplacian of the 2x-subsampled image,
            block-averaged back (CR edges are sharper than the PSF);
      S'  = L+ / (2 N) minus its 5x5 median, N = sky + Poisson noise model;
      F   = fine-structure image  med3(I) - med7(med3(I));
      CR  = S' > sigclip  and  L+ / F > objlim   (the F test protects
            undersampled stars, whose cores are also sharp but symmetric),
      grown into neighbours with S' > sigfrac * sigclip.
    Cleaned pixels are replaced by the 5x5 median of the remaining pixels.
    Returns (cleaned, mask)."""
    img = np.nan_to_num(np.asarray(data, np.float64))
    crmask = np.zeros(img.shape, bool)
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float64)
    for _ in range(niter):
        sub = np.kron(img, np.ones((2, 2)))
        lp = np.clip(ndimage.convolve(sub, lap, mode='nearest'), 0, None)
        lplus = lp.reshape(img.shape[0], 2, img.shape[1], 2).mean(axis=(1, 3))
        med5 = ndimage.median_filter(img, size=5)
        sky_sig = _robust_sigma((img - med5)[::7, ::7])
        var = sky_sig ** 2 + (np.clip(med5, 0, None) / exptime if exptime > 0 else 0.0)
        noise = np.sqrt(np.maximum(var, 1e-30))
        s = lplus / (2.0 * noise)
        sp = s - ndimage.median_filter(s, size=5)
        m3 = ndimage.median_filter(img, size=3)
        fine = np.maximum(m3 - ndimage.median_filter(m3, size=7), 0.01 * noise)
        new = (sp > sigclip) & (lplus / fine > objlim) & ~crmask
        if not new.any():
            break
        grown = ndimage.binary_dilation(new, structure=np.ones((3, 3))) & (sp > sigfrac * sigclip)
        crmask |= new | grown
        img = _masked_median5(img, crmask)
    return img, crmask


def _masked_median5(img, mask):
    """5x5 median of the unmasked pixels (masked ones excluded)."""
    a = np.where(mask, np.nan, img)
    pad = np.pad(a, 2, mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(pad, (5, 5))
    out = img.copy()
    ys, xs = np.nonzero(mask)
    if ys.size:
        vals = np.nanmedian(win[ys, xs].reshape(ys.size, 25), axis=1)
        out[ys, xs] = np.where(np.isfinite(vals), vals, ndimage.median_filter(img, 5)[ys, xs])
    return out


def reject_bad_pixels(data, dq=None, cr_reject=False, dq_fill_size=7, cr_size=5, cr_n_sigma=4.0,
                      cr_method='median', exptime=0.0):
    n_dq = 0
    if dq is not None:
        bad_dq = (dq != 0)
        n_dq = int(bad_dq.sum())
        if n_dq:
            med = ndimage.median_filter(data, size=dq_fill_size)
            data = np.where(bad_dq, med, data)
    n_cr = 0
    if cr_reject and cr_method == 'lacosmic':
        data, crm = lacosmic_mask(data, exptime=exptime)
        return data, n_dq, int(crm.sum())
    if cr_reject:
        med = ndimage.median_filter(data, size=cr_size)
        resid = data - med
        global_mad = max(float(np.nanmedian(np.abs(resid - np.nanmedian(resid)))) * 1.4826, 1e-6)
        bad_cr = np.abs(resid) > (cr_n_sigma * global_mad)
        n_cr = int(bad_cr.sum())
        if n_cr:
            data = np.where(bad_cr, med, data)
    return data, n_dq, n_cr


class _FlatBackground:
    """Constant sky with the attributes of photutils.Background2D used here."""

    def __init__(self, shape, level, rms):
        self.background = np.full(shape, float(level), dtype=np.float64)
        self.background_rms = np.full(shape, max(float(rms), 1e-30), dtype=np.float64)
        self.background_median = float(level)


def masked_background(img, box_size=150, nodata_mask=None, n_iter=3, smooth_sigmas=(2.0,),
                      detect_nsigma=3.0, dilate_px=8, max_mask_frac=0.6, exclude_percentile=10.0):
    """Source-masked iterative 2-D sky (cell 25, unchanged logic). The final
    source mask is attached as bkg.source_mask (used by select_sky_model)."""
    img_f = np.asarray(img, dtype=np.float64)
    base = np.zeros(img_f.shape, dtype=bool) if nodata_mask is None else np.asarray(nodata_mask, bool)
    sc = SigmaClip(sigma=3.0, maxiters=5)

    def _fit(mask):
        if box_size == 'flat':                       # development stage: constant sky (drizzled products)
            v = img_f[~mask]
            v = v[::max(1, v.size // 2_000_000)]
            if v.size < 100:
                raise ValueError('too few sky pixels for a flat sky')
            _, med, std = sigma_clipped_stats(v, sigma=3.0, maxiters=5)
            return _FlatBackground(img_f.shape, med, std)
        return Background2D(img_f, box_size=box_size, filter_size=3, sigma_clip=sc,
                            bkg_estimator=MedianBackground(),
                            mask=(mask if mask.any() else None),
                            exclude_percentile=exclude_percentile)

    bkg = _fit(base)
    bkg.source_mask = np.zeros(img_f.shape, dtype=bool)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    for _ in range(n_iter):
        resid = (img_f - bkg.background).astype(np.float32)
        src = np.zeros(img_f.shape, dtype=bool)
        for ssig in smooth_sigmas:
            sm = cv2.GaussianBlur(resid, (0, 0), ssig)
            ref = sm[~base]
            if ref.size == 0:
                continue
            sig = 1.4826 * float(np.median(np.abs(ref - np.median(ref))))
            if sig <= 0:
                continue
            src |= (sm > np.median(ref) + detect_nsigma * sig) & ~base
        src = ndimage.binary_opening(src, structure=np.ones((3, 3)))
        src = cv2.dilate(src.astype(np.uint8), kern).astype(bool)
        if (src.sum() / max(float((~base).sum()), 1.0)) > max_mask_frac:
            break
        try:
            bkg = _fit(base | src)
            bkg.source_mask = src
        except Exception:
            break
    return bkg


# ── Sky model selection (large-galaxy support) ──────────────────────────
# A sky box much smaller than a galaxy absorbs the galaxy into the 'sky'
# (verified on HCG 87: the 150 px sky map traces the 87a/b/c envelopes and
# subtracts them before detection). A 'coarse' model -- boxes ~1/4 frame,
# sources masked at a large smoothing scale -- keeps them, but cannot follow
# genuine sky gradients (e.g. un-sky-subtracted JWST _crf frames). The model
# is therefore chosen per image by comparing both fits on SKY pixels only:
# the simpler (coarse) model wins unless it leaves > sky_select_tol more
# residual scatter. Classical model selection, no learning.
def sky_kwargs(model, box, coarse_box):
    """masked_background() keyword arguments for a sky model."""
    if model == 'coarse':
        return dict(box_size=int(coarse_box), smooth_sigmas=(2.0, coarse_box / 10.0),
                    dilate_px=max(8, int(coarse_box / 20)), exclude_percentile=80.0, max_mask_frac=0.85)
    if model == 'flat':
        return dict(box_size='flat', smooth_sigmas=(2.0, coarse_box / 10.0),
                    dilate_px=max(8, int(coarse_box / 20)), max_mask_frac=0.85)
    return dict(box_size=int(box))


def select_sky_model(data, nodata, box, coarse_box, tol, log=print, candidates=('coarse', 'local'),
                     tolerances=None):
    """Return (model, diagnostics) for one linear image. Each candidate is
    fitted with sources masked and scored by its residual scatter on pixels
    that are sky for EVERY candidate; the simplest candidate (order
    flat < coarse < local) within its tolerance of the best score wins.

    development stage: 'local' (150 px boxes) is not offered for drizzled products --
    their sky was already subtracted by AstroDrizzle, and a box smaller than
    a galaxy absorbs the galaxy (HCG 87 earlier stage; HCG 42 earlier stage: NGC 3091 was
    removed entirely because local 'won' on halo-contaminated sky pixels)."""
    order = [m for m in ('flat', 'coarse', 'local') if m in candidates]
    tolerances = tolerances or {}
    if 'local' in order and coarse_box <= 2 * box and len(order) == 2 and 'flat' not in order:
        return 'local', {}
    fits_, sky = {}, ~nodata
    for m in order:
        try:
            fits_[m] = masked_background(data, nodata_mask=nodata, **sky_kwargs(m, box, coarse_box))
            sky &= ~fits_[m].source_mask
        except Exception as e:
            log(f'  sky model {m} failed ({type(e).__name__}: {e})')
    if not fits_:
        return order[-1], {}
    if sky.sum() < 1000:
        return next(iter(fits_)), {}
    step = max(1, int(sky.sum() // 2_000_000))
    sig = {m: _robust_sigma((data - b.background)[sky][::step]) for m, b in fits_.items()}
    best = min(sig.values())
    model = next(m for m in order if m in sig and sig[m] <= (1.0 + tolerances.get(m, tol)) * best)
    log(f'  sky model: {model} (sky-pixel scatter ' + ', '.join(f'{m} {s:.4g}' for m, s in sig.items())
        + f'; boxes {box} / {int(coarse_box)} px)')
    return model, {f'sigma_{m}': s for m, s in sig.items()}


def _robust_sigma(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def eight_bit_sky_check(img01, nodata_mask=None, box_size=150, sky_kw=None):
    """Does the [0, 1] detection image still represent the sky properly once
    quantised to 8 bits?  Two objective failure modes (both seen on real
    JWST NIRCam frames in this project):
      * 'quantised'  -- median Background2D sky RMS (the quantity the
        detector divides by) < 1 grey level: every k*RMS threshold
        degenerates (JWST F150W here: RMS exactly 0 in every box);
      * 'sky-bright' -- sky median above mid-grey: the clip limits came from
        noise rather than sources, so sources have < half the grey range.
    Returns (ok, reason, sky_median, sky_rms) in grey levels."""
    u8 = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8).astype(np.float64)
    v = u8[~nodata_mask] if nodata_mask is not None and (~nodata_mask).any() else u8.ravel()
    med = float(np.median(v[::max(1, v.size // 2_000_000)]))
    try:
        # exactly the estimator background_relative_watershed() will use
        rms = masked_background(u8, nodata_mask=nodata_mask, **(sky_kw or {'box_size': box_size})).background_rms
        sig = float(np.median(rms[~nodata_mask] if nodata_mask is not None and (~nodata_mask).any() else rms))
    except Exception:
        sig = _robust_sigma(v)
    if sig < 1.0:
        return False, 'quantised', med, sig
    if med > 127.5:
        return False, 'sky-bright', med, sig
    return True, 'ok', med, sig


def process_channel(data, dq=None, cr_reject=False, box_size=150, h=10, nodata_mask=None,
                    stretch='auto', log=print, sky_kw=None):
    """DQ/CR rejection -> NaN fill -> source-masked background subtraction
    -> 8-bit -> NLM denoise -> asinh, output in [0, 1].

    stretch='percentile' : the validated earlier stage chain (1 / 99.5 percentile
        clip, quantise, NLM, asinh normalised at the 99th percentile).
    stretch='noise'      : noise-referenced asinh (Lupton, Gunn & Szalay 1999:
        softening = sky sigma) applied in FLOAT before quantising, so the
        sky noise spans many grey levels whatever the scene's dynamic range.
    stretch='auto'       : 'percentile', unless eight_bit_sky_check() shows the
        result cannot represent the sky -- then 'noise'. Images that already
        worked therefore give identical output."""
    data, n_dq, n_cr = reject_bad_pixels(data, dq=dq, cr_reject=cr_reject)
    if n_dq or n_cr:
        log(f'    artifact rejection: {n_dq} DQ-flagged px, {n_cr} additional CR px replaced '
            f'({(n_dq + n_cr) / data.size:.2%} of frame)')
    data = np.nan_to_num(data, nan=np.nanmedian(data))
    bkg = masked_background(data, nodata_mask=nodata_mask, **(sky_kw or {'box_size': box_size}))
    data_sub = data - bkg.background

    if stretch in ('percentile', 'auto'):
        d_min, d_max = np.percentile(data_sub, 1), np.percentile(data_sub, 99.5)
        data_8 = ((np.clip(data_sub, d_min, d_max) - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(data_8, h=h)
        norm = ImageNormalize(vmin=0, vmax=np.percentile(denoised, 99), stretch=AsinhStretch())
        out = norm(denoised)
        if np.ma.isMaskedArray(out):
            out = out.filled(0)
        out = np.clip(out, 0, 1)
        if stretch == 'percentile':
            return out
        ok, why, med, sig = eight_bit_sky_check(out, nodata_mask, box_size, sky_kw)
        if ok:
            return out
        log(f'    8-bit sky check failed ({why}: sky median {med:.1f}, sky RMS {sig:.2f} grey levels) '
            f'-> noise-referenced asinh stretch')

    valid = data_sub[~nodata_mask] if nodata_mask is not None and (~nodata_mask).any() else data_sub
    sky_sig = max(_robust_sigma(valid[::max(1, valid.size // 2_000_000)]), 1e-12)
    y = np.arcsinh(data_sub / sky_sig)
    y_lo, y_hi = np.arcsinh(-3.0), float(np.percentile(y, 99.9))
    data_8 = ((np.clip(y, y_lo, y_hi) - y_lo) / max(y_hi - y_lo, 1e-8) * 255).astype(np.uint8)
    return np.clip(cv2.fastNlMeansDenoising(data_8, h=h).astype(np.float32) / 255.0, 0, 1)


def build_nodata_mask(raw, wht, p, log):
    """NaN pixels + (optionally) near-zero-weight drizzle pixels (WHT), with
    the earlier stage mode-based reference and safety ceiling."""
    nodata = ~np.isfinite(raw)
    if wht is not None and wht.shape == raw.shape:
        finite_w = wht[np.isfinite(wht) & (wht > 0)]
        if finite_w.size > 0:
            counts, edges = np.histogram(finite_w, bins=50)
            mb = int(np.argmax(counts))
            w_ref = 0.5 * (edges[mb] + edges[mb + 1])
            low = np.isfinite(wht) & (wht < p['weight_frac_thresh'] * w_ref)
            cand = nodata | low
            if cand.mean() > p['weight_mask_max_frac']:
                log(f'  WHT mask skipped: weight_frac_thresh={p["weight_frac_thresh"]} would mark '
                    f'{cand.mean():.1%} of the frame as no-data (NaN alone {nodata.mean():.1%})')
            else:
                added = int((low & ~nodata).sum())
                nodata = cand
                if added:
                    log(f'  WHT mask: +{added} low-weight px ({added / nodata.size:.2%} of frame)')
    return nodata


def nodata_footprint(nodata, min_area):
    """Connected-component area filter: keep only no-data regions of at
    least min_area px (the mosaic outline / chip gaps), dropping isolated
    flagged pixels."""
    if not nodata.any():
        return nodata
    lab, n = ndimage.label(nodata)
    if n == 0:
        return nodata
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


# ════════════════════════════════════════════════════════════════════════
# 5. DETECTION + SEGMENTATION (cells 97 / 98 / 100 -- unchanged logic)
# ════════════════════════════════════════════════════════════════════════
def adaptive_k_sigma(sig_map, rms, target_lo, target_hi, k_hi, k_lo, k_step, k_max, valid_mask=None):
    mid = 0.5 * (target_lo + target_hi)
    denom = float(valid_mask.sum()) if valid_mask is not None else float(sig_map.size)

    def frac_at(kk):
        hit = sig_map > (kk * rms)
        if valid_mask is not None:
            hit &= valid_mask
        return float(hit.sum()) / denom if denom > 0 else 0.0

    k = k_hi
    best_k, best_frac, best_diff = k_hi, frac_at(k_hi), 1e9
    while k >= k_lo:
        frac = frac_at(k)
        if target_lo <= frac <= target_hi:
            return k, frac
        diff = abs(frac - mid)
        if diff < best_diff:
            best_diff, best_k, best_frac = diff, k, frac
        k -= k_step
    f_hi = frac_at(k_hi)
    if f_hi > target_hi:
        kk = k_hi
        best_k, best_frac, best_diff = k_hi, f_hi, abs(f_hi - mid)
        while kk < k_max:
            kk *= 1.15
            frac = frac_at(kk)
            if frac < target_lo:
                # Step-like fraction (saturated plateaus of a frame-filling
                # galaxy): never tighten past the band's lower edge into
                # 'nothing detected' -- keep the last k that still detected.
                break
            diff = abs(frac - mid)
            if diff < best_diff:
                best_diff, best_k, best_frac = diff, kk, frac
            if frac <= target_hi:
                break
    return best_k, best_frac


def background_relative_watershed(img_u8, p, box_size, nodata_mask=None, sky_kw=None):
    img_f = img_u8.astype(np.float64)
    bkg = masked_background(img_f, nodata_mask=nodata_mask, **(sky_kw or {'box_size': box_size}))
    sig_map = img_f - bkg.background
    valid = None if nodata_mask is None else ~nodata_mask
    k_sigma, _ = adaptive_k_sigma(sig_map, bkg.background_rms, p['fg_frac_lo'], p['fg_frac_hi'],
                                  p['k_hi'], p['k_lo'], p['k_step'], p['k_max'], valid_mask=valid)
    fg = sig_map > (k_sigma * bkg.background_rms)
    if nodata_mask is not None:
        fg &= ~nodata_mask
    fg_frac = float(fg.mean())
    smoothed = cv2.GaussianBlur(sig_map.astype(np.float32), (0, 0), sigmaX=p['smooth_sigma'])
    smoothed_pos = np.clip(smoothed, 0, None)
    lm = smoothed_pos.copy()
    lm[~fg] = 0
    if fg.any() and lm.max() > 0:
        h_value = max(p['h_sigma_mult'] * float(np.median(bkg.background_rms)), 1e-6)
        coords = peak_local_max(h_maxima(lm, h_value), min_distance=p['min_distance'], labels=fg)
    else:
        coords = np.empty((0, 2), dtype=int)
    peaks = np.zeros(img_f.shape, dtype=bool)
    if len(coords):
        peaks[tuple(coords.T)] = True
    markers, _ = ndimage.label(peaks)
    ws = watershed(-smoothed_pos, markers, mask=fg)
    return fg, ws, bkg, k_sigma, fg_frac


def clean_labeled_mask(ws_labels, kernel):
    cleaned = np.zeros_like(ws_labels, dtype=np.int32)
    for rp in measure.regionprops(ws_labels):
        minr, minc, maxr, maxc = rp.bbox
        r0, r1 = max(0, minr - 3), min(ws_labels.shape[0], maxr + 3)
        c0, c1 = max(0, minc - 3), min(ws_labels.shape[1], maxc + 3)
        sub = (ws_labels[r0:r1, c0:c1] == rp.label).astype(np.uint8) * 255
        if sub.sum() == 0:
            continue
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kernel, iterations=1)
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel, iterations=1)
        filled = binary_fill_holes(sub > 0)
        cleaned[r0:r1, c0:c1][filled] = rp.label
    return cleaned


def separate_stars_galaxies(labeled, img_u8, bkg_background, min_area_px, psf_radius_px,
                            conc_thresh, area_max_px):
    """Cell 104, unchanged logic; aperture test done on a bbox sub-array
    (identical result, avoids full-frame masks per object)."""
    regions_all = measure.regionprops(labeled, intensity_image=img_u8)
    seg_mask = np.zeros_like(labeled)
    kept, lbl = [], 0
    for rp in regions_all:
        if rp.area >= min_area_px:
            lbl += 1
            seg_mask[labeled == rp.label] = lbl
            kept.append(rp)
    sig = np.clip(img_u8.astype(np.float64) - bkg_background, 0, None)
    star_labels, galaxy_labels, obj_info = [], [], []
    for rp in kept:
        minr, minc, maxr, maxc = rp.bbox
        sub_mask = labeled[minr:maxr, minc:maxc] == rp.label
        sub_sig = sig[minr:maxr, minc:maxc]
        total = sub_sig[sub_mask].sum()
        if total == 0:
            continue
        cy, cx = rp.centroid
        yy, xx = np.ogrid[minr:maxr, minc:maxc]
        ap = ((yy - cy) ** 2 + (xx - cx) ** 2) <= psf_radius_px ** 2
        center_frac = sub_sig[sub_mask & ap].sum() / total
        is_star = (center_frac > conc_thresh) and (rp.area < area_max_px)
        obj_info.append({'label': rp.label, 'area': rp.area, 'center_frac': round(center_frac, 4),
                         'centroid_r': round(cy, 1), 'centroid_c': round(cx, 1),
                         'obj_type': 'star' if is_star else 'galaxy'})
        (star_labels if is_star else galaxy_labels).append(rp.label)
    return {'n_raw': int(labeled.max()), 'kept': kept, 'seg_mask': seg_mask, 'lbl': lbl,
            'star_labels': star_labels, 'galaxy_labels': galaxy_labels, 'obj_info': obj_info}


def reject_edge_galaxies(sep, labeled, nodata_eff, flux_img, max_edge_flux_frac=0.05, edge_px=10):
    """Generic replacement for the hand-placed HCG 16 edge fixes. A galaxy
    is truncated -- and its CAS values meaningless -- when a non-negligible
    share of its LINEAR flux lies against the frame border or the (buffered)
    no-data footprint. Measuring flux rather than boundary length keeps a
    large galaxy whose faint halo merely reaches an edge, while mosaic-edge
    artefacts (all their flux on the edge) are still removed. Removed labels
    stay in the segmentation map. Returns the removed labels."""
    H, W = labeled.shape
    nd = np.zeros((H, W), dtype=bool) if nodata_eff is None else nodata_eff.astype(bool).copy()
    nd[:1, :] = nd[-1:, :] = True
    nd[:, :1] = nd[:, -1:] = True
    near = cv2.dilate(nd.astype(np.uint8), np.ones((2 * edge_px + 1,) * 2, np.uint8)) > 0
    pos = np.clip(flux_img, 0, None)
    removed = []
    for rp in sep['kept']:
        if rp.label not in sep['galaxy_labels']:
            continue
        minr, minc, maxr, maxc = rp.bbox
        sub = labeled[minr:maxr, minc:maxc] == rp.label
        f = pos[minr:maxr, minc:maxc]
        tot = float(f[sub].sum())
        if tot <= 0:
            continue
        if float(f[sub & near[minr:maxr, minc:maxc]].sum()) / tot > max_edge_flux_frac:
            removed.append(rp.label)
    sep['galaxy_labels'] = [l for l in sep['galaxy_labels'] if l not in set(removed)]
    for o in sep['obj_info']:
        if o['label'] in removed:
            o['obj_type'] = 'edge'
    return removed


def _normalized_smooth(img, weight, sigma):
    """Normalized convolution (Knutsson & Westin 1993): Gaussian smoothing
    of the valid pixels only, renormalised by the smoothed weight. Thin
    no-data features (WFPC2 chip seams, flagged columns) are interpolated
    from their neighbours instead of appearing as deep valleys."""
    w = weight.astype(np.float32)
    num = cv2.GaussianBlur((img * w).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0).astype(np.float32), den


def _adjacent_saddles(lab, val):
    """For every pair of 4-adjacent non-zero labels: the saddle height =
    highest value along their shared boundary (taking the lower side of
    each boundary pixel pair). Returns {(a, b): saddle} with a < b."""
    out = {}
    for la, lb, va, vb in [(lab[:, :-1], lab[:, 1:], val[:, :-1], val[:, 1:]),
                           (lab[:-1, :], lab[1:, :], val[:-1, :], val[1:, :])]:
        m = (la != lb) & (la > 0) & (lb > 0)
        if not m.any():
            continue
        a, b = np.minimum(la[m], lb[m]), np.maximum(la[m], lb[m])
        v = np.minimum(va[m], vb[m])
        df = pd.DataFrame({'a': a, 'b': b, 'v': v}).groupby(['a', 'b'])['v'].max()
        for (x, y), sv in df.items():
            out[(x, y)] = max(out.get((x, y), -np.inf), sv)
    return out


def grow_segments(labeled, sep, sig_map, nodata, footprint, fg, p, kernel, log=print):
    """Large-galaxy extent: hysteresis region growing + dynamics merging.

    1. Smoothing by normalized convolution (thin no-data gaps interpolated).
    2. GROW: every kept detection is a seed; marker-controlled watershed
       over the smoothed signal claims pixels down to a LOW threshold of
       grow_k x the sky noise of the smoothed map (measured on sky pixels).
       Low-threshold regions holding no seed are dropped (hysteresis).
    3. MERGE ('dynamics' of maxima, the criterion behind h-maxima): two
       adjacent regions are one object when the lower region's peak barely
       rises above the highest point of their shared boundary,
           (lower_peak - saddle) / (lower_peak - sky) < merge_contrast.
       Re-joins a galaxy split by a chip seam or into several bright knots,
       and a nucleus mis-seeded as a 'star' with its own bulge; two
       galaxies, or a foreground star on a galaxy, have a real valley.
       Groups containing a galaxy seed become one galaxy; star-only groups
       are left alone. Unmerged stars keep only their core segment.
    Returns (grown_labels, sep) with sep['kept'/'galaxy_labels'/'star_labels']
    updated (merged groups take the label of their largest galaxy seed)."""
    weight = ~nodata if nodata is not None else np.ones(sig_map.shape, bool)
    sm, den = _normalized_smooth(sig_map, weight, p['grow_smooth_sigma'])
    region_ok = den > 0.2
    if footprint is not None:
        region_ok &= ~footprint
    sky = region_ok & weight & ~(cv2.dilate(fg.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
    if sky.sum() < 1000:
        return labeled, sep
    ref = sm[sky][::max(1, int(sky.sum() // 2_000_000))]
    med, s = float(np.median(ref)), _robust_sigma(ref)
    if s <= 0:
        return labeled, sep
    kept_ids = np.array([rp.label for rp in sep['kept']], dtype=labeled.dtype)
    seeds = np.where(np.isin(labeled, kept_ids), labeled, 0).astype(np.int32)
    low = ((sm > med + p['grow_k'] * s) & region_ok) | (seeds > 0)
    grown = watershed(-sm, seeds, mask=low).astype(np.int32)

    # dynamics merging (union-find over adjacent pairs)
    gal, star = set(sep['galaxy_labels']), set(sep['star_labels'])
    nmax = int(grown.max())
    peak = ndimage.maximum(sm, grown, index=np.arange(nmax + 1))
    area = np.bincount(grown.ravel(), minlength=nmax + 1)
    parent = {int(l): int(l) for l in kept_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), sad in _adjacent_saddles(grown, sm).items():
        a, b = int(a), int(b)
        if a not in parent or b not in parent:
            continue
        lo = min(peak[a], peak[b])
        if (lo - sad) / max(lo - med, 1e-6) < p['merge_contrast']:
            parent[find(a)] = find(b)
    groups = {}
    for l in parent:
        groups.setdefault(find(l), []).append(l)
    relabel = np.arange(nmax + 1, dtype=np.int32)
    absorbed, n_merged = set(), 0
    for members in groups.values():
        g_members = [m for m in members if m in gal]
        if len(members) < 2 or not g_members:
            continue
        rep = max(g_members, key=lambda m: area[m])
        for m in members:
            relabel[m] = rep
            if m != rep:
                absorbed.add(m)
        n_merged += len(members) - 1
    grown = relabel[grown]

    # unmerged stars keep only their core
    lone_stars = np.array([l for l in star if l not in absorbed], dtype=np.int32)
    if lone_stars.size:
        sp = np.isin(grown, lone_stars)
        grown[sp & (grown != seeds)] = 0
    grown = clean_labeled_mask(grown, kernel)

    sep['star_labels'] = [l for l in sep['star_labels'] if l not in absorbed]
    sep['galaxy_labels'] = [l for l in sep['galaxy_labels'] if l not in absorbed]
    grown_rp = {rp.label: rp for rp in measure.regionprops(grown)}
    gal_now = set(sep['galaxy_labels'])
    sep['kept'] = [grown_rp.get(rp.label, rp) if rp.label in gal_now else rp
                   for rp in sep['kept'] if rp.label not in absorbed]
    for o in sep['obj_info']:
        if o['label'] in absorbed:
            o['obj_type'] = 'merged'
    log(f'  region growing: low threshold {p["grow_k"]:.1f} sigma of smoothed sky; '
        f'{n_merged} fragment(s) merged into neighbours (dynamics < {p["merge_contrast"]})')
    return grown, sep


# ════════════════════════════════════════════════════════════════════════
# 6. PSF FWHM, CUTOUTS, CATALOGUE MATCHING
# ════════════════════════════════════════════════════════════════════════
def radial_fwhm(cutout, cy, cx, max_r=15):
    h, w = cutout.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r_int = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    prof = []
    for radius in range(0, max_r + 1):
        m = r_int == radius
        if m.sum():
            prof.append((radius, float(np.median(cutout[m]))))
    if len(prof) < 3:
        return None
    radii = np.array([q[0] for q in prof])
    vals = np.array([q[1] for q in prof])
    peak, floor = vals[0], vals[-1]
    if peak <= floor:
        return None
    half = floor + (peak - floor) / 2.0
    below = np.where(vals <= half)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i1 = below[0]
    i0 = i1 - 1
    r0, r1 = radii[i0], radii[i1]
    v0, v1 = vals[i0], vals[i1]
    r_half = r0 if v1 == v0 else r0 + (half - v0) * (r1 - r0) / (v1 - v0)
    return 2.0 * r_half


def measure_target_fwhm(kept, star_labels, linear_data, assumed_fwhm_px, pad_r=15):
    meas = []
    stars = set(star_labels)
    for rp in kept:
        if rp.label not in stars:
            continue
        cy, cx = rp.centroid
        r0, r1, c0, c1 = int(cy - pad_r), int(cy + pad_r), int(cx - pad_r), int(cx + pad_r)
        if r0 < 0 or c0 < 0 or r1 > linear_data.shape[0] or c1 > linear_data.shape[1]:
            continue
        f = radial_fwhm(linear_data[r0:r1, c0:c1], cy - r0, cx - c0, max_r=pad_r - 1)
        if f is not None and 0.5 < f < pad_r:
            meas.append(f)
    if len(meas) >= 3:
        return float(np.median(meas)), float(np.std(meas)), meas
    return assumed_fwhm_px, None, meas


def extract_cutouts(kept, galaxy_labels, labeled, display_rgb, gray_u8, prefix, pad):
    """Generic cutouts for every galaxy candidate, ranked by area
    ({prefix}_Obj1, _Obj2, ...)."""
    cands = [rp for rp in kept if rp.label in set(galaxy_labels)]
    cutouts = {}
    for i, rp in enumerate(sorted(cands, key=lambda r: r.area, reverse=True), start=1):
        minr, minc, maxr, maxc = rp.bbox
        r0, r1 = max(0, minr - pad), min(display_rgb.shape[0], maxr + pad)
        c0, c1 = max(0, minc - pad), min(display_rgb.shape[1], maxc + pad)
        cutouts[f'{prefix}_Obj{i}'] = {'rgb': display_rgb[r0:r1, c0:c1], 'gray': gray_u8[r0:r1, c0:c1],
                                       'mask': labeled[r0:r1, c0:c1] == rp.label,
                                       'bbox': (r0, c0, r1, c1), 'rp': rp}
    return cutouts


def match_catalog(cutouts, labeled, wcs, catalog, radius_arcsec, pix_scale_arcsec):
    """Pure-WCS cross-match (replaces the SQ-only topology heuristic): a
    catalogue position is assigned to the galaxy whose segment CONTAINS it,
    else to the nearest galaxy centroid within radius_arcsec. One-to-one.
    Returns {cutout_name: catalogue_row}."""
    if catalog is None or len(catalog) == 0 or wcs is None or not cutouts:
        return {}
    H, W = labeled.shape
    label_to_name = {d['rp'].label: n for n, d in cutouts.items()}
    cents = {n: d['rp'].centroid for n, d in cutouts.items()}
    rad_px = radius_arcsec / max(pix_scale_arcsec, 1e-6)
    out, used = {}, set()
    for _, row in catalog.iterrows():
        try:
            px, py = (float(v) for v in wcs.all_world2pix(row['ra'], row['dec'], 0))
        except Exception:
            continue
        if not (np.isfinite(px) and np.isfinite(py)) or not (0 <= px < W and 0 <= py < H):
            continue
        name = label_to_name.get(int(labeled[int(py), int(px)]))
        if name is None or name in used:
            r_row = row.get('match_radius_arcsec') if hasattr(row, 'get') else None
            best, best_d = None, (rad_px if r_row is None or not np.isfinite(r_row)
                                  else float(r_row) / max(pix_scale_arcsec, 1e-6))
            for n, (cy, cx) in cents.items():
                if n in used:
                    continue
                d = math.hypot(cx - px, cy - py)
                if d <= best_d:
                    best, best_d = n, d
            name = best
        if name is not None:
            out[name] = row
            used.add(name)
    return out


# ════════════════════════════════════════════════════════════════════════
# 7. CAS + GRANULOMETRY + SHAPE (cell 104 -- unchanged)
# ════════════════════════════════════════════════════════════════════════
def compute_concentration_cog(cutout_gray, mask):
    masked = cutout_gray.astype(np.float64) * mask
    total = masked.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0, 0.0, 0.0
    wv = masked[ys, xs]
    cx, cy = np.average(xs, weights=wv), np.average(ys, weights=wv)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    idx = np.argsort(d)
    sd, cf = d[idx], np.cumsum(wv[idx]) / total
    r20 = sd[min(np.searchsorted(cf, 0.20), len(sd) - 1)]
    r80 = sd[min(np.searchsorted(cf, 0.80), len(sd) - 1)]
    r20 = max(r20, 1.0)
    if r80 <= r20:
        return 0.0, r20, r80
    return 5.0 * np.log10(r80 / r20), r20, r80


def compute_asymmetry(cutout_gray, mask, bg_mask):
    f = cutout_gray.astype(np.float64)
    masked = f * mask
    total = np.abs(masked).sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    ys, xs = np.where(mask)
    wv = masked[ys, xs]
    cx, cy = np.average(xs, weights=wv), np.average(ys, weights=wv)
    H, W = masked.shape
    centred = np.roll(np.roll(masked, int(round(H / 2.0 - cy)), axis=0), int(round(W / 2.0 - cx)), axis=1)
    A_raw = np.abs(centred - ndimage_rotate(centred, 180, reshape=False, order=1)).sum() / total
    bg = f * bg_mask
    bg_flux = np.abs(bg).sum()
    A_bg = np.abs(bg - ndimage_rotate(bg, 180, reshape=False, order=1)).sum() / bg_flux if bg_flux > 0 else 0.0
    return max(A_raw - A_bg, 0.0), A_raw, A_bg


def compute_smoothness(cutout_gray, mask, bg_mask, sigma):
    f = cutout_gray.astype(np.float64)
    total = np.abs(f[mask]).sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    filled = np.where(mask, f, float(np.median(f[mask])))
    sm = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), sigmaX=sigma).astype(np.float64)
    S_raw = np.abs(f[mask] - sm[mask]).sum() / total
    bg_flux = np.abs(f[bg_mask]).sum()
    if bg_flux > 0:
        bgf = np.where(bg_mask, f, float(np.median(f[bg_mask])))
        bgs = cv2.GaussianBlur(bgf.astype(np.float32), (0, 0), sigmaX=sigma).astype(np.float64)
        S_bg = np.abs(f[bg_mask] - bgs[bg_mask]).sum() / bg_flux
    else:
        S_bg = 0.0
    return max(S_raw - S_bg, 0.0), S_raw, S_bg


def compute_granulometry(cutout_gray, mask, r_values):
    f = cutout_gray.astype(np.float64)
    if np.abs(f[mask]).sum() == 0:
        return np.zeros(len(r_values))
    filled = np.where(mask, f, float(np.median(f[mask]))).astype(np.float32)
    vols = []
    for r in r_values:
        opened = filled if r == 0 else cv2.morphologyEx(
            filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        vols.append(float(np.abs(opened[mask].astype(np.float64)).sum()))
    return np.array(vols)


def granulometric_clumpiness_scalar(volumes, r_values, r_target):
    i0 = list(r_values).index(0)
    it = min(range(len(r_values)), key=lambda i: abs(r_values[i] - r_target))
    V0 = volumes[i0]
    return 0.0 if V0 == 0 else max(0.0, (V0 - volumes[it]) / V0)


def compute_shape_moments(cutout_gray, mask):
    masked = cutout_gray.astype(np.float64) * mask.astype(np.float64)
    m8 = mask.astype(np.uint8)
    Mb, M = cv2.moments(m8, binaryImage=True), cv2.moments(masked)
    if M['m00'] == 0 or Mb['m00'] == 0:
        return {'area': 0, 'total_flux': 0, 'perimeter': 0, 'eccentricity': 0, 'ellipticity': 0,
                'hu_moments': [0] * 7}
    mu20, mu02, mu11 = M['mu20'], M['mu02'], M['mu11']
    tr, det = mu20 + mu02, mu20 * mu02 - mu11 ** 2
    disc = max(tr ** 2 / 4 - det, 0)
    lam1, lam2 = tr / 2 + np.sqrt(disc), tr / 2 - np.sqrt(disc)
    lam2 = max(min(lam2, lam1), 0.0)
    ecc = np.sqrt(1.0 - lam2 / lam1) if lam1 > 0 else 0.0
    ell = 1.0 - np.sqrt(lam2 / lam1) if lam1 > 0 else 0.0
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu = cv2.HuMoments(Mb).flatten()
    return {'area': int(Mb['m00']), 'total_flux': round(M['m00'], 2),
            'perimeter': round(sum(cv2.arcLength(c, True) for c in contours), 1),
            'eccentricity': round(ecc, 4), 'ellipticity': round(ell, 4),
            'hu_moments': [round(-np.sign(h) * np.log10(abs(h) + 1e-30), 4) for h in hu]}


# ════════════════════════════════════════════════════════════════════════
# 7b. COMMON-RESOLUTION SAMPLING + MORPHOLOGY ON LINEAR FLUX (development stage)
# ════════════════════════════════════════════════════════════════════════
# Why this section exists (development stage audit against Hickson 1989 morphologies):
#  * Sections 7's C/A/S functions were fed the 8-bit, asinh-stretched,
#    NLM-denoised DETECTION image (clipped at its 99.5th percentile). Bright
#    galaxy cores saturate at 255, so the curve of growth flattens and C
#    collapses to ~2 for every galaxy -- ellipticals cannot be separated.
#  * compute_asymmetry() normalised the background term by the BACKGROUND
#    flux instead of the galaxy flux (Conselice 2003, eq. 2), so A_bg (~0.3-
#    0.5) always exceeded A_raw and A was clipped to 0.000 for every
#    catalogued member.
# The functions below measure on linear, sky-subtracted flux with the
# standard non-parametric definitions (Conselice 2003; Lotz, Primack &
# Madau 2004; the conventions of the statmorph reference implementation):
#   Petrosian radius (eta = 0.2, circular), C = 5 log10(r80/r20) inside
#   1.5 r_p, A about the asymmetry-minimising centre with a galaxy-flux-
#   normalised background term, S (clumpiness) with a 0.25 r_p boxcar,
#   Gini and M20 on the Lotz 'Gini segmentation map'.
# All classical: apertures, moments, order statistics, linear filtering.
#
# Images are first block-averaged to a common physical pixel scale
# (work_scale_arcsec, default 0.2"/px): CAS values depend on resolution, so
# HST ACS (0.05"), WFPC2 (0.0455"), WFC3 (0.04") and JWST (0.031") frames
# would otherwise not be comparable; it also makes 5000x5000 ACS mosaics
# ~16x cheaper to process.  Sampling theory from earlier stage: averaging before
# decimation is the anti-aliasing step.
def bin_factor(pix_scale_arcsec, work_scale_arcsec):
    if not work_scale_arcsec or not pix_scale_arcsec or pix_scale_arcsec <= 0:
        return 1
    return max(1, int(round(float(work_scale_arcsec) / float(pix_scale_arcsec))))


def block_reduce_masked(data, nodata, f, min_valid_frac=0.5):
    """f x f block average of VALID pixels only. A block with fewer than
    min_valid_frac valid pixels becomes no-data. Returns (img, nodata)."""
    data = np.asarray(data)
    nodata = np.asarray(nodata, bool) | ~np.isfinite(data)
    if f <= 1:
        return np.where(nodata, 0.0, data).astype(np.float32), nodata
    H, W = data.shape
    h, w = H // f, W // f
    v = (~nodata[:h * f, :w * f]).reshape(h, f, w, f)
    d = np.where(v, data[:h * f, :w * f].reshape(h, f, w, f), 0.0)
    n = v.sum(axis=(1, 3))
    out = d.sum(axis=(1, 3), dtype=np.float64) / np.maximum(n, 1)
    nd = n < min_valid_frac * f * f
    out[nd] = 0.0
    return out.astype(np.float32), nd


class BinnedWCS:
    """Celestial WCS of an f x f block-averaged image (pixel centres map to
    the centre of their block in the native frame). origin is always 0."""

    def __init__(self, wcs, f, shift=(0.0, 0.0)):
        self.wcs, self.f = wcs, int(f)
        self.shift = tuple(float(v) for v in shift)        # (dx, dy) working px: true = header + shift

    def all_pix2world(self, x, y, origin=0):
        o = (self.f - 1) / 2.0
        x = np.asarray(x, float) - self.shift[0]
        y = np.asarray(y, float) - self.shift[1]
        return self.wcs.all_pix2world(x * self.f + o, y * self.f + o, 0)

    def all_world2pix(self, ra, dec, origin=0, **kw):
        o = (self.f - 1) / 2.0
        x, y = self.wcs.all_world2pix(ra, dec, 0, quiet=True)
        return (np.asarray(x) - o) / self.f + self.shift[0], (np.asarray(y) - o) / self.f + self.shift[1]


def refine_wcs_shift(seg, lin, galaxy_labels, wcs, catalog, scale, p, log=print):
    """Astrometric sanity check / registration. With >= 2 catalogue members
    in the frame, each member's position is compared with the nearest
    detected galaxy NUCLEUS (smoothed peak) within wcs_search_arcsec. If the
    offsets agree (all within wcs_agree_arcsec of their median) and the
    median shift exceeds wcs_min_shift_arcsec, the header WCS is in error by
    that rigid shift and it is applied. Agreement between independent
    galaxies is the safeguard -- with one member, or inconsistent offsets,
    nothing is changed. (SQ WFC3 iacr02ywq: GSC-2.4 solution off by ~24".)
    Returns (dx, dy) in working pixels."""
    if catalog is None or len(catalog) < 2 or not galaxy_labels:
        return (0.0, 0.0)
    sm = cv2.GaussianBlur(np.asarray(lin, np.float32), (0, 0), 1.5)
    min_area = p['wcs_min_area_arcsec2'] / scale ** 2
    gal = set(galaxy_labels)
    big = np.isin(seg, [rp.label for rp in measure.regionprops(seg) if rp.label in gal and rp.area >= min_area])
    if not big.any():
        return (0.0, 0.0)
    # local nuclei (several per segment: blended pairs share one segment)
    # only DOMINANT peaks count as nuclei (>= wcs_peak_frac of their segment's
    # maximum, >= 5" apart): with every star-forming knot as a candidate,
    # chance coincidences produced a wrong 10" 'registration' on SQ
    segbig = np.where(big, seg, 0)
    pk = peak_local_max(np.where(big, sm, 0), min_distance=max(2, int(round(5.0 / scale))),
                        labels=segbig, num_peaks_per_label=5)
    if len(pk) < 2:
        return (0.0, 0.0)
    smax = ndimage.maximum(sm, segbig, index=np.arange(int(segbig.max()) + 1))
    keep = sm[pk[:, 0], pk[:, 1]] >= p['wcs_peak_frac'] * smax[segbig[pk[:, 0], pk[:, 1]]]
    pk = pk[keep]
    if len(pk) < 2:
        return (0.0, 0.0)
    peaks = pk[:, ::-1].astype(float)                       # (x, y)
    pos = []
    for _, row in catalog.iterrows():
        x, y = (float(v) for v in wcs.all_world2pix(row['ra'], row['dec'], 0))
        if np.isfinite(x) and np.isfinite(y):
            pos.append((x, y))
    if len(pos) < 2:
        return (0.0, 0.0)
    pos = np.array(pos)
    rad, tol = p['wcs_search_arcsec'] / scale, p['wcs_agree_arcsec'] / scale
    # offset voting: every member->nucleus offset is a candidate rigid shift;
    # the winner brings the most members onto a nucleus (ties: smallest shift)
    cands = [pk_ - m for m in pos for pk_ in peaks if np.hypot(*(pk_ - m)) <= rad]
    best = (0, 0.0, None)
    for c in cands:
        hits = [np.min(np.hypot(*(peaks - (m + c)).T)) <= tol for m in pos]
        n = int(sum(hits))
        if n > best[0] or (n == best[0] and best[2] is not None and np.hypot(*c) < best[1]):
            best = (n, float(np.hypot(*c)), c)
    if best[0] < 2 or best[2] is None:
        return (0.0, 0.0)
    c = best[2]
    offs = np.array([peaks[np.argmin(np.hypot(*(peaks - (m + c)).T))] - m for m in pos
                     if np.min(np.hypot(*(peaks - (m + c)).T)) <= tol])
    med = np.median(offs, axis=0)
    spread = float(np.max(np.hypot(*(offs - med).T))) * scale
    mag = float(np.hypot(*med)) * scale
    # the null hypothesis (header correct) must do clearly worse
    n0 = int(sum(np.min(np.hypot(*(peaks - m).T)) <= tol for m in pos))
    if mag > p['wcs_min_shift_arcsec'] and len(offs) > n0 and spread <= p['wcs_agree_arcsec']:
        log(f'  WCS registration: {len(offs)}/{len(pos)} catalogue members land on nuclei after a '
            f'{mag:.1f}" shift (vs {n0} unshifted; scatter {spread:.1f}") -> header WCS corrected')
        return (float(med[0]), float(med[1]))
    if mag > p['wcs_min_shift_arcsec']:
        log(f'  WCS check: offsets {mag:.1f}" but inconsistent (scatter {spread:.1f}") -> header WCS kept')
    return (0.0, 0.0)


def _ap_sum(img, xc, yc, r_out, r_in=0.0):
    """Exact-area circular aperture / annulus sum (photutils, method='exact').
    Returns (sum, area)."""
    from photutils.aperture import CircularAperture, CircularAnnulus
    if r_in > 0:
        ap = CircularAnnulus((xc, yc), r_in=r_in, r_out=r_out)
    else:
        ap = CircularAperture((xc, yc), r=max(r_out, 1e-3))
    s = ap.do_photometry(img, method='exact')[0]
    return float(np.atleast_1d(s)[0]), float(ap.area)


def petrosian_radius(img, xc, yc, r_max, eta=0.2):
    """Circular Petrosian radius: smallest r with
    mean SB in annulus [0.8 r, 1.25 r] / mean SB inside r  <=  eta.
    Returns (r_p, ok)."""
    radii = np.unique(np.concatenate([np.arange(1.0, 10.0, 0.5),
                                      np.geomspace(10.0, max(r_max, 11.0), 60)]))
    radii = radii[radii <= r_max]
    prev = None
    for r in radii:
        f_in, a_in = _ap_sum(img, xc, yc, r)
        f_an, a_an = _ap_sum(img, xc, yc, 1.25 * r, 0.8 * r)
        if f_in <= 0 or a_an <= 0:
            prev = None
            continue
        ratio = (f_an / a_an) / (f_in / a_in)
        if ratio <= eta:
            if prev is not None and prev[1] > eta:          # linear interpolation in r
                r0, q0 = prev
                return float(r0 + (eta - q0) * (r - r0) / (ratio - q0)), True
            return float(r), True
        prev = (r, ratio)
    return float(r_max / 1.5), False


def _rotate180(img, xc, yc):
    """I(2xc - x, 2yc - y), bilinear, zero outside."""
    yy, xx = np.indices(img.shape, dtype=np.float64)
    return ndimage.map_coordinates(img, [2.0 * yc - yy, 2.0 * xc - xx], order=1, cval=0.0)


def _asym_raw(img, xc, yc, R, valid=None):
    """sum|I - I180|, sum|I| and the effective pixel area inside radius R
    about (xc, yc). With a validity map, a pixel counts only if both it and
    its 180-degree partner are valid (statmorph convention), so chip gaps /
    neighbours / frame edges do not masquerade as asymmetry."""
    if valid is None:
        w = np.ones(img.shape)
    else:
        v = valid.astype(np.float64)
        w = ((v * _rotate180(v, xc, yc)) > 0.999).astype(np.float64)
    d = np.abs(img - _rotate180(img, xc, yc)) * w
    num, _ = _ap_sum(d, xc, yc, R)
    den, _ = _ap_sum(np.abs(img) * w, xc, yc, R)
    area, _ = _ap_sum(w, xc, yc, R)
    return num, den, area


def find_sky_boxes(sky_ok, box):
    """Centres (r, c) of box x box windows that contain only sky pixels."""
    ii = np.pad(np.cumsum(np.cumsum(sky_ok.astype(np.int32), 0), 1), ((1, 0), (1, 0)))
    H, W = sky_ok.shape
    out = []
    step = max(1, box // 2)
    for r in range(0, H - box + 1, step):
        for c in range(0, W - box + 1, step):
            if ii[r + box, c + box] - ii[r, c + box] - ii[r + box, c] + ii[r, c] == box * box:
                out.append((r + box // 2, c + box // 2))
    return np.array(out, dtype=float).reshape(-1, 2)


def sky_box_terms(lin, centres, box, near_rc, smooth_size, frac_xy=(0.0, 0.0)):
    """Per-pixel background terms from the sky box nearest to near_rc:
    asymmetry  sum|B - B180| / N  and smoothness  sum(B - B_s)+ / N.
    B is rotated about a point with the SAME sub-pixel offset as the
    galaxy's asymmetry centre, so both carry identical interpolation
    smoothing of the noise (otherwise A is biased low at low S/N)."""
    if centres is None or len(centres) == 0:
        return None
    j = int(np.argmin(np.hypot(centres[:, 0] - near_rc[0], centres[:, 1] - near_rc[1])))
    r, c = (int(v) for v in centres[j])
    B = np.asarray(lin[r - box // 2:r - box // 2 + box, c - box // 2:c - box // 2 + box], np.float64)
    # remove a least-squares plane: a box on a giant galaxy's halo carries a
    # light GRADIENT, which is perfectly antisymmetric under 180-degree
    # rotation and would be counted as 'noise asymmetry' (HCG 42a: A_bg 0.14)
    yy, xx = np.indices(B.shape, dtype=np.float64)
    G = np.column_stack([np.ones(B.size), xx.ravel(), yy.ravel()])
    coef, *_ = np.linalg.lstsq(G, B.ravel(), rcond=None)
    B = B - (G @ coef).reshape(B.shape)
    bc = (box - 1) / 2.0
    B180 = _rotate180(B, bc + frac_xy[0], bc + frac_xy[1])
    inner = np.zeros(B.shape, bool)
    inner[2:-2, 2:-2] = True                              # rotated samples all inside the box
    a = float(np.abs(B - B180)[inner].sum()) / inner.sum()
    k = max(1, int(smooth_size))
    Bs = ndimage.uniform_filter(B, size=k)
    s = float(np.clip(B - Bs, 0, None).sum()) / B.size
    return {'asym_per_px': a, 'smooth_per_px': s}


def gini_coefficient(x):
    x = np.sort(np.abs(np.asarray(x, np.float64)))
    n = x.size
    if n < 2 or x.mean() == 0:
        return float('nan')
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * x) / (x.mean() * n * (n - 1)))


def m20_moment(img, mask):
    ys, xs = np.nonzero(mask)
    f = np.asarray(img[ys, xs], np.float64)
    pos = f > 0
    ys, xs, f = ys[pos], xs[pos], f[pos]
    if f.size < 5:
        return float('nan')
    xc, yc = np.average(xs, weights=f), np.average(ys, weights=f)
    m = f * ((xs - xc) ** 2 + (ys - yc) ** 2)
    mtot = m.sum()
    if mtot <= 0:
        return float('nan')
    o = np.argsort(f)[::-1]
    cf = np.cumsum(f[o])
    k = int(np.searchsorted(cf, 0.2 * cf[-1])) + 1
    return float(np.log10(m[o][:k].sum() / mtot))


# ── neighbour-model run: smooth symmetric neighbour models ───────────────────────────
# Masking a neighbour's SEGMENT (below) removes its core, but not its halo:
# the wings that extend beyond its segment boundary into the galaxy being
# measured are a light gradient across that galaxy, which is antisymmetric
# under a 180-degree rotation and is counted as asymmetry (HCG 79a/b/c,
# HCG 95a, HCG 56d). Classical fix (cf. isophote-model subtraction in
# crowded fields): give every detected object a smooth, point-symmetric
# model -- the MEDIAN surface brightness in elliptical annuli (ellipse from
# its flux-weighted second moments), forced to decrease outwards, and cut
# where it reaches the sky -- and, when measuring object i, subtract the
# models of all OTHER objects. The median rejects the minority of an
# annulus that falls on a neighbour; a second pass refits every model on
# the image with the other models removed.
def _segment_ellipse(img, own):
    """Centre, axis ratio and PA of an object from the second moments of its
    HALF-LIGHT region (the brightest pixels of the segment holding 50 % of
    its flux, connected to the peak) -- a tidal tail or a shared envelope in
    the segment's outskirts must not drag the centre off the nucleus. The
    size a_seg is from the whole segment's area."""
    f = np.where(own, np.clip(img, 0, None), 0.0)
    q_area = float(own.sum())
    if f.sum() <= 0:
        f = own.astype(np.float64)
    v = np.sort(f[own])[::-1]
    t = v[min(int(np.searchsorted(np.cumsum(v), 0.5 * v.sum())), v.size - 1)]
    core = own & (f >= t)
    lab, n = ndimage.label(core)
    if n > 1:
        pk = np.unravel_index(int(np.argmax(ndimage.gaussian_filter(f, 1.0))), f.shape)
        core = lab == (lab[pk] if lab[pk] > 0 else 1 + int(np.argmax(ndimage.sum(f, lab, range(1, n + 1)))))
    ys, xs = np.nonzero(core)
    w = f[ys, xs] + 1e-30
    xc, yc = np.average(xs, weights=w), np.average(ys, weights=w)
    dx, dy = xs - xc, ys - yc
    cxx, cyy, cxy = (np.average(dx * dx, weights=w), np.average(dy * dy, weights=w),
                     np.average(dx * dy, weights=w))
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
    q = float(np.clip(math.sqrt(l2 / max(l1, 1e-6)), 0.2, 1.0))
    theta = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    a_seg = float(np.sqrt(q_area / (math.pi * q)))           # semi-major axis of the segment's area
    return xc, yc, q, theta, a_seg


def _object_model(img, seg, label, valid, ell, extent, max_half, add_back=None):
    """Median elliptical-annulus profile of one object -> (r0, c0, model).
    add_back = (r0, c0, model) is added to img inside the window first."""
    H, W = img.shape
    xc, yc, q, th, a_seg = ell
    half = int(min(max_half, max(8, math.ceil(extent * a_seg))))
    r0, r1 = max(0, int(yc) - half), min(H, int(yc) + half + 1)
    c0, c1 = max(0, int(xc) - half), min(W, int(xc) + half + 1)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    dx, dy = xx - xc, yy - yc
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    rr = np.hypot(u, v / q)
    win = np.asarray(img[r0:r1, c0:c1], np.float64)
    if add_back is not None:
        ar, ac, am = add_back
        y0, y1, x0, x1 = max(ar, r0), min(ar + am.shape[0], r1), max(ac, c0), min(ac + am.shape[1], c1)
        if y1 > y0 and x1 > x0:
            win = win.copy()
            win[y0 - r0:y1 - r0, x0 - c0:x1 - c0] += am[y0 - ar:y1 - ar, x0 - ac:x1 - ac]
    sub_seg = seg[r0:r1, c0:c1]
    use = valid[r0:r1, c0:c1] & ((sub_seg == 0) | (sub_seg == label)) & (rr <= half)
    edges = np.unique(np.concatenate([[0.0], np.arange(1.0, 6.0, 1.0),
                                      np.geomspace(6.0, max(half, 7.0), 40)]))
    b = np.digitize(rr, edges) - 1
    nb = len(edges) - 1
    ok = use & (b >= 0) & (b < nb)
    idx = np.arange(nb)
    cnt = np.bincount(b[ok], minlength=nb)[:nb]
    med = np.array(ndimage.median(win[ok], labels=b[ok], index=idx), float)
    med[cnt < 5] = np.nan
    rc = 0.5 * (edges[1:] + edges[:-1])
    good = np.isfinite(med)
    if good.sum() < 2:
        return None
    med = np.interp(rc, rc[good], med[good])
    # cut at the sky: first bin beyond the segment's own extent whose median <= 0
    beyond = np.nonzero((rc > a_seg) & (med <= 0))[0]
    if beyond.size:
        med[beyond[0]:] = 0.0
    med = np.minimum.accumulate(np.clip(med, 0, None))         # never brighter further out
    model = np.interp(rr, rc, med, right=0.0)
    model[rr > half] = 0.0
    return r0, c0, model.astype(np.float32)


def build_neighbour_models(lin, seg, labels, nodata, extent=4.0, max_half_frac=0.35, n_pass=8, damping=0.5):
    """Smooth symmetric model of every listed segment. Returns (total, own)
    where total is the full-frame sum and own[label] = (r0, c0, model).

    Pass 1 fits every object on the raw image, so each model still holds
    some of its neighbours' halos; each later pass refits every object with
    all OTHER current models removed. Plain re-fitting oscillates (pass 2
    over-corrects what pass 1 over-estimated), so successive passes are
    blended (damped fixed-point iteration, new = damping * old +
    (1 - damping) * refit), which converges in a few passes."""
    H, W = lin.shape
    valid = ~nodata if nodata is not None else np.ones(lin.shape, bool)
    max_half = int(max(16, max_half_frac * max(H, W)))
    objs = ndimage.find_objects(seg)
    ells = {}
    for lb in labels:
        sl = objs[lb - 1] if 0 < lb <= len(objs) else None
        if sl is None:
            continue
        own = seg[sl] == lb
        if own.sum() < 3:
            continue
        xc, yc, q, th, a = _segment_ellipse(np.asarray(lin[sl], np.float64), own)
        ells[lb] = (xc + sl[1].start, yc + sl[0].start, q, th, a)
    total = np.zeros(lin.shape, np.float32)
    own_models = {}
    for it in range(n_pass):
        base = lin if it == 0 else lin - total          # later passes: every other model removed ...
        new_own = {}
        for lb, ell in ells.items():
            res = _object_model(base, seg, lb, valid, ell, extent, max_half,
                                add_back=own_models.get(lb))       # ... except this object's own
            if res is None:
                continue
            if it > 0 and lb in own_models:
                r0, c0, m = res
                m = (damping * own_models[lb][2] + (1.0 - damping) * m).astype(np.float32)
                res = (r0, c0, m)
            new_own[lb] = res
        total = np.zeros(lin.shape, np.float32)
        for r0, c0, m in new_own.values():
            total[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] += m
        own_models = new_own
    return total, own_models


def measure_morphology(lin, seg, label, rp, sky_sigma, sky_centres, sky_box, nodata=None,
                       nb_total=None, nb_own=None, debug=False, a_max_clipped=None, a_min_rp=0.5):
    """Non-parametric morphology of one segment on LINEAR sky-subtracted
    flux. Other segments and no-data pixels are zeroed (= sky level).
    With nb_total / nb_own (build_neighbour_models), the smooth models of
    all OTHER objects are subtracted first (neighbour-model run).
    aperture/disc experiments aperture recovery: with a_max_clipped set, an object whose
    1.5 r_p aperture is cut by the frame / no-data edge by more than that
    share has A measured inside the largest concentric circle R_A >= a_min_rp
    x r_p that is clipped by at most that share (C, S, Gini, M20 and the
    asymmetry centre are unchanged). Returned as a_clipped_frac (clipped
    share of the aperture A was measured in) and r_a_over_rp.
    Returns a dict (NaN where a quantity could not be measured)."""
    H, W = lin.shape
    minr, minc, maxr, maxc = rp.bbox
    half = int(max(maxr - minr, maxc - minc))
    out = {'C': np.nan, 'A': np.nan, 'A_raw': np.nan, 'A_bg_local': np.nan, 'S': np.nan,
           'S_raw': np.nan, 'S_bg_local': np.nan, 'gini': np.nan, 'm20': np.nan,
           'r_petro_px': np.nan, 'r20': np.nan, 'r80': np.nan, 'petro_ok': False,
           'sn_per_pixel': np.nan, 'aperture_clipped_frac': np.nan, 'unrecoverable_frac': np.nan}
    for attempt in range(2):
        cy0, cx0 = rp.centroid
        r0, r1 = max(0, int(cy0) - half), min(H, int(cy0) + half + 1)
        c0, c1 = max(0, int(cx0) - half), min(W, int(cx0) + half + 1)
        sub_seg = seg[r0:r1, c0:c1]
        img = np.asarray(lin[r0:r1, c0:c1], np.float64).copy()
        if nb_total is not None:
            img -= nb_total[r0:r1, c0:c1]
            om = (nb_own or {}).get(label)
            if om is not None:
                ar, ac, am = om
                y0, y1, x0, x1 = max(ar, r0), min(ar + am.shape[0], r1), max(ac, c0), min(ac + am.shape[1], c1)
                if y1 > y0 and x1 > x0:
                    img[y0 - r0:y1 - r0, x0 - c0:x1 - c0] += am[y0 - ar:y1 - ar, x0 - ac:x1 - ac]
        bad = (sub_seg != 0) & (sub_seg != label)
        if nodata is not None:
            bad |= nodata[r0:r1, c0:c1]
        img[bad] = 0.0
        own = sub_seg == label
        if not own.any() or img[own].sum() <= 0:
            return out
        wpos = np.clip(img, 0, None) * own
        yy, xx = np.indices(img.shape)
        xc = float((xx * wpos).sum() / wpos.sum())
        yc = float((yy * wpos).sum() / wpos.sum())
        r_max = float(min(half, max(img.shape)))
        rp_c, ok = petrosian_radius(img, xc, yc, r_max)
        R = 1.5 * rp_c
        if R <= 0.9 * half or attempt == 1 or half >= max(H, W):
            break
        half = int(np.ceil(R / 0.9)) + 2                 # aperture larger than cutout: enlarge once
    # asymmetry-minimising centre (Nelder-Mead from the centroid, Conselice 2000)
    from scipy.optimize import minimize
    valid = ~bad

    def _fa(c):
        n, d, _ = _asym_raw(img, c[0], c[1], R, valid)
        return n / d if d > 0 else 1e9
    # two starting points: flux centroid and the smoothed peak (the centroid
    # of a galaxy truncated by the frame edge lies far from its nucleus)
    smp = ndimage.gaussian_filter(np.where(own, img, 0.0), 1.0)
    kp = int(np.argmax(smp))
    starts = [(xc, yc), (float(kp % img.shape[1]), float(kp // img.shape[1]))]
    best = None
    for sx, sy in starts:
        try:
            res = minimize(_fa, x0=[sx, sy], method='Nelder-Mead',
                           options={'xatol': 0.05, 'fatol': 1e-5, 'maxiter': 200})
            cx_, cy_ = (float(v) for v in res.x)
            if np.hypot(cx_ - sx, cy_ - sy) > 0.5 * rp_c:      # wandered off: keep the start
                cx_, cy_ = sx, sy
        except Exception:
            cx_, cy_ = sx, sy
        val = _fa((cx_, cy_))
        if best is None or val < best[0]:
            best = (val, cx_, cy_)
    xa, ya = best[1], best[2]
    num, den, area = _asym_raw(img, xa, ya, R, valid)
    # share of the 1.5 r_p aperture (about the asymmetry centre) that is NOT
    # this object's valid data: no-data / outside the frame (truncation),
    # and separately the share covered by neighbours
    yy, xx = np.indices(img.shape)
    inap_full = np.pi * R * R
    inap = np.hypot(xx - xa, yy - ya) <= R
    nd_sub = (nodata[r0:r1, c0:c1] if nodata is not None else np.zeros(img.shape, bool))
    missing_frac = float(np.clip(1.0 - (inap & ~nd_sub).sum() / max(inap_full, 1.0), 0, 1))
    neighbour_frac = float((inap & ~nd_sub & bad).sum() / max(inap_full, 1.0))
    # ... of which UNRECOVERABLE: missing pixels whose 180-degree partner is
    # missing too (symmetric filling recovers the rest -- HCG 7b, cut by the
    # frame edge near its nucleus, still measures as a clean S0)
    Ri = int(np.ceil(R)) + 1
    YY, XX = np.mgrid[int(round(ya)) - Ri:int(round(ya)) + Ri + 1, int(round(xa)) - Ri:int(round(xa)) + Ri + 1]

    def _valid_at(Y, X):
        ok = (Y >= 0) & (Y < img.shape[0]) & (X >= 0) & (X < img.shape[1])
        v = np.zeros(Y.shape, bool)
        v[ok] = ~nd_sub[Y[ok], X[ok]]
        return v
    ap2 = np.hypot(XX - xa, YY - ya) <= R
    v_here = _valid_at(YY, XX)
    v_mirr = _valid_at(np.rint(2 * ya - YY).astype(int), np.rint(2 * xa - XX).astype(int))
    unrecoverable_frac = float((ap2 & ~v_here & ~v_mirr).sum() / max(ap2.sum(), 1))
    if den <= 0:
        return out
    # masked pixels whose 180-degree partner is valid are filled by symmetry
    # for C / S / Gini / M20 (a chip gap through a galaxy otherwise reads as
    # a dark lane); rp_c is re-measured on the filled image.
    v180 = _rotate180(valid.astype(np.float64), xa, ya) > 0.999
    fill = bad & v180
    if fill.any():
        img = np.where(fill, _rotate180(img, xa, ya), img)
        rp_c, ok = petrosian_radius(img, xa, ya, max(r_max, R))
        R = 1.5 * rp_c
        num, den, area = _asym_raw(img, xa, ya, R, valid)
        if den <= 0:
            return out
    bad = bad & ~fill
    # aperture/disc experiments: clipped share of the aperture actually used for A, and recovery
    a_clipped, r_a_over_rp = missing_frac, 1.5
    if a_max_clipped is not None and rp_c > 0:
        rr_c = np.hypot(xx - xa, yy - ya)

        def _clipped(Rq):
            return float(np.clip(1.0 - ((rr_c <= Rq) & ~nd_sub).sum() / max(np.pi * Rq * Rq, 1.0), 0, 1))
        a_clipped = _clipped(R)
        if a_clipped > a_max_clipped:
            for Rq in np.linspace(R, a_min_rp * rp_c, 40)[1:]:
                cq = _clipped(Rq)
                if cq <= a_max_clipped:
                    num, den, area = _asym_raw(img, xa, ya, Rq, valid)
                    if den <= 0:
                        return out
                    a_clipped, r_a_over_rp = cq, Rq / rp_c
                    break
    k_s = max(1, int(round(0.25 * rp_c)))
    bgt = sky_box_terms(lin, sky_centres, sky_box, (ya + r0, xa + c0), k_s,
                        frac_xy=(xa - round(xa), ya - round(ya)))
    if bgt is None:                                       # Gaussian-noise expectation, 2 sigma / sqrt(pi)
        bgt = {'asym_per_px': 2.0 * sky_sigma / np.sqrt(np.pi), 'smooth_per_px': 0.0}
    A_raw = num / den
    A_bg = bgt['asym_per_px'] * area / den
    # concentration on the curve of growth inside 1.5 r_p
    f_tot, _ = _ap_sum(img, xa, ya, R)
    radii = np.linspace(0.5, R, 120)
    cog = np.array([_ap_sum(img, xa, ya, r)[0] for r in radii]) / max(f_tot, 1e-30)
    cog = np.maximum.accumulate(cog)
    r20 = float(np.interp(0.2, cog, radii)) if cog[-1] >= 0.2 else np.nan
    r80 = float(np.interp(0.8, cog, radii)) if cog[-1] >= 0.8 else np.nan
    C = 5.0 * np.log10(r80 / r20) if (np.isfinite(r20) and np.isfinite(r80) and r20 > 0) else np.nan
    # smoothness (clumpiness): positive residual from a 0.25 r_p boxcar, 0.25-1.5 r_p annulus
    diff = np.clip(img - ndimage.uniform_filter(img, size=k_s), 0, None)
    s_num, s_area = _ap_sum(diff, xa, ya, R, 0.25 * rp_c)
    s_den, _ = _ap_sum(img, xa, ya, R, 0.25 * rp_c)
    if s_den > 0:
        S_raw = s_num / s_den
        S_bg = bgt['smooth_per_px'] * s_area / s_den
    else:
        S_raw = S_bg = np.nan
    # Gini / M20 on the Lotz et al. (2004) segmentation map
    sm = ndimage.gaussian_filter(img, sigma=max(rp_c / 5.0, 0.5))
    mu_an, a_an = _ap_sum(sm, xa, ya, 1.25 * rp_c, 0.8 * rp_c)
    gmask = sm >= (mu_an / a_an if a_an > 0 else np.inf)
    lab, nl = ndimage.label(gmask)
    ic, jc = int(round(ya)), int(round(xa))
    if nl and 0 <= ic < lab.shape[0] and 0 <= jc < lab.shape[1] and lab[ic, jc] > 0:
        gmask = lab == lab[ic, jc]
    gmask &= ~bad
    if gmask.sum() >= 5:
        out['gini'] = gini_coefficient(img[gmask])
        out['m20'] = m20_moment(img, gmask)
        out['sn_per_pixel'] = float(np.mean(img[gmask]) / max(sky_sigma, 1e-30))
    out['aperture_clipped_frac'] = missing_frac
    out['a_clipped_frac'] = a_clipped
    out['r_a_over_rp'] = r_a_over_rp
    out['unrecoverable_frac'] = unrecoverable_frac
    out['neighbour_frac'] = neighbour_frac
    out.update({'C': C, 'A': A_raw - A_bg, 'A_raw': A_raw, 'A_bg_local': A_bg,
                'S': (S_raw - S_bg) if np.isfinite(S_raw) else np.nan, 'S_raw': S_raw, 'S_bg_local': S_bg,
                'r_petro_px': rp_c, 'r20': r20, 'r80': r80, 'petro_ok': bool(ok),
                'centre_xy': (xa + c0, ya + r0)})
    if debug:
        out['_debug'] = {'img': img, 'valid': valid, 'xa': xa, 'ya': ya, 'R': R, 'r0': r0, 'c0': c0,
                         'R_A': r_a_over_rp * rp_c, 'rp_c': rp_c, 'nd_sub': nd_sub, 'bad': bad}
    return out


def detect_linear(lin, nodata, p, log=print):
    """development stage detection on the LINEAR common-resolution image (used whenever
    the image was resampled): normalized-convolution smoothing, a FIXED
    significance threshold (k_detect x the smoothed sky sigma, sigma-clipped
    over all valid pixels) with a minimum area, then multi-threshold
    deblending (SExtractor-style; photutils.deblend_sources): a branch of the
    isophote tree becomes its own object when it holds > deblend_contrast of
    its parent's flux. Replaces the adaptive 'foreground-fraction' k search,
    which on halo-dominated frames climbed to k ~ 11-80 sigma, shattered the
    halo into hundreds of cores and lost every faint galaxy.
    Returns (labels, smoothed_sigma, smoothed_image)."""
    from photutils.segmentation import detect_sources, deblend_sources
    valid = ~nodata
    sm, den = _normalized_smooth(np.asarray(lin, np.float32), valid, p['detect_smooth_px'])
    sm = np.where(valid & (den > 0.2), sm, 0.0).astype(np.float64)
    v = sm[valid][::max(1, int(valid.sum() // 2_000_000))]
    _, med, std = sigma_clipped_stats(v, sigma=3.0, maxiters=10)
    sm = sm - med
    thr = p['k_detect'] * std
    segm = detect_sources(sm, thr, n_pixels=int(p['min_area_px']), mask=nodata)
    if segm is None:
        log(f'  detection (linear): k={p["k_detect"]:.1f} sigma -> no sources')
        return np.zeros(lin.shape, np.int32), std, sm
    n0 = segm.nlabels
    # two-tier deblending: small blends with the SExtractor default contrast;
    # LARGE parents (resolved galaxies, >= deblend_big_arcsec2) with a higher
    # contrast, so a spiral's HII regions / arms (each a few % of its light)
    # stay attached while separate member galaxies (>= ~10 % each) still
    # split. With one low contrast HCG 7d was shredded into ~12 'galaxies'
    # and its own arms were masked out of its CAS measurement.
    big_px = float(p['deblend_big_arcsec2']) / float(p['_scale']) ** 2
    areas = segm.areas
    big = [int(l) for l, a in zip(segm.labels, areas) if a >= big_px]
    small = [int(l) for l, a in zip(segm.labels, areas) if a < big_px]
    out = np.zeros(lin.shape, np.int32)
    nxt = 0
    # Large parents: after deblending, a child that holds no local maximum of
    # the heavily smoothed light (sigma = deblend_big_smooth_arcsec) is a
    # fragment (arm, HII knot) and is re-attached to the neighbour it shares
    # the longest boundary with; children WITH such a maximum (a nucleus)
    # stay separate. HCG 7c's arms each held > 5 % of its light and were
    # split off; deblending directly on the smoothed image instead merged the
    # interacting pair HCG 90b/90d -- the nucleus test keeps both apart.
    sig_big = max(1.0, float(p['deblend_big_smooth_arcsec']) / float(p['_scale']))
    sm_big, _ = _normalized_smooth(np.where(valid, sm, 0.0).astype(np.float32), valid, sig_big)
    sm_big = np.where(valid, sm_big, 0.0).astype(np.float64)
    n_frag = 0
    for labs, con, is_big in [(small, p['deblend_contrast'], False), (big, p['deblend_contrast_big'], True)]:
        if not labs:
            continue
        sub = segm.copy()
        sub.keep_labels(labs)
        db = deblend_sources(sm, sub, n_pixels=int(p['min_area_px']), n_levels=int(p['deblend_nlevels']),
                             contrast=float(con), mode='exponential', progress_bar=False)
        d = db.data.astype(np.int32)
        if is_big:
            d, n_frag = merge_children_without_nucleus(d, sub.data > 0, sm_big, scale=float(p['_scale']))
        out[d > 0] = d[d > 0] + nxt
        nxt += int(d.max())
    log(f'  detection (linear): k={p["k_detect"]:.1f} sigma of the {p["detect_smooth_px"]:.1f}px-smoothed sky, '
        f'min area {int(p["min_area_px"])} px -> {n0} sources ({len(big)} large), {len(np.unique(out)) - 1} '
        f'after deblending (contrast {p["deblend_contrast"]} / {p["deblend_contrast_big"]} for large; '
        f'{n_frag} nucleus-less fragment(s) of large galaxies re-attached)')
    return out, std, sm


def merge_children_without_nucleus(d, parent_mask, sm_big, scale):
    """Re-attach deblended children that contain no true local maximum of
    the smoothed image sm_big to their longest-boundary neighbour (union-find,
    repeated until stable). Returns (labels, n_merged)."""
    md = max(2, int(round(1.0 / scale)))
    pk = peak_local_max(np.where(parent_mask, sm_big, -np.inf), min_distance=md, exclude_border=False,
                        threshold_abs=0.0)
    has_nuc = set(int(v) for v in d[pk[:, 0], pk[:, 1]] if v > 0) if len(pk) else set()
    labs = [int(v) for v in np.unique(d) if v > 0]
    frag = [l for l in labs if l not in has_nuc]
    if not frag or not has_nuc:
        return d, 0
    n = 0
    for _ in range(10):
        bnd = []
        for la, lb in [(d[:, :-1], d[:, 1:]), (d[:-1, :], d[1:, :])]:
            m = (la != lb) & (la > 0) & (lb > 0)
            if m.any():
                bnd.append(pd.DataFrame({'a': la[m], 'b': lb[m]}))
        if not bnd:
            break
        b = pd.concat(bnd)
        b = pd.concat([b, b.rename(columns={'a': 'b', 'b': 'a'})])
        cnt = b.groupby(['a', 'b']).size().reset_index(name='n')
        cnt = cnt[cnt['a'].isin([l for l in np.unique(d) if l > 0 and int(l) not in has_nuc])]
        if cnt.empty:
            break
        target = cnt.sort_values('n', ascending=False).drop_duplicates('a').set_index('a')['b']
        lut = np.arange(int(d.max()) + 1, dtype=np.int32)
        changed = 0
        for a, t in target.items():
            if lut[int(a)] == int(a):
                lut[int(a)] = int(t)
                changed += 1
        if not changed:
            break
        d = lut[d]
        n += changed
    return d, n


def merge_across_mask(seg, mask, min_frac):
    """Merge every pair of 4-adjacent segments whose shared boundary lies
    for >= min_frac on `mask` (here: inpainted chip-gap pixels, where any
    valley found by the deblender is an interpolation artefact, not data).
    Merged groups take the label of their largest member. Returns
    (seg, n_merged)."""
    pairs = []
    for la, lb, ma, mb in [(seg[:, :-1], seg[:, 1:], mask[:, :-1], mask[:, 1:]),
                           (seg[:-1, :], seg[1:, :], mask[:-1, :], mask[1:, :])]:
        m = (la != lb) & (la > 0) & (lb > 0)
        if m.any():
            pairs.append(pd.DataFrame({'a': np.minimum(la[m], lb[m]), 'b': np.maximum(la[m], lb[m]),
                                       'g': (ma[m] | mb[m]).astype(np.int32)}))
    if not pairs:
        return seg, 0
    st = pd.concat(pairs).groupby(['a', 'b'])['g'].agg(['mean', 'size'])
    st = st[(st['mean'] >= min_frac) & (st['size'] >= 3)]
    if st.empty:
        return seg, 0
    area = np.bincount(seg.ravel())
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b) in st.index:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            if area[ra] < area[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
    lut = np.arange(area.size, dtype=seg.dtype)
    n = 0
    for x in list(parent):
        r = find(x)
        if r != x:
            lut[x] = r
            n += 1
    return lut[seg], n


def point_source_test(seg, lin, nodata, p):
    """Star / galaxy separation on linear flux by CORE CONCENTRATION:
        ratio = F(r < r_core) / F(r < r_outer)
    about the object's peak, after subtracting the median of a local
    annulus (other objects and no-data excluded). A point source puts nearly
    all of its light inside the PSF core; a resolved galaxy does not. The
    ratio is immune to diffraction spikes (they enlarge the segment, not the
    core), to saturation plateaus (still inside 0.3") and to a halo pedestal
    (removed by the annulus) -- the three ways the earlier stage test
    (flux within psf_radius / total segment flux) failed.
    Returns ({label: ratio}, set_of_star_labels)."""
    sm = cv2.GaussianBlur(np.asarray(lin, np.float32), (0, 0), 0.7)
    r_core, r_out = float(p['psf_radius_px']), float(p['star_outer_px'])
    a_in, a_out = r_out + 1.0, r_out + 4.0
    ratios, stars = {}, set()
    H, W = lin.shape
    for rp in measure.regionprops(seg):
        minr, minc, maxr, maxc = rp.bbox
        sub = seg[minr:maxr, minc:maxc] == rp.label
        k = int(np.argmax(np.where(sub, sm[minr:maxr, minc:maxc], -np.inf)))
        py, px = minr + k // (maxc - minc), minc + k % (maxc - minc)
        r0, r1 = max(0, py - int(a_out) - 1), min(H, py + int(a_out) + 2)
        c0, c1 = max(0, px - int(a_out) - 1), min(W, px + int(a_out) + 2)
        cut = np.asarray(lin[r0:r1, c0:c1], np.float64)
        yy, xx = np.indices(cut.shape)
        rr = np.hypot(yy - (py - r0), xx - (px - c0))
        ring = (rr >= a_in) & (rr <= a_out) & ~nodata[r0:r1, c0:c1]
        ss = seg[r0:r1, c0:c1]
        ring &= (ss == 0) | (ss == rp.label)
        bg = float(np.median(cut[ring])) if ring.sum() >= 8 else 0.0
        f_core, _ = _ap_sum(cut - bg, px - c0, py - r0, r_core)
        f_out, _ = _ap_sum(cut - bg, px - c0, py - r0, r_out)
        ratio = f_core / f_out if f_out > 0 else 0.0
        ratios[rp.label] = round(float(ratio), 4)
        if ratio > p['star_core_ratio'] and rp.area < p['star_max_area_arcsec2'] / p['_scale'] ** 2:
            stars.add(rp.label)
    return ratios, stars


# ════════════════════════════════════════════════════════════════════════
# 8. PER-JOB PIPELINE
# ════════════════════════════════════════════════════════════════════════
def _cr_reject_for(rec, p):
    if p['cr_reject'] == 'auto':
        # development stage: a drizzled file built from ONE exposure is not CR-combined
        # (SQ WFC3 iacr02ywq_drc: ASN_ID=NONE, CRSPLIT=1, CRCORR=OMIT ->
        # 0.46 % sharp outliers vs 0.15-0.18 % in true multi-exposure drizzles)
        return rec.product in SINGLE_EXPOSURE_PRODUCTS or rec.n_exp == 1
    return bool(p['cr_reject'])


def _save(fig, path, show):
    fig.savefig(path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def run_image_job(job, cfg, out_root, catalog=None, log=print):
    """Full pipeline for one image job. Returns a results dict."""
    p = params_for(job, cfg)
    det = job.detect
    field_slug = _slug(job.field_name)
    prefix = f'{field_slug}_{_slug(det.filter)}' if len(job.bands) == 1 else \
        f'{field_slug}_{_slug("_".join(b.filter for b in job.bands))}'
    out_dir = os.path.join(out_root, field_slug, _slug(job.label))
    os.makedirs(out_dir, exist_ok=True)
    title = f'{job.field_name} — {det.telescope} {det.instrument}' \
            f'{"/" + det.detector if det.detector else ""} {"+".join(b.filter for b in job.bands)}'
    log(f"\n{'=' * 78}\n{title}\n  file(s): {', '.join(os.path.basename(b.path) for b in job.bands)}"
        f"\n{'=' * 78}")

    # development stage: artefact rejection at native resolution, then block-average to
    # the common working scale (see section 7b). f = 1 reproduces earlier stage.
    f = bin_factor(det.pix_scale_arcsec, p.get('work_scale_arcsec'))
    scale = det.pix_scale_arcsec * f
    if f > 1:
        # structural sizes were tuned in native pixels of ~0.05"/px frames;
        # keep their PHYSICAL size on the coarser grid (noise-related values --
        # k thresholds, min_area as a 'resolved enough for CAS' cut -- stay).
        for key, arcsec, floor in [('grow_smooth_sigma', 'grow_smooth_arcsec', 1.0),
                                   ('smooth_sigma', 'smooth_arcsec', 1.0),
                                   ('min_distance', 'min_distance_arcsec', 3),
                                   ('psf_radius_px', 'psf_radius_arcsec', 1.5)]:
            v = max(floor, p[arcsec] / scale)
            p[key] = int(round(v)) if isinstance(floor, int) else float(v)
        p['morph_kernel'] = max(3, 2 * int(round(p['morph_kernel_arcsec'] / scale / 2)) + 1)

    def _prep(rec, nd_native=None):
        r = load_sci(rec)
        nd = (build_nodata_mask(r, load_ext(rec, 'WHT'), p, log) if nd_native is None
              else nd_native | ~np.isfinite(r))
        d, n_dq, n_cr = reject_bad_pixels(r, dq=load_ext(rec, 'DQ'), cr_reject=_cr_reject_for(rec, p),
                                          cr_method=p['cr_method'], exptime=rec.exptime)
        if n_cr:
            log(f'    {rec.filter}: {n_cr} cosmic-ray px cleaned ({p["cr_method"]}; '
                f'{rec.n_exp or "?"} input exposure(s))')
        del r
        b, ndb = block_reduce_masked(d, nd, f)
        if (~ndb).any():
            b[ndb] = np.median(b[~ndb])
        return b, ndb, nd

    raw, nodata, nd_native = _prep(det)
    H, W = raw.shape
    box = int(min(p['bg_box'], max(p['bg_box_min'], min(H, W) // p['bg_min_boxes_per_axis'])))
    log(f'  frame {H}x{W} px @ {scale:.4f}"/px (native {det.pix_scale_arcsec:.4f}"/px, binned {f}x{f}), '
        f'product={det.product}, no-data {nodata.mean():.1%}, background box {box}px, '
        f'CR rejection {"on" if _cr_reject_for(det, p) else "off"}')

    lin = raw.astype(np.float64)
    coarse = int(min(H, W) * p['coarse_box_frac'])
    sky_model = p['sky_model']
    if sky_model == 'auto' and det.product not in COMBINED_PRODUCTS:
        # Only drizzled/stage-3 products (_drz/_drc/_i2d) are sky-subtracted by
        # the calibration pipeline; single-exposure frames (_crf/_cal/_flt/_flc)
        # can carry real detector-level gradients a coarse sky cannot follow
        # (seen on JWST F150W), so they always use the local model.
        sky_model = 'local'
        log(f'  sky model: local ({det.product} product is not pipeline sky-subtracted)')
    elif sky_model == 'auto':
        sky_model, _ = select_sky_model(lin, nodata, box, coarse, p['sky_select_tol'], log,
                                        candidates=('flat', 'coarse'),
                                        tolerances={'flat': p['flat_select_tol']})
    sky_kw = sky_kwargs(sky_model, box, coarse)
    linear_data = lin - masked_background(lin, nodata_mask=nodata, **sky_kw).background
    del lin

    proc = {}
    extra_lin = {}
    want_bands = {str(x).upper() for x in (p.get('measure_bands') or [])}
    for b in job.bands:
        if b is det:
            braw, bnd = raw, nodata
        else:
            braw, bnd, _ = _prep(b, nd_native)
            bnd = bnd | nodata
            if b.filter.upper() in want_bands:
                bl = braw.astype(np.float64)
                extra_lin[b.filter] = bl - masked_background(bl, nodata_mask=bnd, **sky_kw).background
                del bl
        proc[b.filter] = process_channel(braw, dq=None, cr_reject=False, box_size=box,
                                         h=p["nlm_h"], nodata_mask=bnd, log=log, sky_kw=sky_kw)
    del nd_native
    detect_proc = proc[det.filter]
    detect_u8 = (np.clip(detect_proc, 0, 1) * 255).astype(np.uint8)
    chans = [proc[b.filter] for b in job.bands][-3:]           # three reddest, blue -> red
    if len(chans) == 1:
        display_rgb = np.dstack([chans[0]] * 3)
    elif len(chans) == 2:
        display_rgb = np.dstack([chans[1], 0.5 * (chans[0] + chans[1]), chans[0]])
    else:
        display_rgb = np.dstack([chans[2], chans[1], chans[0]])

    del raw

    # Only the large connected no-data footprint (mosaic outline, chip gaps)
    # is buffered / used for edge rejection; isolated flagged speckles
    # (saturated cores, bad columns) stay masked but are not dilated.
    footprint = nodata_footprint(nodata, min_area=(box * box) // 4)
    buf = p['nodata_buffer_px']
    if buf == 'auto':
        # development stage: physical width (tuned as 50 native px ~ 2.5" in earlier stage), so
        # it does not grow to ~10" -- and cut into galaxies beside an ACS chip
        # gap (HCG 31, HCG 42) -- when working on a coarser grid.
        buf = (max(1, int(round(p['nodata_buffer_arcsec'] / scale)))
               if footprint.mean() > p['nodata_buffer_trigger_frac'] else 0)
    nodata_eff, footprint_eff = nodata, footprint
    if buf:
        cand = cv2.dilate(footprint.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(buf) + 1,) * 2)) > 0
        if (cand | nodata).mean() > p['nodata_buffer_max_frac']:
            log(f'  no-data buffer {buf}px skipped (would cover {(cand | nodata).mean():.1%} of frame)')
        else:
            footprint_eff = cand
            nodata_eff = nodata | cand
            log(f'  no-data footprint buffered by {buf}px ({nodata.mean():.1%} -> {nodata_eff.mean():.1%})')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p['morph_kernel'],) * 2)
    star_ratios = {}
    if f > 1:
        # development stage linear-flux path (see detect_linear / point_source_test)
        p['_scale'] = scale
        # Bridge THIN no-data structures (ACS/WFPC2 chip gaps + their buffer)
        # for detection only, so a galaxy straddling a gap stays one segment
        # (HCG 42a's halo beyond the gap became 3 spurious 'galaxies').
        # Thin = no-data removed by an opening of radius gap_bridge_arcsec;
        # the exterior frame outline survives the opening. Gap pixels are
        # interpolated by normalized convolution; measurement still treats
        # them as missing (symmetric masking in measure_morphology).
        r_thin = max(3, int(round(p['gap_bridge_arcsec'] / scale)))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_thin + 1,) * 2)
        exterior = cv2.morphologyEx(nodata_eff.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0
        thin = nodata_eff & ~exterior                     # chip gap + its buffer strip
        footprint_eff = footprint_eff & exterior          # edge test: frame outline only
        det_img, det_mask = linear_data, nodata_eff
        if thin.any():
            # Telea (2004) fast-marching inpainting -- continues the gradient
            # across the gap instead of sagging like a smoothing fill
            lo, hi = np.percentile(linear_data[~nodata_eff], [0.1, 99.99])
            src = np.clip(np.where(nodata_eff, 0.0, linear_data), lo, hi).astype(np.float32)
            det_img = cv2.inpaint(src, thin.astype(np.uint8), max(3, r_thin // 2), cv2.INPAINT_TELEA)
            det_img = np.where(thin, det_img, linear_data)
            det_mask = nodata_eff & ~thin
            log(f'  chip-gap bridging: {thin.mean():.1%} of frame inpainted (Telea) for detection only')
        labeled, sm_sigma, _ = detect_linear(det_img, det_mask, p, log=log)
        del det_img
        if thin.any() and labeled.max():
            labeled, n_m = merge_across_mask(labeled, thin, p['gap_merge_frac'])
            if n_m:
                log(f'  {n_m} deblended split(s) along the inpainted gap re-joined')
            # objects living mostly on inpainted pixels are gap-edge artefacts
            n_all = np.bincount(labeled.ravel())
            n_thin = np.bincount(labeled[thin], minlength=n_all.size)
            drop = np.nonzero((n_thin > p['gap_object_max_frac'] * n_all) & (np.arange(n_all.size) > 0))[0]
            if drop.size:
                labeled[np.isin(labeled, drop)] = 0
                log(f'  {drop.size} object(s) lying > {p["gap_object_max_frac"]:.0%} on the inpainted gap removed')
        fg = labeled > 0
        n_cores, k_sigma, fg_frac = int(labeled.max()), float(p['k_detect']), float(fg.mean())
        star_ratios, star_set = point_source_test(labeled, linear_data, nodata_eff, p)
        kept = [rp for rp in measure.regionprops(labeled) if rp.area >= p['min_area_px']]
        sep = {'n_raw': n_cores, 'kept': kept, 'seg_mask': labeled, 'lbl': len(kept),
               'star_labels': [rp.label for rp in kept if rp.label in star_set],
               'galaxy_labels': [rp.label for rp in kept if rp.label not in star_set],
               'obj_info': [{'label': rp.label, 'area': rp.area, 'center_frac': star_ratios.get(rp.label),
                             'centroid_r': round(rp.centroid[0], 1), 'centroid_c': round(rp.centroid[1], 1),
                             'obj_type': 'star' if rp.label in star_set else 'galaxy'} for rp in kept]}
        n_gal_cores = len(sep['galaxy_labels'])
        seg_labels = labeled
        bkg = None
    else:
        fg, ws, bkg, k_sigma, fg_frac = background_relative_watershed(detect_u8, p, box, nodata_mask=nodata_eff,
                                                                       sky_kw=sky_kw)
        n_cores = int(ws.max())
        log(f'  detection: k={k_sigma:.2f} sigma, {n_cores} cores, foreground {fg_frac:.2%}')
        if fg_frac > p['fg_frac_hi']:
            log(f'  ! foreground fraction above the {p["fg_frac_hi"]:.1%} target even at the strictest k')
        labeled = clean_labeled_mask(ws, kernel)
        sep = separate_stars_galaxies(labeled, detect_u8, bkg.background, p['min_area_px'],
                                      p['psf_radius_px'], p['conc_thresh'], p['area_max_px'])
        n_gal_cores = len(sep['galaxy_labels'])
        seg_labels = labeled
        if p['grow_segments'] and sep['galaxy_labels']:
            core_area = sum(rp.area for rp in sep['kept'] if rp.label in set(sep['galaxy_labels']))
            seg_labels, sep = grow_segments(labeled, sep, detect_u8.astype(np.float64) - bkg.background,
                                            nodata_eff, footprint_eff, fg, p, kernel, log=log)
            grown_area = sum(rp.area for rp in sep['kept'] if rp.label in set(sep['galaxy_labels']))
            log(f'  galaxy area {core_area} -> {grown_area} px ({grown_area / max(core_area, 1):.1f}x)')
    if p.get('debug_dump_dir'):
        os.makedirs(p['debug_dump_dir'], exist_ok=True)
        np.savez_compressed(os.path.join(p['debug_dump_dir'], f'{prefix}_debug.npz'),
                            labeled=labeled, seg_labels=seg_labels, detect_u8=detect_u8,
                            linear=np.asarray(linear_data, np.float32), nodata=nodata_eff, fg=fg,
                            kept=np.array([rp.label for rp in sep['kept']]),
                            stars=np.array(sep['star_labels']), gals=np.array(sep['galaxy_labels']))
    edge = (reject_edge_galaxies(sep, seg_labels, footprint_eff, linear_data, p['edge_flux_frac'],
                                 edge_px=max(1, int(round(p['edge_px_arcsec'] / scale))))
            if p['reject_edge_objects'] else [])
    log(f'  area filter (>= {p["min_area_px"]} px): {sep["lbl"]} objects -> {len(sep["star_labels"])} stars, '
        f'{len(sep["galaxy_labels"])} galaxies' + (f', {len(edge)} edge-truncated removed' if edge else ''))
    if p['print_obj_diagnostics'] and sep['obj_info']:
        od = pd.DataFrame(sep['obj_info']).sort_values('area', ascending=False)
        log('    ' + od[['label', 'area', 'center_frac', 'obj_type']].to_string(index=False).replace('\n', '\n    '))

    fwhm, fwhm_std, fwhm_list = measure_target_fwhm(sep['kept'], sep['star_labels'], linear_data,
                                                    max(1.0, p['assumed_fwhm_px'] / f))
    log(f'  PSF FWHM: {fwhm:.2f} px' + (f' (std {fwhm_std:.2f}, n={len(fwhm_list)})' if fwhm_std is not None
                                         else ' (assumed -- fewer than 3 usable stars)'))

    cutouts = extract_cutouts(sep['kept'], sep['galaxy_labels'], seg_labels, display_rgb, detect_u8,
                              prefix, p['cutout_pad'])
    with fits.open(det.path) as hdul:
        wcs = BinnedWCS(WCS(hdul[det.sci_ext].header, fobj=hdul).celestial, f)
    cat_here = _catalog_rows_in_footprint(catalog, det.corners) if catalog is not None else None
    wcs_shift = refine_wcs_shift(seg_labels, linear_data, sep['galaxy_labels'], wcs, cat_here, scale, p, log)
    if wcs_shift != (0.0, 0.0):
        wcs = BinnedWCS(wcs.wcs, f, shift=wcs_shift)
    matches = match_catalog(cutouts, seg_labels, wcs, cat_here, p['catalog_match_radius_arcsec'], scale)
    if cat_here is not None and len(cat_here):
        log(f'  catalogue: {len(matches)}/{len(cat_here)} members inside this frame matched by WCS '
            f'({", ".join(str(r["galaxy_id"]) for r in matches.values()) or "none"})')

    # development stage: morphology is measured on LINEAR sky-subtracted flux (section
    # 7b), not on the 8-bit display/detection image.
    sky_ok = ~nodata_eff & ~(cv2.dilate((seg_labels > 0).astype(np.uint8) | fg.astype(np.uint8),
                                        np.ones((15, 15), np.uint8)) > 0)
    sky_sigma = _robust_sigma(linear_data[sky_ok][::max(1, int(sky_ok.sum() // 2_000_000))]) \
        if sky_ok.sum() > 100 else _robust_sigma(linear_data[~nodata_eff])
    sky_box = int(p['sky_box_px'])
    sky_centres = find_sky_boxes(sky_ok, sky_box)
    if len(sky_centres) == 0:
        sky_box = max(8, sky_box // 2)
        sky_centres = find_sky_boxes(sky_ok, sky_box)
    log(f'  linear sky sigma {sky_sigma:.4g}; {len(sky_centres)} empty {sky_box}x{sky_box}px sky boxes '
        f'for background terms')
    r_fwhm = max(1, int(round(fwhm / 2.0)))
    nb_total = nb_own = None
    if p.get('neighbour_model') and cutouts:
        import time
        t0 = time.time()
        nb_total, nb_own = build_neighbour_models(linear_data, seg_labels, [rp.label for rp in sep['kept']],
                                                  nodata_eff, extent=float(p['nb_model_extent']))
        log(f'  neighbour models: {len(nb_own)} objects modelled ({time.time() - t0:.1f}s); '
            f'each galaxy measured with all OTHER models subtracted')

    if p.get('save_measure_cache') and cutouts:
        names = list(cutouts)
        np.savez_compressed(
            os.path.join(out_dir, '_measure_cache.npz'),
            lin=np.asarray(linear_data, np.float32), seg=np.asarray(seg_labels, np.int32),
            nodata=np.asarray(nodata_eff, bool), sky_sigma=float(sky_sigma),
            sky_centres=np.asarray(sky_centres, float), sky_box=int(sky_box), scale=float(scale),
            obj_name=np.array(names), obj_label=np.array([cutouts[n]['rp'].label for n in names]),
            obj_bbox=np.array([cutouts[n]['rp'].bbox for n in names]),
            obj_centroid=np.array([cutouts[n]['rp'].centroid for n in names]),
            obj_catalog=np.array([str(matches[n]['galaxy_id']) if n in matches else '' for n in names]))

    band_sigma = {bf: (_robust_sigma(bl[sky_ok][::max(1, int(sky_ok.sum() // 2_000_000))]) if sky_ok.sum() > 100
                       else _robust_sigma(bl[~nodata_eff])) for bf, bl in extra_lin.items()}
    if extra_lin:
        log(f"  also measuring on band(s) {', '.join(extra_lin)} with the same segmentation")

    rec_kw = ({'a_max_clipped': p['max_clipped_frac'], 'a_min_rp': float(p.get('recover_min_rp', 0.5))}
              if p.get('recover_aperture') and p.get('max_clipped_frac') is not None else {})
    if rec_kw:
        log(f"  aperture recovery on: A inside the largest circle clipped <= {p['max_clipped_frac']:.0%} "
            f"(>= {rec_kw['a_min_rp']:g} r_p)")

    def _measurable(mm):
        clip_a = mm.get('a_clipped_frac', mm['aperture_clipped_frac'])
        cap = p.get('max_clipped_frac_recover')
        return bool(mm['petro_ok'] and np.isfinite(mm['C']) and np.isfinite(mm['A'])
                    and np.isfinite(mm['sn_per_pixel']) and mm['sn_per_pixel'] >= p['min_sn_per_pixel']
                    and np.isfinite(mm['unrecoverable_frac']) and mm['unrecoverable_frac'] <= p['max_missing_frac']
                    and (p.get('max_clipped_frac') is None or (np.isfinite(clip_a) and clip_a <= p['max_clipped_frac']))
                    and (cap is None or (np.isfinite(mm['aperture_clipped_frac']) and mm['aperture_clipped_frac'] <= cap)))
    rows = []
    for cname, d in cutouts.items():
        mask = d['mask']
        r0, c0, r1, c1 = d['bbox']
        mm = measure_morphology(linear_data, seg_labels, d['rp'].label, d['rp'], sky_sigma, sky_centres,
                                sky_box, nodata=nodata_eff, nb_total=nb_total, nb_own=nb_own, **rec_kw)
        band_mm = {bf: measure_morphology(bl, seg_labels, d['rp'].label, d['rp'], band_sigma[bf], sky_centres,
                                          sky_box, nodata=nodata_eff) for bf, bl in extra_lin.items()}
        mm0 = None
        if p.get('measure_legacy') and nb_total is not None:
            mm0 = measure_morphology(linear_data, seg_labels, d['rp'].label, d['rp'], sky_sigma, sky_centres,
                                     sky_box, nodata=nodata_eff)
        C, A, S = mm['C'], mm['A'], mm['S']
        lin_cut = np.clip(np.asarray(linear_data[r0:r1, c0:c1], np.float64), 0, None)
        r_vals = list(range(0, max(r_fwhm * 4, 6) + 1))
        S_gran = granulometric_clumpiness_scalar(compute_granulometry(lin_cut, mask, r_vals), r_vals, r_fwhm)
        shp = compute_shape_moments(lin_cut, mask)
        if 'centre_xy' in mm:
            cx, cy = mm['centre_xy']
        else:
            cy, cx = d['rp'].centroid
        ra, dec = (float(v) for v in wcs.all_pix2world(cx, cy, 0))
        m = matches.get(cname)
        gal_name = str(m['galaxy_id']) if m is not None else cname
        d['display_name'] = gal_name
        row = {'Field': job.field_name, 'Image': job.label, 'Telescope': det.telescope,
               'Instrument': det.instrument, 'Detector': det.detector,
               'Filter': '+'.join(b.filter for b in job.bands), 'is_primary': job.is_primary,
               'Galaxy': gal_name, 'object_id': cname,
               'catalog_id': None if m is None else str(m['galaxy_id']),
               'morph_gt': None if m is None or pd.isna(m.get('morph_gt')) else str(m['morph_gt']),
               'catalog_group': None if m is None or pd.isna(m.get('group')) else str(m['group']),
               'label_source': None if m is None else str(m.get('label_source', m.get('catalog_file', ''))),
               'ra': round(ra, 6), 'dec': round(dec, 6),
               'C': _r(C, 4), 'r20': _r(mm['r20'], 2), 'r80': _r(mm['r80'], 2),
               'A': _r(A, 4), 'A_raw': _r(mm['A_raw'], 4), 'A_bg_local': _r(mm['A_bg_local'], 4),
               'S': _r(S, 4), 'S_raw': _r(mm['S_raw'], 4), 'S_bg_local': _r(mm['S_bg_local'], 4),
               'gini': _r(mm['gini'], 4), 'm20': _r(mm['m20'], 4),
               'r_petro_arcsec': _r(mm['r_petro_px'] * scale, 2), 'petro_ok': mm['petro_ok'],
               'sn_per_pixel': _r(mm['sn_per_pixel'], 2),
               'aperture_clipped_frac': _r(mm['aperture_clipped_frac'], 3),
               'a_clipped_frac': _r(mm.get('a_clipped_frac'), 3), 'r_a_over_rp': _r(mm.get('r_a_over_rp'), 3),
               'neighbour_frac': _r(mm.get('neighbour_frac'), 3),
               'unrecoverable_frac': _r(mm.get('unrecoverable_frac'), 3),
               'measurable': _measurable(mm), 'neighbour_model': nb_total is not None,
               'work_scale_arcsec': round(scale, 4), 'S_gran': round(S_gran, 4),
               'eccentricity': shp['eccentricity'], 'ellipticity': shp['ellipticity'],
               'area_px': shp['area'], 'total_flux': shp['total_flux'], 'perimeter_px': shp['perimeter']}
        for i, h in enumerate(shp['hu_moments']):
            row[f'Hu_{i + 1}'] = h
        for bf, bm in band_mm.items():
            for k in ('C', 'A', 'S', 'sn_per_pixel'):
                row[f'{k}_{bf}'] = _r(bm[k], 4)
            row[f'r_petro_arcsec_{bf}'] = _r(bm['r_petro_px'] * scale, 2)
            row[f'measurable_{bf}'] = _measurable(bm)
        if mm0 is not None:
            for k in ('C', 'A', 'S', 'gini', 'm20', 'sn_per_pixel'):
                row[f'{k}_v12'] = _r(mm0[k], 4)
            row['r_petro_arcsec_v12'] = _r(mm0['r_petro_px'] * scale, 2)
            row['measurable_v12'] = _measurable(mm0)
        rows.append(row)
        plt.imsave(os.path.join(out_dir, f'cutout_{_slug(gal_name)}.png'), np.clip(d['rgb'], 0, 1), origin='lower')
    df = pd.DataFrame(rows)
    log(f'  {len(df)} galaxies measured')

    show = p['show_figures']
    # segmentation overlay
    fig, ax = plt.subplots(figsize=(10, 10 * H / max(W, 1)))
    ov = np.clip(display_rgb, 0, 1) * 0.4
    shown = np.where(np.isin(seg_labels, [rp.label for rp in sep['kept']]), seg_labels, 0)
    _, shown = np.unique(shown, return_inverse=True)          # compact 0..n for the colour map
    shown = shown.reshape(seg_labels.shape)
    cm = plt.cm.nipy_spectral(shown / max(shown.max(), 1))
    mb = shown > 0
    ov[mb] = 0.45 * ov[mb] + 0.55 * cm[mb, :3]                # translucent, so structure stays visible
    if nodata_eff is not None:
        ov[nodata_eff] = ov[nodata_eff] * 0.3 + np.array([0.15, 0.0, 0.0])
    ax.imshow(ov, origin='lower')
    for cname, d in cutouts.items():
        r0, c0, r1, c1 = d['bbox']
        ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, lw=1.2, edgecolor='yellow', facecolor='none'))
        ax.text(c0 + 3, r1 + 5, d['display_name'].replace(prefix + '_', ''), color='yellow', fontsize=7,
                fontweight='bold')
    ax.set_title(f'{title}\n{len(cutouts)} galaxies, {len(sep["star_labels"])} stars'
                 + (f', {len(edge)} edge-truncated' if edge else ''), fontsize=10)
    ax.axis('off')
    seg_png = os.path.join(out_dir, f'{prefix}_segmentation_overlay.png')
    _save(fig, seg_png, show)

    # cutout gallery (grid, not one long row). development stage: bounded -- catalogue
    # members first, then the largest objects, at most gallery_max_objects
    # panels (an unbounded figure for 500+ objects overflowed matplotlib's
    # rasteriser: 'FT_Render_Glyph ... raster overflow', HCG 7 earlier stage).
    gal_png = None
    if cutouts:
        order = sorted(cutouts.items(), key=lambda kv: (kv[1]['display_name'] == kv[0], -kv[1]['rp'].area))
        order = order[:int(p['gallery_max_objects'])]
        n = len(order)
        ncol = min(n, p['gallery_max_cols'])
        nrow = int(math.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.8 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for a in axes:
            a.axis('off')
        for a, (cname, d) in zip(axes, order):
            a.imshow(np.clip(d['rgb'], 0, 1), origin='lower')
            a.set_title(d['display_name'].replace(prefix + '_', ''), fontsize=8, fontweight='bold')
        more = f' (largest {n} of {len(cutouts)})' if len(cutouts) > n else ''
        fig.suptitle(f'{title} — galaxy cutouts{more}', fontweight='bold')
        fig.tight_layout()
        gal_png = os.path.join(out_dir, f'{prefix}_cutout_gallery.png')
        _save(fig, gal_png, show)

    # morphology distributions: fixed-size histograms (any number of objects),
    # catalogue members marked individually
    cas_png = None
    if len(df):
        cols = [c for c in ['C', 'A', 'S', 'gini', 'm20'] if c in df.columns]
        fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.6))
        colors = {'C': '#3498db', 'A': '#e74c3c', 'S': '#2ecc71', 'gini': '#9b59b6', 'm20': '#e67e22'}
        mem = df[df['catalog_id'].notna()]
        for a, col in zip(np.atleast_1d(axes), cols):
            v = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v):
                a.hist(v, bins=min(30, max(5, len(v) // 3)), color=colors[col], alpha=0.8, edgecolor='white')
                a.axvline(v.median(), ls='--', color='black', lw=1.2, label=f'median {v.median():.3f}')
            for _, r in mem.iterrows():
                if np.isfinite(r[col]):
                    a.axvline(r[col], color='black', lw=0.8, alpha=0.7)
                    a.text(r[col], a.get_ylim()[1] * 0.95, str(r['Galaxy']), rotation=90, fontsize=6,
                           va='top', ha='right')
            a.set_xlabel(col)
            a.legend(fontsize=7)
        fig.suptitle(f'Morphology distributions ({len(df)} galaxies; lines = catalogue members) — {title}',
                     fontweight='bold', fontsize=9)
        fig.tight_layout()
        cas_png = os.path.join(out_dir, f'{prefix}_cas_distributions.png')
        _save(fig, cas_png, show)

    return {'job': job, 'title': title, 'out_dir': out_dir, 'prefix': prefix, 'params': p,
            'bg_box': box, 'sky_model': sky_model, 'sky_box': sky_kw['box_size'],
            'n_gal_cores': n_gal_cores,
            'nodata_frac': float(nodata.mean()), 'nodata_buffer_px': int(buf or 0),
            'n_cores': n_cores, 'k_sigma': k_sigma, 'fg_frac': fg_frac, 'sep': sep, 'n_edge': len(edge),
            'cutouts': cutouts, 'df_cas': df, 'n_catalog_in_frame': 0 if cat_here is None else len(cat_here),
            'n_catalog_matched': len(matches),
            'measured_fwhm_px': fwhm, 'fwhm_std_px': fwhm_std, 'fwhm_measurements': fwhm_list,
            'figures': {'segmentation': seg_png, 'gallery': gal_png, 'cas': cas_png}}


# ════════════════════════════════════════════════════════════════════════
# 9. CLASSIFICATION (H1, earlier stage -- unchanged rules, generic scoring)
# ════════════════════════════════════════════════════════════════════════
def choose_primary(results, log=print):
    """Pick, per field, the image whose run yielded the most measurable
    galaxies (ties -> larger sky area). Only primary rows calibrate the H1
    thresholds, so a galaxy imaged several times is not counted several
    times, and a shallow image doesn't dominate a deep one."""
    by_field = {}
    for r in results:
        by_field.setdefault(r['job'].field_name, []).append(r)
    for fname, rs in by_field.items():
        best = max(rs, key=lambda r: (len(r['df_cas']), r['job'].sky_area_arcmin2))
        for r in rs:
            r['job'].is_primary = r is best
            if len(r['df_cas']):
                r['df_cas']['is_primary'] = r is best
        if len(rs) > 1:
            log(f"  primary image for {fname}: {best['job'].label} ({len(best['df_cas'])} galaxies)")
    return results


def derive_thresholds(df):
    n = len(df)
    if n >= MIN_N_FOR_AUTO_CALIBRATION:
        return dict(mode=f'auto (n={n})', C=df['C'].median(), A=df['A'].median(),
                    A_hi=max(df['A'].quantile(0.75), 0.35),
                    S=df['S_gran'].median(), S_hi=df['S_gran'].quantile(0.75))
    return dict(mode=f'literature-fallback (n={n} < {MIN_N_FOR_AUTO_CALIBRATION})',
                C=2.5, A=0.20, A_hi=0.35, S=0.10, S_hi=None)


def classify_cas_row_v2(row, th):
    C, A, Sg = row['C'], row['A'], row['S_gran']
    if A >= th['A_hi'] or (th['S_hi'] is not None and Sg >= th['S_hi']):
        return 'Irr'
    if C > th['C'] and A < th['A'] and Sg < th['S']:
        return 'E'
    if C <= th['C'] and A >= th['A']:
        return 'S'
    return 'Irr'


def classify_all_median_calibration(df_all, log=print):
    """earlier stage behaviour, kept for comparison. Thresholds are calibrated on PRIMARY-image rows only (one image per
    field, so a galaxy seen in several images is not counted several
    times), then applied to every row."""
    if len(df_all) == 0:
        return df_all, None, None
    calib = df_all[df_all['is_primary']] if df_all['is_primary'].any() else df_all
    th = derive_thresholds(calib)
    df_all = df_all.copy()
    df_all['morph_class'] = df_all.apply(classify_cas_row_v2, axis=1, th=th)
    s_hi = 'n/a' if th['S_hi'] is None else f"{th['S_hi']:.4f}"
    log(f"\n{'=' * 78}\nCLASSIFICATION -- mode: {th['mode']} (calibrated on primary images)\n"
        f"  C_mod={th['C']:.4f}  A_mod={th['A']:.4f}  A_hi={th['A_hi']:.4f}  "
        f"S_gran_mod={th['S']:.4f}  S_gran_hi={s_hi}\n{'=' * 78}")
    for (fname, img), g in df_all.groupby(['Field', 'Image'], sort=False):
        cc = g['morph_class'].value_counts()
        log(f"  {fname} / {img}: {len(g)} galaxies -> E {cc.get('E', 0)}, S {cc.get('S', 0)}, "
            f"Irr {cc.get('Irr', 0)}")

    scored = df_all.dropna(subset=['morph_gt'])
    acc = None
    if len(scored):
        classes = ['E', 'S', 'Irr']
        per_image = []
        for (fname, img), g in scored.groupby(['Field', 'Image'], sort=False):
            ok = int((g['morph_class'] == g['morph_gt']).sum())
            per_image.append({'Field': fname, 'Image': img, 'n_scored': len(g), 'n_correct': ok,
                              'accuracy_pct': round(100 * ok / len(g), 1)})
        cm = pd.crosstab(scored['morph_gt'], scored['morph_class']).reindex(
            index=classes, columns=classes, fill_value=0)
        ok_all = int((scored['morph_class'] == scored['morph_gt']).sum())
        acc = {'per_image': pd.DataFrame(per_image), 'confusion': cm,
               'n_scored': len(scored), 'n_correct': ok_all}
        log(f"\nACCURACY vs catalogue ground truth ({len(scored)} matched galaxies, all images):")
        log(acc['per_image'].to_string(index=False))
        log('Confusion matrix (rows = ground truth, cols = predicted):\n' + cm.to_string())
        log(f'CAVEAT: n={len(scored)} is far too small for a statistically meaningful accuracy figure.')
    else:
        log('\nNo catalogue-matched galaxies -- nothing to score (add a catalogue CSV to enable).')
    return df_all, th, acc


# ── development stage: supervised, cross-validated calibration of the same rule ─────
# The earlier stage 'auto' mode set C/A/S thresholds to SAMPLE MEDIANS. A median
# splits whatever happens to be in the sample (HCG 42's 132 noise blobs set
# A_mod = 0.028) and never looks at the truth. Here the thresholds of the
# rule are chosen on the catalogue galaxies with published morphologies
# (Hickson 1989; NED/GZ2 for SQ) -- one measurement per galaxy -- by an
# exhaustive grid search, and the accuracy that is REPORTED is
# leave-one-group-out (LOGO) cross-validated: every galaxy is classified by
# thresholds fitted without any galaxy of its own group. Still a fixed,
# human-readable rule with 3-4 numbers -- no learned model.
CLASSES = ('E', 'S', 'Irr')
MIN_N_SUPERVISED = 15
LITERATURE_RULE = {'family': 'CA', 'C_E': 3.5, 'A_E': 0.10, 'A_hi': 0.35, 'S_hi': None,
                   'mode': 'literature fallback (Conselice 2003 ranges, approximate)'}


def predict_rule(df, th):
    """Irr  if A >= A_hi  (or S >= S_hi);  E  if C > C_E and A < A_E;  else S.
    Rows flagged measurable == False -> 'unclassified'."""
    C = df['C'].to_numpy(float)
    A = df['A'].to_numpy(float)
    S = df['S'].to_numpy(float) if 'S' in df else np.zeros(len(df))
    out = np.full(len(df), 'S', dtype=object)
    irr = A >= th['A_hi']
    if th.get('S_hi') is not None:
        irr |= S >= th['S_hi']
    e = (C > th['C_E']) & (A < th['A_E']) & ~irr
    out[irr] = 'Irr'
    out[e] = 'E'
    meas = df['measurable'].to_numpy(bool) if 'measurable' in df else np.ones(len(df), bool)
    out[~(meas & np.isfinite(C) & np.isfinite(A))] = 'unclassified'
    return out


def _balanced_acc(truth, pred):
    rec = [np.mean(pred[truth == c] == c) for c in CLASSES if np.any(truth == c)]
    return float(np.mean(rec)) if rec else 0.0


def _cands(v, n=14):
    v = np.unique(np.round(v[np.isfinite(v)], 4))
    if v.size < 2:
        return v
    mids = 0.5 * (v[1:] + v[:-1])
    if mids.size > n:
        mids = np.unique(np.quantile(mids, np.linspace(0, 1, n)))
    return mids


def fit_rule(lab, family='CA'):
    """Exhaustive search of the rule thresholds maximising balanced accuracy
    (ties: plain accuracy, then the candidate nearest the tie-set median, so
    each boundary sits mid-way between classes rather than on one galaxy)."""
    truth = lab['morph_gt'].to_numpy(object)
    C, A, S = (lab[c].to_numpy(float) for c in ('C', 'A', 'S'))
    s_opts = [None] + list(_cands(S)) if family == 'CAS' else [None]
    res = []
    for ce in _cands(C):
        for ae in _cands(A):
            for ah in _cands(A):
                if ah < ae:
                    continue
                for sh in s_opts:
                    th = {'C_E': float(ce), 'A_E': float(ae), 'A_hi': float(ah),
                          'S_hi': None if sh is None else float(sh)}
                    pr = predict_rule(lab, th)
                    res.append((_balanced_acc(truth, pr), float(np.mean(pr == truth)), th))
    best_b = max(r[0] for r in res)
    best_a = max(r[1] for r in res if r[0] == best_b)
    ties = [r[2] for r in res if r[0] == best_b and r[1] == best_a]
    keys = ['C_E', 'A_E', 'A_hi']
    med = {k: float(np.median([t[k] for t in ties])) for k in keys}
    n_none = sum(t['S_hi'] is None for t in ties)
    pool = [t for t in ties if (t['S_hi'] is None) == (n_none * 2 >= len(ties))] or ties
    th = dict(min(pool, key=lambda t: sum(abs(t[k] - med[k]) for k in keys)))
    th['family'] = family
    return th


def logo_cv(lab, family):
    """Leave-one-group-out predictions for every labelled galaxy."""
    pred = pd.Series(index=lab.index, dtype=object)
    for g in lab['group'].unique():
        tr, te = lab[lab['group'] != g], lab[lab['group'] == g]
        if tr['morph_gt'].nunique() < 2:
            pred[te.index] = 'S'
            continue
        pred[te.index] = predict_rule(te, fit_rule(tr, family))
    return pred


def labelled_table(df_all):
    """One measurement per catalogue galaxy: the reddest available image
    (rest-frame red light traces the stellar mass distribution best), ties ->
    higher S/N."""
    m = df_all[df_all['catalog_id'].notna() & df_all['morph_gt'].notna()].copy()
    if m.empty:
        return m
    lam = m['Filter'].astype(str).str.split('+').str[-1].str.extract(r'F(\d+)')[0].astype(float)
    m['_lam'] = np.where(m['Telescope'].astype(str).str.upper() == 'JWST', lam * 10, lam)
    m = m.sort_values(['catalog_id', '_lam', 'sn_per_pixel'], ascending=[True, False, False])
    m = m.drop_duplicates('catalog_id')
    # LOGO groups = the catalogue GROUP (neighbour-model run: a group may span several
    # fields/pointings, and HyperLEDA galaxies carry the group of their field)
    m['group'] = m['catalog_group'].where(m['catalog_group'].notna(), m['Field']) \
        if 'catalog_group' in m else m['Field']
    return m.drop(columns=['_lam'])


def classify_all(df_all, log=print, frozen=None):
    """development stage classification: supervised + LOGO-cross-validated thresholds
    when >= MIN_N_SUPERVISED labelled, measurable galaxies exist, else the
    literature fallback. With `frozen` (a rule dict, e.g. read from a saved
    classification_rule.json) the rule is applied UNCHANGED -- nothing is
    refitted -- and its accuracy on the labelled galaxies is a genuine
    hold-out score. Returns (df_all, thresholds, accuracy)."""
    if len(df_all) == 0:
        return df_all, None, None
    df_all = df_all.copy()
    if 'measurable' not in df_all:
        df_all['measurable'] = True
    lab_all = labelled_table(df_all)
    lab = lab_all[lab_all['measurable'].astype(bool)] if len(lab_all) else lab_all
    acc = None
    if frozen is not None:
        th = dict(frozen)
        th['mode'] = f"FROZEN rule ({th.get('mode', 'saved rule')}) -- applied unchanged, no refitting"
        if len(lab):
            lab = lab.assign(pred_cv=predict_rule(lab, th), pred_fit=predict_rule(lab, th))
            a_ = float(np.mean(lab['pred_cv'] == lab['morph_gt']))
            acc = {'cv_accuracy': a_, 'cv_balanced': _balanced_acc(lab['morph_gt'].to_numpy(object),
                                                                    lab['pred_cv'].to_numpy(object)),
                   'cv_by_family': {}, 'resub_accuracy': a_, 'holdout': True,
                   'confusion': pd.crosstab(lab['morph_gt'], lab['pred_cv']).reindex(
                       index=list(CLASSES), columns=list(CLASSES) + ['unclassified'], fill_value=0),
                   'table': lab, 'n_unmeasurable': int(len(lab_all) - len(lab)),
                   'unmeasurable': lab_all[~lab_all['measurable'].astype(bool)]}
    elif len(lab) >= MIN_N_SUPERVISED and lab['morph_gt'].nunique() >= 2:
        truth = lab['morph_gt'].to_numpy(object)
        cv = {}
        for fam in ('CA', 'CAS'):
            pr = logo_cv(lab, fam).to_numpy(object)
            cv[fam] = (_balanced_acc(truth, pr), float(np.mean(pr == truth)), pr)
        fam = 'CA' if cv['CA'][:2] >= cv['CAS'][:2] else 'CAS'     # simpler family wins ties
        th = fit_rule(lab, fam)
        th['mode'] = (f"supervised on {len(lab)} catalogue galaxies, rule family {fam}; "
                      f"LOGO-CV accuracy {cv[fam][1]:.0%} (balanced {cv[fam][0]:.0%})")
        lab = lab.assign(pred_cv=cv[fam][2], pred_fit=predict_rule(lab, th))
        acc = {'cv_accuracy': cv[fam][1], 'cv_balanced': cv[fam][0],
               'cv_by_family': {f: {'accuracy': round(v[1], 3), 'balanced': round(v[0], 3)}
                                for f, v in cv.items()},
               'resub_accuracy': float(np.mean(lab['pred_fit'] == lab['morph_gt'])),
               'confusion': pd.crosstab(lab['morph_gt'], lab['pred_cv']).reindex(
                   index=list(CLASSES), columns=list(CLASSES) + ['unclassified'], fill_value=0),
               'table': lab, 'n_unmeasurable': int(len(lab_all) - len(lab)),
               'unmeasurable': lab_all[~lab_all['measurable'].astype(bool)]}
    else:
        th = dict(LITERATURE_RULE)
        th['mode'] += f' (only {len(lab)} labelled measurable galaxies)'
    df_all['morph_class'] = predict_rule(df_all, th)
    s_txt = f" or S >= {th['S_hi']:.3f}" if th.get('S_hi') is not None else ''
    log(f"\n{'=' * 78}\nCLASSIFICATION -- {th['mode']}\n"
        f"  rule: Irr if A >= {th['A_hi']:.3f}{s_txt}; E if C > {th['C_E']:.3f} and A < {th['A_E']:.3f}; "
        f"otherwise S\n{'=' * 78}")
    for (fname, img), g in df_all.groupby(['Field', 'Image'], sort=False):
        cc = g['morph_class'].value_counts()
        log(f"  {fname} / {img}: {len(g)} galaxies -> E {cc.get('E', 0)}, S {cc.get('S', 0)}, "
            f"Irr {cc.get('Irr', 0)}, unclassified {cc.get('unclassified', 0)}")
    if acc is not None:
        log(f"\nACCURACY vs published morphologies ({len(acc['table'])} measurable catalogue galaxies, "
            f"one measurement each; {acc['n_unmeasurable']} more flagged unmeasurable):")
        if acc.get('holdout'):
            log(f"  HOLD-OUT accuracy of the frozen rule: {acc['cv_accuracy']:.1%} "
                f"(balanced {acc['cv_balanced']:.1%})")
        else:
            log(f"  leave-one-group-out cross-validated: {acc['cv_accuracy']:.1%} "
                f"(balanced {acc['cv_balanced']:.1%})   [fit on all, resubstitution: {acc['resub_accuracy']:.1%}]")
        log('  CV confusion matrix (rows = published, cols = predicted):\n' + acc['confusion'].to_string())
        t = acc['table']
        bad = t[t['pred_cv'] != t['morph_gt']]
        if len(bad):
            log('  CV misclassifications: ' + ', '.join(f'{r.catalog_id} ({r.morph_gt}->{r.pred_cv})'
                                                         for r in bad.itertuples()))
    return df_all, th, acc


# ════════════════════════════════════════════════════════════════════════
# 10. REPORT
# ════════════════════════════════════════════════════════════════════════
def _md_table(df, floats=(), ints=()):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in floats and isinstance(v, (int, float, np.floating)):
                cells.append(f'{v:.4f}')
            elif c in ints and isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f'{int(v)}')
            else:
                cells.append('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def write_report(results, df_all, th, acc, manifest, report_path):
    rdir = os.path.dirname(os.path.abspath(report_path))

    def rel(p):
        return os.path.relpath(p, rdir).replace(os.sep, '/') if p else None

    L = ['# Morphological Classification of Galaxies — Results Report',
         f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
         '\nPipeline: classical, deterministic DIP (background modelling, watershed, mathematical '
         'morphology, CAS + granulometry, rule-based E/S/Irr). No machine learning or deep learning. '
         'Images, fields and parameters are discovered from the FITS headers; nothing is hard-coded '
         'per target.\n',
         '## Input images\n',
         _md_table(manifest[['field', 'file', 'telescope', 'instrument', 'detector', 'filter', 'product',
                             'shape', 'pix_scale_arcsec', 'status']]), '']
    if th is not None:
        s_txt = f" or S ≥ {th['S_hi']:.3f}" if th.get('S_hi') is not None else ''
        L += ['## Classification rule\n',
              f"Mode: **{th['mode']}**.\n",
              f"Rule: **Irr** if A ≥ {th['A_hi']:.3f}{s_txt}; **E** if C > {th['C_E']:.3f} and "
              f"A < {th['A_E']:.3f}; otherwise **S**. Objects whose measurement is unreliable "
              f"(⟨S/N⟩ per pixel < 2.5, no Petrosian radius, or > 25 % of the 1.5 r_p aperture "
              f"unrecoverably off the data) are reported as *unclassified*.\n",
              'C, A, S, Gini and M20 are measured on linear, sky-subtracted flux at a common 0.2"/px scale '
              '(Conselice 2003; Lotz et al. 2004 definitions).\n']
    if acc is not None:
        t = acc['table']
        L += ['## Accuracy against published morphologies\n',
              "Ground truth: Hickson (1989) types for HCG members, NED/GZ2 for Stephan's Quintet; "
              "E/S0 → E, Sa–Sd → S, Sdm/Sm/Im → Irr. One measurement per galaxy (reddest image).\n",
              f"- **Leave-one-group-out cross-validated accuracy: {acc['cv_accuracy']:.1%}** "
              f"(balanced {acc['cv_balanced']:.1%}) on {len(t)} measurable galaxies — every galaxy "
              f"is classified with thresholds fitted without its own group.",
              f"- Resubstitution (thresholds fitted on all, applied to all): {acc['resub_accuracy']:.1%} "
              f"— optimistic, shown for reference only.",
              f"- {acc['n_unmeasurable']} catalogued galaxy(ies) flagged unmeasurable: "
              + (', '.join(acc['unmeasurable']['catalog_id'].astype(str)) or 'none') + '\n',
              'Cross-validated confusion matrix (rows = published, columns = predicted):\n',
              _md_table(acc['confusion'].reset_index().rename(columns={'morph_gt': 'published \\ predicted'})), '',
              _md_table(t[['catalog_id', 'Field', 'Filter', 'C', 'A', 'S', 'gini', 'm20', 'morph_gt',
                           'pred_cv', 'pred_fit']].rename(columns={'pred_cv': 'pred (CV)',
                                                                  'pred_fit': 'pred (final rule)'}),
                        floats=('C', 'A', 'S', 'gini', 'm20')), '',
              f"Sample size n={len(t)}: a single galaxy moves the accuracy by {100 / max(len(t), 1):.1f} points.\n"]
    L.append('---\n')
    for res in results:
        job = res['job']
        L += [f"## {res['title']}\n",
              f"- File(s): {', '.join('`' + os.path.basename(b.path) + '`' for b in job.bands)}"
              f"{' — **primary image for this field**' if job.is_primary else ''}",
              f"- Sky model **{res['sky_model']}** (box {res['sky_box']} px); no-data {res['nodata_frac']:.1%}"
              + (f", buffered {res['nodata_buffer_px']} px" if res['nodata_buffer_px'] else ''),
              f"- Detection: k={res['k_sigma']:.2f}σ, **{res['n_cores']}** cores → **{res['sep']['lbl']}** after "
              f"area filter → **{len(res['sep']['star_labels'])}** stars, **{len(res['cutouts'])}** galaxies"
              + (f" ({res['n_edge']} edge-truncated removed)" if res['n_edge'] else ''),
              f"- PSF FWHM: {res['measured_fwhm_px']:.2f} px"
              + (f" (std {res['fwhm_std_px']:.2f}, n={len(res['fwhm_measurements'])})"
                 if res['fwhm_std_px'] is not None else ' (assumed)'),
              f"- Catalogue members in frame: {res['n_catalog_in_frame']}, matched: {res['n_catalog_matched']}\n"]
        for key, cap in [('segmentation', 'segmentation overlay'), ('gallery', 'galaxy cutouts'),
                         ('cas', 'CAS distributions')]:
            if res['figures'].get(key):
                L.append(f"![{res['title']} — {cap}]({rel(res['figures'][key])})\n")
        sub = df_all[df_all['Image'] == job.label] if len(df_all) else df_all
        if len(sub):
            cols = ['Galaxy', 'C', 'A', 'S', 'gini', 'm20', 'r_petro_arcsec', 'sn_per_pixel', 'area_px',
                    'morph_class', 'morph_gt']
            sub = sub.sort_values('area_px', ascending=False)
            if len(sub) > 40:
                L.append(f'*{len(sub)} galaxies; catalogue members and the 40 largest shown '
                         f'(all rows in cas_features_all_images.csv).*\n')
                sub = pd.concat([sub[sub['catalog_id'].notna()], sub[sub['catalog_id'].isna()].head(40)])
            L.append(_md_table(sub[[c for c in cols if c in sub.columns]],
                               floats=('C', 'A', 'S', 'gini', 'm20', 'r_petro_arcsec', 'sn_per_pixel'),
                               ints=('area_px',)))
        L.append('\n---\n')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return report_path


# ════════════════════════════════════════════════════════════════════════
# 11. TOP-LEVEL DRIVER
# ════════════════════════════════════════════════════════════════════════
def _slim_result(res):
    """Drop the heavy per-object objects (regionprops, masks) from a job result so it
    can be returned from a worker process; keeps everything write_report and the
    notebook read."""
    res = dict(res)
    sep = res.get('sep') or {}
    res['sep'] = {'lbl': sep.get('lbl'), 'star_labels': list(sep.get('star_labels', []))}
    res['cutouts'] = {k: {'display_name': v.get('display_name', k)} for k, v in (res.get('cutouts') or {}).items()}
    return res


def _job_worker(job, cfg, output_dir, catalog):
    """Process-pool entry point: run one job, capture its log lines."""
    import matplotlib
    matplotlib.use('Agg')
    lines = []
    try:
        res = run_image_job(job, cfg, output_dir, catalog=catalog, log=lambda *a: lines.append(' '.join(map(str, a))))
        return _slim_result(res), None, lines
    except Exception as e:
        import traceback
        lines.append(f'\n! {job.field_name} / {job.label} failed: {type(e).__name__}: {e} -- skipped')
        lines.append(traceback.format_exc(limit=3))
        return None, f'{type(e).__name__}: {e}', lines


def _run_jobs_parallel(jobs, cfg, output_dir, catalog, workers, log):
    """baseline run: image jobs are independent, so run them in a process pool. Each job's
    log is printed as one block when it finishes; results keep the job order, so the
    outputs are identical to a serial run."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    workers = max(1, min(workers, len(jobs), os.cpu_count() or 1))
    threads = str(max(1, (os.cpu_count() or 1) // workers))
    saved = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    for k in saved:                                    # inherited by the spawned workers: no oversubscription
        os.environ[k] = threads
    log(f'\nRunning {len(jobs)} job(s) on {workers} worker processes ({threads} BLAS thread(s) each) ...')
    out = [None] * len(jobs)
    failures = []
    # big multi-band frames first, so the slowest jobs do not start last
    order = sorted(range(len(jobs)), key=lambda i: -sum(b.shape[0] * b.shape[1] if getattr(b, 'shape', None)
                                                        else 0 for b in jobs[i].bands))
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_job_worker, jobs[i], cfg, output_dir, catalog): i for i in order}
            for done, fu in enumerate(as_completed(futs), 1):
                i = futs[fu]
                try:
                    res, err, lines = fu.result()
                except Exception as e:                 # worker crashed (e.g. out of memory)
                    res, err, lines = None, f'{type(e).__name__}: {e}', [f'\n! {jobs[i].label} crashed: {e}']
                for ln in lines:
                    log(ln)
                log(f'  [{done}/{len(jobs)} jobs finished]')
                if res is not None:
                    res['job'].is_primary = jobs[i].is_primary
                    jobs[i] = res['job']                # keep the parent's job list pointing at the returned copy
                    out[i] = res
                else:
                    failures.append((jobs[i].label, err))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return [r for r in out if r is not None], failures


def run_all(config_path=None, data_dir=None, output_dir=None, only=None, show_figures=None, log=print,
            rule_file=None, workers=None, defaults=None):
    """Discover -> build jobs -> run every job -> classify -> report.
    Returns a dict with manifest, results, df_cas_all, thresholds, accuracy.
    workers > 1 (or config key 'workers') runs the image jobs in parallel processes."""
    cfg = load_config(config_path)
    if defaults:                                       # per-call overrides of the config 'defaults'
        cfg['defaults'].update(defaults)
    data_dir = data_dir or cfg['data_dir']
    output_dir = output_dir or cfg['output_dir'] or os.path.join(os.path.dirname(os.path.abspath(data_dir)),
                                                                  'pipeline_output')
    if show_figures is not None:
        cfg['defaults']['show_figures'] = bool(show_figures)
    os.makedirs(output_dir, exist_ok=True)
    catalog = load_catalogs(cfg.get('catalog_dir'))

    log(f'Discovering FITS images under {data_dir} ...')
    included, excluded = discover_images(data_dir, cfg, catalog)
    if only:
        pats = [only] if isinstance(only, str) else list(only)
        keep = [r for r in included if _matches_any(r, pats)]
        for r in included:
            if r not in keep:
                r.status = 'not selected (--only)'
                excluded.append(r)
        included = keep
    jobs = build_jobs(included, cfg)
    manifest = manifest_frame(included + excluded)
    log(manifest[['field', 'file', 'instrument', 'detector', 'filter', 'product', 'status']].to_string(index=False))
    log(f'\n{len(included)} image(s) in {len({r.field_name for r in included})} field(s) -> {len(jobs)} job(s); '
        f'{len(excluded)} excluded. Catalogue rows loaded: {len(catalog)}')

    results, failures = [], []
    workers = int(workers if workers is not None else cfg.get('workers') or 1)
    if workers > 1 and len(jobs) > 1:
        results, failures = _run_jobs_parallel(jobs, cfg, output_dir, catalog, workers, log)
    else:
        for job in jobs:
            try:
                results.append(run_image_job(job, cfg, output_dir, catalog=catalog, log=log))
            except Exception as e:
                import traceback
                failures.append((job.label, f'{type(e).__name__}: {e}'))
                log(f'\n! {job.field_name} / {job.label} failed: {type(e).__name__}: {e} -- skipped')
                log(traceback.format_exc(limit=3))

    choose_primary(results, log=log)
    frames = [r['df_cas'] for r in results if len(r['df_cas'])]
    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frozen = None
    if rule_file:
        with open(rule_file, 'r', encoding='utf-8') as fh:
            frozen = json.load(fh)
        log(f'\nApplying FROZEN classification rule from {rule_file}')
    df_all, th, acc = classify_all(df_all, log=log, frozen=frozen)
    if th is not None:
        keep = {k: th.get(k) for k in ('family', 'C_E', 'A_E', 'A_hi', 'S_hi', 'mode')}
        keep.update({'saved': datetime.now().strftime('%Y-%m-%d %H:%M'),
                     'n_labelled': None if acc is None else int(len(acc['table'])),
                     'accuracy': None if acc is None else round(acc['cv_accuracy'], 4),
                     'accuracy_kind': None if acc is None else ('hold-out' if acc.get('holdout') else 'LOGO-CV')})
        with open(os.path.join(output_dir, 'classification_rule.json'), 'w', encoding='utf-8') as fh:
            json.dump(keep, fh, indent=1)

    status = {r['job'].detect.path: 'processed' for r in results}
    for lbl, err in failures:
        for j in jobs:
            if j.label == lbl:
                status[j.detect.path] = f'failed: {err}'
    manifest['status'] = [status.get(p, s) for p, s in zip(manifest['path'], manifest['status'])]

    manifest.to_csv(os.path.join(output_dir, 'image_manifest.csv'), index=False)
    if len(df_all):
        df_all.to_csv(os.path.join(output_dir, 'cas_features_all_images.csv'), index=False)
        df_all[['Field', 'Image', 'Galaxy', 'ra', 'dec', 'C', 'A', 'S_gran', 'morph_class', 'morph_gt',
                'is_primary']].to_csv(os.path.join(output_dir, 'galaxy_classifications.csv'), index=False)
    report = write_report(results, df_all, th, acc, manifest, os.path.join(output_dir, 'DIP_Results_Report.md'))
    log(f'\nOutputs written to {output_dir}\n  report: {report}')
    if failures:
        log(f'  {len(failures)} job(s) failed: ' + '; '.join(f'{a} ({b})' for a, b in failures))
    return {'config': cfg, 'manifest': manifest, 'jobs': jobs, 'results': results, 'failures': failures,
            'df_cas_all': df_all, 'thresholds': th, 'accuracy': acc, 'output_dir': output_dir,
            'report': report}


# ════════════════════════════════════════════════════════════════════════
# 12. RUN RELOAD -- reload a finished run; per-image catalogue coverage audit
# ════════════════════════════════════════════════════════════════════════
def load_run(output_dir, log=print, frozen=None):
    """Rebuild what run_all returns from a FINISHED run's files (no image is
    re-processed): cas_features_all_images.csv is re-classified with
    classify_all (deterministic, so the accuracy equals the run's), and the
    per-job figures are found on disk. Lets a notebook be re-executed without
    repeating the full pipeline run."""
    csv = os.path.join(output_dir, 'cas_features_all_images.csv')
    if not os.path.exists(csv):
        raise FileNotFoundError(f'{csv} not found -- run the pipeline first (run_all / run_pipeline.py)')
    df = pd.read_csv(csv, low_memory=False)
    for c in ('catalog_id', 'morph_gt', 'catalog_group', 'label_source'):
        if c in df:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    df_all, th, acc = classify_all(df, log=log, frozen=frozen)
    results = []
    for (field, image), g in df_all.groupby(['Field', 'Image'], sort=False):
        d = os.path.join(output_dir, _slug(field), _slug(image))
        figs = {}
        for key, pat in (('segmentation', '*_segmentation_overlay.png'), ('gallery', '*_cutout_gallery.png'),
                         ('cas', '*_cas_distributions.png')):
            hit = sorted(glob.glob(os.path.join(d, pat)))
            figs[key] = hit[0] if hit else None
        results.append({'title': f'{field} — {image}', 'field': field, 'image': image, 'out_dir': d,
                        'figures': figs, 'df_cas': g})
    man = os.path.join(output_dir, 'image_manifest.csv')
    manifest = pd.read_csv(man) if os.path.exists(man) else pd.DataFrame()
    return {'manifest': manifest, 'results': results, 'failures': [], 'df_cas_all': df_all,
            'thresholds': th, 'accuracy': acc, 'output_dir': output_dir,
            'report': os.path.join(output_dir, 'DIP_Results_Report.md')}


def catalog_coverage(manifest, df_all, catalog, path_map=None, max_clipped_frac=None):
    """Per-image audit of the LABELLED catalogue members: which fall on valid
    pixels of each processed detection image, which were detected and matched,
    which were measurable, and -- for every member not matched -- the likely
    reason (segment merged into a neighbouring member / truncated by the frame
    or no-data edge / not detected). path_map(path) -> local path, for runs
    made on another machine. Returns a DataFrame, one row per member and image."""
    from astropy.wcs import WCS as _WCS
    cat = catalog[catalog['morph_gt'].notna()].reset_index(drop=True)
    rows = []
    proc = manifest[manifest['status'].astype(str) == 'processed']
    for _, m in proc.iterrows():
        mpath = str(m['path']).replace('\\', '/')           # runs made on Windows store backslash paths
        tail = _slug(os.path.splitext(os.path.basename(mpath))[0])[:24]
        sub = df_all[df_all['Image'].astype(str).str.endswith(tail)]
        if sub.empty:                       # not the detection band of its job
            continue
        image = sub['Image'].iloc[0]
        p = path_map(m['path']) if path_map else mpath
        if not os.path.exists(p):                   # other machine: find the file by name under the data dir
            _dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mastDownload')
            _hit = [os.path.join(r, f) for r, _, fs in os.walk(_dd) for f in fs if f == os.path.basename(mpath)]
            if _hit:
                p = _hit[0]
        with fits.open(p, memmap=True) as h:
            ext = next(i for i, x in enumerate(h) if x.data is not None and np.ndim(x.data) == 2)
            w = _WCS(h[ext].header, fobj=h).celestial
            data = h[ext].data
            scale = float(m['pix_scale_arcsec'])
            x, y = w.wcs_world2pix(cat['ra'].to_numpy(float), cat['dec'].to_numpy(float), 0)
            H, W = data.shape
            for i in np.flatnonzero(np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W) & (y >= 0) & (y < H)):
                yi, xi = int(y[i]), int(x[i])
                v = np.asarray(data[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3], float)
                if np.mean(np.isfinite(v) & (v != 0)) < 0.5:     # 5x5: a single flagged core pixel is not 'off the data'
                    continue
                c = cat.iloc[i]
                gid = str(c['galaxy_id'])
                hit = sub[sub['catalog_id'].astype(str) == gid]
                row = {'Field': m['field'], 'Image': image, 'Filter': sub['Filter'].iloc[0], 'galaxy_id': gid,
                       'morph_gt': c['morph_gt'], 'matched': len(hit) > 0,
                       'measurable': bool(len(hit) and bool(hit['measurable'].iloc[0])), 'note': ''}
                if len(hit) and not row['measurable']:
                    h0 = hit.iloc[0]
                    why = []
                    if np.isfinite(h0.get('sn_per_pixel', np.nan)) and h0['sn_per_pixel'] < DEFAULTS['min_sn_per_pixel']:
                        why.append(f"S/N per px {h0['sn_per_pixel']:.1f} < {DEFAULTS['min_sn_per_pixel']}")
                    if np.isfinite(h0.get('unrecoverable_frac', np.nan)) and h0['unrecoverable_frac'] > DEFAULTS['max_missing_frac']:
                        why.append(f"{h0['unrecoverable_frac']:.0%} of aperture off the data")
                    if not bool(h0.get('petro_ok', True)):
                        why.append('no Petrosian radius')
                    if max_clipped_frac is not None and np.isfinite(h0.get('aperture_clipped_frac', np.nan))                             and h0['aperture_clipped_frac'] > max_clipped_frac:
                        why.append(f"{h0['aperture_clipped_frac']:.0%} of aperture cut by the frame edge (> {max_clipped_frac:.0%})")
                    row['note'] = 'unmeasurable: ' + ('; '.join(why) or 'C/A not finite')
                elif not len(hit):
                    sep = np.hypot((sub['ra'] - c['ra']) * np.cos(np.radians(c['dec'])), sub['dec'] - c['dec']) * 3600
                    j = sep.idxmin() if len(sep) else None
                    near = sub.loc[j] if j is not None else None
                    r_half = 0.5 * float(c['diam_arcsec']) if np.isfinite(c.get('diam_arcsec', np.nan)) else 10.0
                    rr = max(2, int(r_half / scale))
                    y0, y1 = max(0, int(y[i]) - rr), min(H, int(y[i]) + rr + 1)
                    x0, x1 = max(0, int(x[i]) - rr), min(W, int(x[i]) + rr + 1)
                    box = np.asarray(data[y0:y1, x0:x1], float)
                    edge = (y0 == 0 or x0 == 0 or y1 == H or x1 == W
                            or np.mean(~np.isfinite(box) | (box == 0)) > 0.02)
                    if near is not None and isinstance(near.get('catalog_id'), str) and \
                            np.isfinite(near.get('r_petro_arcsec', np.nan)) and sep[j] <= 1.5 * near['r_petro_arcsec']:
                        row['note'] = (f"inside the segment of {near['catalog_id']} ({sep[j]:.0f}\" away, "
                                       f"1.5 r_p = {1.5 * near['r_petro_arcsec']:.0f}\")")
                    elif edge:
                        row['note'] = 'at the frame / no-data edge (edge-truncated objects are dropped)'
                    else:
                        row['note'] = f'not detected (nearest object {sep[j]:.0f}" away)' if j is not None else 'not detected'
                rows.append(row)
    return pd.DataFrame(rows)
