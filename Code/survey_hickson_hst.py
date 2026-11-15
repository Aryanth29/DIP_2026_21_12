"""
Survey MAST for archival HST imaging of every Hickson compact group in
catalogs/hickson_compact_groups.csv and pick images per group with a fixed,
group-independent rule -- no hand-picking, so adding data cannot be used to
steer the results.

Selection rule (identical for every group, decided before any evaluation):
  1. HST, calibrated drizzled science image (Hubble Advanced Products,
     obs_id 'hst_*', single filter), instrument ACS/WFC, WFC3/UVIS, WFC3/IR or WFPC2;
  2. broad-band optical filter, red preferred (rest-frame optical traces the
     stellar mass): F814W > F775W > F850LP > F625W > F606W > F702W > F675W > F555W,
     then (neighbour-model run) WFC3/IR F160W > F110W > F125W > F140W;
  3. the image footprint (MAST s_region) must contain >= MIN_MEMBERS labelled
     members of the group;
  4. among the candidates: most labelled members covered, then instrument
     (ACS/WFC > WFC3/UVIS > WFC3/IR > WFPC2), then filter priority, then exposure time.
  5. (neighbour-model run) several pointings per group: after the first pick, keep adding
     the best image by the same key, counting only labelled members NOT yet
     covered, while it adds >= MIN_MEMBERS new ones (greedy set cover). With
     --multi, images already in mastDownload/ count as picked first, so
     existing data are kept and only the missing members are filled in.

development stage used MIN_MEMBERS = 2 and one image per group, and (bug) never matched
WFPC2, which MAST lists as 'WFPC2/PC' / 'WFPC2/WFC'. neighbour-model run uses
MIN_MEMBERS = 1, --multi, prefix matching of the instrument and WFC3/IR.

    python survey_hickson_hst.py               # writes hickson_hst_selection.csv
    python survey_hickson_hst.py --skip-have   # ignore groups already in mastDownload/

Then download the selected obs_ids with fetch_mast_products.py.
Needs astroquery + internet access to MAST.
"""
import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Observations

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
# neighbour-model run: + WFC3/IR broad bands (used only where no optical band covers a member)
FILTER_PRIORITY = ['F814W', 'F775W', 'F850LP', 'F625W', 'F606W', 'F702W', 'F675W', 'F555W',
                   'F160W', 'F110W', 'F125W', 'F140W']
# MAST names WFPC2 products 'WFPC2/PC' / 'WFPC2/WFC'; development stage compared with == 'WFPC2', so no
# WFPC2 image was ever selected (bug, fixed in neighbour-model run by matching on the instrument prefix)
INSTR_PRIORITY = ['ACS/WFC', 'WFC3/UVIS', 'WFC3/IR', 'WFPC2']
MIN_MEMBERS = 1                      # development stage: 2


def _instr_rank(name):
    name = str(name)
    for i, k in enumerate(INSTR_PRIORITY):
        if name == k or name.startswith(k + '/'):
            return i
    return None


def _query(item):
    grp, g = item
    c = SkyCoord(g['ra'].mean() * u.deg, g['dec'].mean() * u.deg)
    rad = max(2.0, SkyCoord(g['ra'].values * u.deg, g['dec'].values * u.deg).separation(c).arcmin.max() + 1.0)
    err = None
    for _ in range(3):
        try:
            obs = Observations.query_criteria(coordinates=c, radius=f'{rad} arcmin', obs_collection='HST',
                                              dataproduct_type='image', calib_level=3, intentType='science')
            return grp, (obs.to_pandas() if len(obs) else pd.DataFrame()), None
        except Exception as e:
            err = e
    return grp, None, err


def _polygons(s_region):
    """MAST STC-S s_region -> list of (N, 2) RA/Dec vertex arrays."""
    polys = []
    for m in re.finditer(r'POLYGON\s+(?:\w+\s+)?([-\d.\s]+)', str(s_region), re.I):
        v = np.array(m.group(1).split(), float)
        if v.size >= 6:
            polys.append(v[: v.size // 2 * 2].reshape(-1, 2))
    return polys


def _inside(poly, ra, dec):
    """Ray-casting point-in-polygon in a local tangent frame (small fields)."""
    ra0, dec0 = poly[:, 0].mean(), poly[:, 1].mean()
    k = np.cos(np.radians(dec0))
    x, y = (poly[:, 0] - ra0) * k, poly[:, 1] - dec0
    px, py = (ra - ra0) * k, dec - dec0
    inside = np.zeros(np.shape(px), bool)
    j = len(x) - 1
    for i in range(len(x)):
        cross = ((y[i] > py) != (y[j] > py)) & (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-30) + x[i])
        inside ^= cross
        j = i
    return inside


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', default=os.path.join(HERE, 'catalogs', 'hickson_compact_groups.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'hickson_hst_selection.csv'))
    ap.add_argument('--skip-have', action='store_true', help='skip groups with data already in mastDownload/')
    ap.add_argument('--groups', nargs='*', default=None, help='only these groups, e.g. "HCG 80" (default: all)')
    ap.add_argument('--append', action='store_true', help='append to an existing selection file')
    ap.add_argument('--multi', action='store_true',
                    help='several pointings per group (greedy cover of labelled members); images already in '
                         'mastDownload/ are counted first')
    a = ap.parse_args()
    local = {os.path.basename(d) for d in glob.glob(os.path.join(HERE, 'mastDownload', '*', '*'))}

    cat = pd.read_csv(a.catalog, comment='#')
    cat['labelled'] = cat['morph_gt'].notna() & (cat['morph_gt'].astype(str).str.len() > 0)
    have = set()
    if a.skip_have:
        import sys
        sys.path.insert(0, HERE)
        import galaxy_pipeline as gp
        cfg = gp.load_config(os.path.join(HERE, 'pipeline_config.json'))
        inc, _ = gp.discover_images(cfg['data_dir'], cfg, gp.load_catalogs(cfg['catalog_dir']), verbose=False)
        have = {r.field_name for r in inc}

    todo = [(grp, g) for grp, g in cat.groupby('group', sort=False)
            if grp not in have and not (a.groups and grp not in a.groups)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:                  # MAST queries in parallel (~3 min for 99 groups)
        answers = {grp: (o, err) for grp, o, err in ex.map(_query, todo)}
    rows = []
    for grp, g in todo:
        o, err = answers[grp]
        if o is None:
            print(f'{grp}: query failed ({err})')
            continue
        if len(o) == 0:
            continue
        o = o[o['obs_id'].astype(str).str.startswith('hst_') & o['instrument_name'].map(_instr_rank).notna()
              & o['filters'].isin(FILTER_PRIORITY)]
        lab = g[g['labelled']]
        cands = []
        for _, r in o.iterrows():
            polys = _polygons(r['s_region'])
            if not polys:
                continue
            ins = np.zeros(len(lab), bool)
            for poly in polys:
                ins |= _inside(poly, lab['ra'].to_numpy(), lab['dec'].to_numpy())
            if ins.any():
                cands.append((r, set(lab['galaxy_id'][ins])))
        covered, picks = set(), []
        if a.multi:                       # images already on disk are picked first
            for r, mem in cands:
                if str(r['obs_id']) in local and mem - covered:
                    covered |= mem
                    picks.append((r, mem, True))
        while True:
            best = None
            for r, mem in cands:
                new = mem - covered
                if len(new) < MIN_MEMBERS or any(r['obs_id'] == q[0]['obs_id'] for q in picks):
                    continue
                key = (len(new), -_instr_rank(r['instrument_name']),
                       -FILTER_PRIORITY.index(r['filters']), float(r['t_exptime'] or 0))
                if best is None or key > best[0]:
                    best = (key, r, mem)
            if best is None:
                break
            covered |= best[2]
            picks.append((best[1], best[2], False))
            if not a.multi:
                break
        for r, mem, on_disk in picks:
            rows.append({'group': grp, 'obs_id': r['obs_id'], 'instrument': r['instrument_name'],
                         'filter': r['filters'], 'exptime': r['t_exptime'], 'n_labelled_covered': len(mem),
                         'members_covered': ' '.join(sorted(mem)), 'n_labelled_in_group': len(lab),
                         'on_disk': on_disk})
            print(f"{grp:8s} {r['instrument_name']:10s} {r['filters']:7s} {r['t_exptime']:7.0f}s  "
                  f"{len(mem)}/{len(lab)} labelled members  {r['obs_id']}{'  (on disk)' if on_disk else ''}",
                  flush=True)
    sel = pd.DataFrame(rows)
    if a.append and os.path.exists(a.out):
        old = pd.read_csv(a.out)
        sel = pd.concat([old[~old['group'].isin(sel['group'] if len(sel) else [])], sel], ignore_index=True)
    sel.to_csv(a.out, index=False)
    n_cov = sum(len(set(' '.join(g['members_covered']).split())) for _, g in sel.groupby('group')) if len(sel) else 0
    print(f'\n{sel["group"].nunique() if len(sel) else 0} group(s), {len(sel)} image(s); {n_cov} distinct '
          f'labelled members covered. Selection written to {a.out}')
    todo = sel[~sel['on_disk'].astype(bool)] if len(sel) and 'on_disk' in sel else sel
    if len(todo):
        print('\nDownload with:\n  python fetch_mast_products.py ' + ' '.join(todo['obs_id']))


if __name__ == '__main__':
    main()
