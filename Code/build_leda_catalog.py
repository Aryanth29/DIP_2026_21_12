"""
neighbour-model run: morphology labels for NON-Hickson galaxies that already lie in our
frames, from HyperLEDA (Makarov et al. 2014), which gives a numeric
de Vaucouleurs type t (with uncertainty e_t) averaged over the literature.

For every field discovered under mastDownload/, HyperLEDA is queried in the
field's RA/Dec box and each galaxy with a measured type is labelled with the
same 3-class scheme as the Hickson members. The rule is fixed BEFORE any
evaluation and is the same for every field:

    objtype 'G' (galaxy), t and e_t both given, e_t <= MAX_E_T (= 1.5 types)
    t <= -1          -> E    (E, S0-, S0, S0+)
    1 <= t <= 7      -> S    (Sa .. Sd)
    t >= 8           -> Irr  (Sdm, Sm, Im)
    -1 < t < 1, 7 < t < 8   -> unlabelled (S0/a and Sd/Sdm boundaries)

Galaxies within EXCLUDE_ARCSEC of any Hickson / Stephan's Quintet catalogue
position, or inside a Hickson member's optical extent (distance < half its
catalogued diameter), are skipped: they are the same object (HyperLEDA and
Hickson positions of big galaxies differ by > 5", e.g. NGC 6027B, NGC 1741)
or a part of it. Each row
gets match_radius_arcsec = MATCH_ARCSEC, so the pipeline only matches it to
the segment that contains it or a galaxy centred within 2" -- a small
background galaxy must not be matched to a neighbour 10" away.
The 'group' column is the field's Hickson group, so leave-one-group-out
cross-validation keeps these galaxies with the rest of their frame.

    python build_leda_catalog.py            # writes catalogs/hyperleda_types.csv
"""
import argparse
import io
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galaxy_pipeline as gp  # noqa: E402

LEDA_URL = 'http://atlas.obs-hp.fr/hyperleda/fG.cgi'
OUT_NAME = 'hyperleda_types.csv'
MAX_E_T = 1.5
EXCLUDE_ARCSEC = 5.0
MATCH_ARCSEC = 2.0


def leda_box(ra0, ra1, dec0, dec1, timeout=120):
    """All HyperLEDA objects in an RA/Dec box (degrees)."""
    sql = (f'al2000 >= {ra0 / 15.0:.7f} and al2000 <= {ra1 / 15.0:.7f} and '
           f'de2000 >= {dec0:.7f} and de2000 <= {dec1:.7f}')
    q = {'n': 'meandata', 'c': 'o', 'of': '1,leda,simbad', 'nra': 'l', 'nakd': '1',
         'd': 'pgc,objname,al2000,de2000,t,e_t,bt,objtype,v', 'sql': sql, 'ob': '', 'a': 'csv[,]'}
    url = LEDA_URL + '?' + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        txt = r.read().decode('utf-8', 'replace')
    lines = [ln for ln in txt.splitlines() if ln and not ln.startswith('#')]
    if len(lines) < 2:
        return pd.DataFrame(columns=['pgc', 'objname', 'al2000', 'de2000', 't', 'e_t', 'bt', 'objtype', 'v'])
    return pd.read_csv(io.StringIO('\n'.join(lines)))


def leda_class(t, e_t):
    if not (np.isfinite(t) and np.isfinite(e_t)) or e_t > MAX_E_T:
        return None
    if t <= -1:
        return 'E'
    if 1 <= t <= 7:
        return 'S'
    if t >= 8:
        return 'Irr'
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(HERE, 'pipeline_config.json'))
    a = ap.parse_args()
    cfg = gp.load_config(a.config)
    cat_dir = cfg['catalog_dir'] if os.path.isabs(cfg['catalog_dir']) else os.path.join(HERE, cfg['catalog_dir'])
    data_dir = cfg['data_dir'] if os.path.isabs(cfg['data_dir']) else os.path.join(HERE, cfg['data_dir'])
    cat = gp.load_catalogs(cat_dir)
    cat = cat[cat['catalog_file'] != OUT_NAME]                    # Hickson + SQ only
    inc, _ = gp.discover_images(data_dir, cfg, cat, verbose=False)

    by_field = {}
    for r in inc:
        by_field.setdefault(r.field_name, []).append(r)
    rows = []
    for fname, recs in sorted(by_field.items()):
        group = fname.split(' (')[0]
        corners = np.vstack([r.corners for r in recs])
        ra0, ra1 = corners[:, 0].min(), corners[:, 0].max()
        dec0, dec1 = corners[:, 1].min(), corners[:, 1].max()
        try:
            t = leda_box(ra0, ra1, dec0, dec1)
        except Exception as e:
            print(f'{fname}: HyperLEDA query failed ({e})')
            continue
        t['ra'] = t['al2000'] * 15.0
        t['dec'] = t['de2000']
        n_all = len(t)
        t = t[t['objtype'].astype(str).str.strip() == 'G']
        t['morph_gt'] = [leda_class(x, y) for x, y in zip(pd.to_numeric(t['t'], errors='coerce'),
                                                           pd.to_numeric(t['e_t'], errors='coerce'))]
        n_typed = int(pd.to_numeric(t['t'], errors='coerce').notna().sum())
        t = t[t['morph_gt'].notna()]
        # inside at least one image footprint (bounding box per image)
        inside = np.zeros(len(t), bool)
        for r in recs:
            c = r.corners
            inside |= ((t['ra'] >= c[:, 0].min()) & (t['ra'] <= c[:, 0].max()) &
                       (t['dec'] >= c[:, 1].min()) & (t['dec'] <= c[:, 1].max())).to_numpy()
        t = t[inside]
        # drop galaxies that are already Hickson / SQ catalogue entries
        if len(t) and len(cat):
            cosd = np.cos(np.radians(t['dec'].to_numpy()))[:, None]
            d = np.hypot((t['ra'].to_numpy()[:, None] - cat['ra'].to_numpy()[None, :]) * cosd,
                         t['dec'].to_numpy()[:, None] - cat['dec'].to_numpy()[None, :]) * 3600.0
            rad = np.maximum(EXCLUDE_ARCSEC, 0.5 * pd.to_numeric(cat.get('diam_arcsec', pd.Series(np.nan, index=cat.index)),
                                                                 errors='coerce').fillna(0).to_numpy())
            t = t[(d > rad[None, :]).all(axis=1)]
        print(f'{fname:22s} HyperLEDA objects in box {n_all:4d}, with a type {n_typed:3d}, '
              f'labelled & new: {len(t):3d}  ' + ' '.join(f'{k}:{v}' for k, v in t['morph_gt'].value_counts().items()))
        for _, x in t.iterrows():
            rows.append({'group': group, 'galaxy_id': f"PGC{int(x['pgc'])}", 'ra': round(x['ra'], 6),
                         'dec': round(x['dec'], 6), 'morph_gt': x['morph_gt'], 'leda_t': x['t'],
                         'leda_e_t': x['e_t'], 'bt': x['bt'], 'v_helio': x['v'],
                         'alt_name': str(x['objname']).strip(), 'label_source': 'HyperLEDA',
                         'match_radius_arcsec': MATCH_ARCSEC})
    out = pd.DataFrame(rows).drop_duplicates('galaxy_id')
    path = os.path.join(cat_dir, OUT_NAME)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('# Non-Hickson galaxies inside our HST frames with a HyperLEDA morphological type t\n'
                 f'# (atlas.obs-hp.fr/hyperleda, meandata). Built by build_leda_catalog.py. Rule fixed before\n'
                 f'# evaluation: e_t <= {MAX_E_T}; t <= -1 -> E; 1 <= t <= 7 -> S; t >= 8 -> Irr; else unlabelled.\n'
                 f'# Galaxies within {EXCLUDE_ARCSEC}" of a Hickson/SQ entry or inside half its diameter are excluded. group = field\'s Hickson group.\n')
        out.to_csv(fh, index=False)
    print(f'\n{len(out)} labelled HyperLEDA galaxies written to {path}  '
          + ' '.join(f'{k}:{v}' for k, v in out['morph_gt'].value_counts().items()) if len(out) else 'none')


if __name__ == '__main__':
    main()
