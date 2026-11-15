"""
neighbour-model run evaluation: compares the development stage measurement (no neighbour models,
columns *_v12) with the neighbour-model run measurement (neighbour models subtracted)
on the SAME labelled galaxies, using the pipeline's own procedure (rule
family chosen by LOGO-CV, thresholds fitted per fold). Also scores the
frozen development stage rule on everything it was not fitted on.

    python evaluate_neighbour_model.py <cas_features_all_images.csv> [--out report.md]

Requires the run to have been made with "measure_legacy": true.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galaxy_pipeline as gp  # noqa: E402

DEVELOPMENT_RULE_FILE = os.path.join(HERE, 'classification_rule_frozen_18groups.json')


def variant(df, which):
    """Copy of df whose C/A/S/measurable columns are the chosen measurement."""
    d = df.copy()
    if which == 'v12':
        for k in ('C', 'A', 'S', 'gini', 'm20', 'sn_per_pixel'):
            d[k] = d[f'{k}_v12']
        d['measurable'] = d['measurable_v12'].astype(bool)
    return d


def cv_score(lab):
    """The pipeline's procedure: LOGO-CV for both rule families, simpler wins ties."""
    truth = lab['morph_gt'].to_numpy(object)
    cv = {}
    for fam in ('CA', 'CAS'):
        pr = gp.logo_cv(lab, fam).to_numpy(object)
        cv[fam] = (gp._balanced_acc(truth, pr), float(np.mean(pr == truth)), pr)
    fam = 'CA' if cv['CA'][:2] >= cv['CAS'][:2] else 'CAS'
    return fam, cv[fam][1], cv[fam][0], cv[fam][2], cv


def labelled(df, which):
    d = variant(df, which)
    lab = gp.labelled_table(d)
    return lab[lab['measurable'].astype(bool)]


def fmt(x):
    return f'{100 * x:.1f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, low_memory=False)
    if 'label_source' not in df:
        df['label_source'] = 'hickson'
    df['label_source'] = df['label_source'].fillna('').astype(str)
    L = []
    P = lambda *s: L.append(' '.join(str(x) for x in s))  # noqa: E731

    fr = json.load(open(DEVELOPMENT_RULE_FILE))
    old_groups = set(fr['groups_fitted'])
    labs = {w: labelled(df, w) for w in ('v12', 'model')}
    # compare on the galaxies measurable in BOTH variants (paired)
    ids = sorted(set(labs['v12']['catalog_id']) & set(labs['model']['catalog_id']))
    P(f'Labelled galaxies measurable: v12 {len(labs["v12"])}, model {len(labs["model"])}, both {len(ids)}')

    subsets = {
        'all labels': lambda t: t,
        'Hickson + SQ labels only': lambda t: t[~t['label_source'].str.contains('HyperLEDA')],
        'development stage groups, Hickson + SQ only': lambda t: t[t['group'].isin(old_groups)
                                                         & ~t['label_source'].str.contains('HyperLEDA')],
    }
    P('\n## LOGO cross-validated accuracy, development stage vs neighbour-model run measurement (same galaxies)\n')
    P('| sample | n | groups | v12 measurement | neighbour models | v12 → model: fixed / broken |')
    P('|---|---|---|---|---|---|')
    res = {}
    for name, sel in subsets.items():
        row = {}
        for w in ('v12', 'model'):
            t = sel(labs[w][labs[w]['catalog_id'].isin(ids)]).sort_values('catalog_id').reset_index(drop=True)
            fam, acc, bal, pr, cv = cv_score(t)
            row[w] = (t, fam, acc, bal, pr)
        t0, t1 = row['v12'][0], row['model'][0]
        ok0 = row['v12'][4] == t0['morph_gt'].to_numpy(object)
        ok1 = row['model'][4] == t1['morph_gt'].to_numpy(object)
        res[name] = row
        P(f"| {name} | {len(t0)} | {t0['group'].nunique()} | {fmt(row['v12'][2])} (bal {fmt(row['v12'][3])}, "
          f"{row['v12'][1]}) | {fmt(row['model'][2])} (bal {fmt(row['model'][3])}, {row['model'][1]}) | "
          f"{int((~ok0 & ok1).sum())} / {int((ok0 & ~ok1).sum())} |")

    t_all = res['all labels']
    P('\n## Asymmetry of E galaxies (published E/S0), median [IQR]\n')
    for name, sel in subsets.items():
        e0 = sel(res[name]['v12'][0]); e0 = e0[e0['morph_gt'] == 'E']['A']
        e1 = sel(res[name]['model'][0]); e1 = e1[e1['morph_gt'] == 'E']['A']
        P(f"- {name}: v12 {e0.median():.3f} [{e0.quantile(.25):.3f}-{e0.quantile(.75):.3f}]  ->  "
          f"model {e1.median():.3f} [{e1.quantile(.25):.3f}-{e1.quantile(.75):.3f}]  (n={len(e1)})")
    t1 = t_all['model'][0].copy()
    t0 = t_all['v12'][0]
    t1['A_v12m'] = t0['A'].to_numpy()
    t1['crowded'] = t1['neighbour_frac'] > 0.15
    P('\nBy crowding (neighbour_frac > 0.15 of the 1.5 r_p aperture on other segments), median A per class:\n')
    P('| class | crowded | n | A v12 | A model |')
    P('|---|---|---|---|---|')
    for (c, cr), g in t1.groupby(['morph_gt', 'crowded']):
        P(f"| {c} | {cr} | {len(g)} | {g['A_v12m'].median():.3f} | {g['A'].median():.3f} |")

    P('\n## Confusion matrices (all labels, LOGO-CV)\n')
    for w in ('v12', 'model'):
        t, fam, acc, bal, pr = t_all[w]
        cm = pd.crosstab(pd.Series(t['morph_gt'].to_numpy(), name='published'),
                         pd.Series(pr, name='predicted')).reindex(index=list(gp.CLASSES),
                                                                  columns=list(gp.CLASSES), fill_value=0)
        P(f'{w} ({fam}):\n')
        P('```\n' + cm.to_string() + '\n```')

    # frozen development stage rule, development stage measurement, on galaxies it never saw
    dev_df = pd.read_csv(os.path.join(HERE, '..', 'Output', 'development_output', 'cas_features_all_images.csv'),
                      low_memory=False)
    seen = set(gp.labelled_table(dev_df).query('measurable == True')['catalog_id'])
    lab0 = labs['v12']
    unseen = lab0[~lab0['catalog_id'].isin(seen)]
    P(f'\n## Frozen development stage rule ({len(seen)} galaxies, {len(old_groups)} groups) applied unchanged, '
      f'development stage measurement, to galaxies it was not fitted on\n')
    if len(unseen):
        for name, u in [('galaxies in NEW groups', unseen[~unseen['group'].isin(old_groups)]),
                        ('new galaxies in development stage groups', unseen[unseen['group'].isin(old_groups)]),
                        ('all unseen galaxies', unseen)]:
            if len(u):
                pr = gp.predict_rule(u, fr)
                tr = u['morph_gt'].to_numpy(object)
                base = u['morph_gt'].value_counts().max() / len(u)
                P(f"- {name}: **{fmt(np.mean(pr == tr))}** ({int(np.sum(pr == tr))}/{len(u)}, balanced "
                  f"{fmt(gp._balanced_acc(tr, pr))}; majority-class baseline {fmt(base)}; "
                  f"{u['group'].nunique()} groups) -- " + ', '.join(f'{i} {t}->{q}' for i, t, q in
                                                              zip(u['catalog_id'], tr, pr)))
    else:
        P('- no unseen labelled galaxies')

    P('\n## Per-galaxy table (all labels)\n')
    tt = t_all['model'][0][['catalog_id', 'group', 'label_source', 'Filter', 'morph_gt', 'neighbour_frac']].copy()
    tt['A_v12'] = t_all['v12'][0]['A'].to_numpy()
    tt['A'] = t_all['model'][0]['A'].to_numpy()
    tt['C_v12'] = t_all['v12'][0]['C'].to_numpy()
    tt['C'] = t_all['model'][0]['C'].to_numpy()
    tt['pred_v12'] = t_all['v12'][4]
    tt['pred_model'] = t_all['model'][4]
    P(tt.round(3).to_markdown(index=False) if hasattr(tt, 'to_markdown') else tt.round(3).to_string(index=False))
    txt = '\n'.join(L)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt)
        tt.to_csv(os.path.splitext(a.out)[0] + '_galaxies.csv', index=False)


if __name__ == '__main__':
    main()
