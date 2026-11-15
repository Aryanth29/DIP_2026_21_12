"""
Command-line runner for the generalised classical-DIP galaxy pipeline.

    python run_pipeline.py                 # every FITS image under mastDownload/
    python run_pipeline.py --list          # only show what was discovered
    python run_pipeline.py --only "HCG 90" # one field / file / TARGNAME pattern
    python run_pipeline.py --data-dir D:/fits --out-dir D:/results
    python run_pipeline.py --workers 6     # run the image jobs in 6 parallel processes

Drop any new HST/JWST FITS file anywhere under the data directory and re-run:
no code change is needed. See pipeline_config.json for optional settings.
No machine learning / deep learning is used anywhere in the pipeline.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galaxy_pipeline as gp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=os.path.join(HERE, 'pipeline_config.json'))
    ap.add_argument('--data-dir', default=None, help='folder searched recursively for *.fits')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--only', action='append', default=None,
                    help='glob on field name / file name / TARGNAME (repeatable)')
    ap.add_argument('--list', action='store_true', help='discover and print images, then exit')
    ap.add_argument('--rule', default=None,
                    help='classification_rule.json of an earlier run: apply that rule unchanged (hold-out test)')
    ap.add_argument('--workers', type=int, default=None,
                    help='parallel worker processes for the image jobs (default: config "workers" or 1)')
    a = ap.parse_args()

    if a.list:
        cfg = gp.load_config(a.config)
        cat = gp.load_catalogs(cfg.get('catalog_dir'))
        inc, exc = gp.discover_images(a.data_dir or cfg['data_dir'], cfg, cat)
        jobs = gp.build_jobs(inc, cfg)
        m = gp.manifest_frame(inc + exc)
        print(m.drop(columns=['path']).to_string(index=False))
        print(f'\n{len(inc)} image(s) -> {len(jobs)} job(s):')
        for j in jobs:
            print(f'  [{"primary" if j.is_primary else "       "}] {j.field_name:<20} {j.label}  '
                  f'({j.sky_area_arcmin2:.2f} arcmin^2)')
        return

    gp.run_all(a.config, data_dir=a.data_dir, output_dir=a.out_dir, only=a.only, show_figures=False,
               rule_file=a.rule, workers=a.workers)


if __name__ == '__main__':
    main()
