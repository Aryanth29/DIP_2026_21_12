"""
Download calibrated, CR-combined science images from MAST into mastDownload/
so that galaxy_pipeline.py picks them up automatically on its next run.

    python fetch_mast_products.py <obs_id> [<obs_id> ...]
    python fetch_mast_products.py --dry-run hst_8717_01_wfpc2_pc_f814w_u63r01

Examples (the HCG 79 / HCG 87 products the notebook was waiting for, taken
from hcg79_observations.csv / hcg87_observations.csv):

    python fetch_mast_products.py hst_8717_01_wfpc2_pc_f814w_u63r01 hst_8538_04_wfpc2_pc_f814w_u5k504

Only drizzled/combined image products are fetched (HST DRZ/DRC, JWST I2D),
because those are already cosmic-ray rejected across exposures. Files land
at mastDownload/<mission>/<obs_id>/<file>.fits (astroquery's own layout).
Needs: astroquery + internet access to MAST.
"""
import argparse
import os

from astroquery.mast import Observations

HERE = os.path.dirname(os.path.abspath(__file__))
COMBINED = ['DRZ', 'DRC', 'I2D']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('obs_ids', nargs='+')
    ap.add_argument('--dest', default=HERE, help='folder that will contain mastDownload/ (default: Code/)')
    ap.add_argument('--dry-run', action='store_true', help='list the matching products, do not download')
    a = ap.parse_args()

    obs = Observations.query_criteria(obs_id=a.obs_ids)
    if len(obs) == 0:
        raise SystemExit(f'No MAST observations found for {a.obs_ids}')
    prods = Observations.get_product_list(obs)
    sci = Observations.filter_products(prods, productType='SCIENCE', extension='fits',
                                       productSubGroupDescription=COMBINED)
    sci = sci[[str(o) in a.obs_ids for o in sci['obs_id']]] if 'obs_id' in sci.colnames else sci
    if len(sci) == 0:
        raise SystemExit('Observations found, but no DRZ/DRC/I2D science products for them.')
    print(sci['obs_id', 'productFilename', 'productSubGroupDescription', 'size'])
    if a.dry_run:
        return
    manifest = Observations.download_products(sci, download_dir=a.dest)
    for row in manifest:
        print(f"{row['Status']:<10} {row['Local Path']}")
    print('\nDone. Run `python run_pipeline.py --list` to confirm the new images were discovered.')


if __name__ == '__main__':
    main()
