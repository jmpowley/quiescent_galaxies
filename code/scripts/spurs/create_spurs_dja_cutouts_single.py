import os
import warnings
warnings.filterwarnings('ignore')
import argparse

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import download_file

# Define paths
base_url = 'https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/'

parser = argparse.ArgumentParser()
parser.add_argument("--filter", type=str)
args = parser.parse_args()
filter = args.filter

print(f"Loading data for filter: {filter}...")

downloaded = False
for i in range(5, -1, -1):
    if not downloaded:
        try:
            root = f"abell2744clu-grizli-v7.{i}"
            file = f'{root}-{filter}_drc_sci.fits.gz'
            url = base_url + file
            hdul = fits.open(url)  # open with memory map
            downloaded = True
        except Exception:
            pass

# Load co-ordinate data
data_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/qso1_spurs"
data_name = "9214.csv"
coord_df = pd.read_csv(os.path.join(data_dir, data_name))
ids = coord_df["ID"]
ra_vals = coord_df["RA"]
dec_vals = coord_df["Dec"]

# Define output paths
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/spurs/dja_cutouts"
fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/spurs/dja_cutouts"

# Use cached image + wcs (created once)
hdu0 = hdul[0]
data = hdu0.data
wcs = WCS(hdu0.header)

# Create HDUList
pri_hdu = fits.PrimaryHDU()
hdus = [pri_hdu]

# Loop over each galaxy
for i, (id, ra, dec) in enumerate(zip(ids, ra_vals, dec_vals)):
    print(f"ID: {id} ({i+1}/{np.size(ids)})")

    # Create cutout
    pos = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    size = 1 * u.arcsec
    cutout = Cutout2D(data, pos, size, wcs=wcs)

    # Save cutout to hdulist
    hdu = fits.ImageHDU(data=cutout.data, header=cutout.wcs.to_header(), name=f"{id:06d}")
    hdus.append(hdu)

# Save HDUList
out_name = f"spurs_{filter.upper()}_cutouts.fits"
hdul = fits.HDUList(hdus)
hdul.writeto(os.path.join(out_dir, out_name), overwrite=True)
print(f"Saved photometry for filter {filter} as: {out_name}")