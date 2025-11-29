import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

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

# Filters to plot
jwst_filters = [
    "f070w-clear",
    "f090w-clear",
    "f115w-clear",
    "f140m-clear",
    "f150w-clear",
    "f182m-clear",
    "f200w-clear",
    "f210m-clear",
    "f250m-clear",
    "f277w-clear",
    "f300m-clear",
    "f335m-clear",
    "f356w-clear",
    "f360m-clear",
    "f410m-clear",
    "f430m-clear",
    "f444w-clear",
    "f460m-clear",
    "f480m-clear",
    ]
n_filters = len(jwst_filters)

# Loop over each galaxy
for i, (id, ra, dec) in enumerate(zip(ids, ra_vals, dec_vals)):
    print(f"ID: {id} ({i+1}/{np.size(ids)})")

    file_name = f"{id:06d}_cutouts.fits"
    hdul = fits.open(os.path.join(out_dir, file_name))

    fig, axes = plt.subplots(nrows=n_filters, ncols=1, figsize=(4, 4*n_filters))

    for i, filter in enumerate(jwst_filters):
        print(f"\t{filter.upper()}")
        ax = axes[i]

        # Plot image
        hdu = hdul[filter.upper()]
        ax.imshow(hdu.data, origin="lower")
        ax.set_title(f"Object ID: {id} ({filter.upper()})")
    
    fig.tight_layout()

    # Save figure
    fig_name = f"{id:06d}_cutouts.pdf"
    fig.savefig(os.path.join(fig_dir, fig_name))