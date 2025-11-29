import os

import pandas as pd

from astropy.io import fits

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

# Loop over each ID
for id in ids:
    print(f"ID: {id}")
    pri_hdu = fits.PrimaryHDU()
    hdus = [pri_hdu]

    # Loop over each filter
    for filter in jwst_filters:
        print(f"\tfilter: {filter.upper()}")
        file_name = f"spurs_{filter.upper()}_cutouts.fits"

        # Add each filter to ID HDUs
        hdul = fits.open(os.path.join(out_dir, file_name))
        hdu = fits.ImageHDU(hdul[f"{id:06d}"].data, header=hdul[f"{id:06d}"].header, name=filter.upper())
        hdus.append(hdu)

    # Save HDUList
    out_name = f"{id:06d}_cutouts.fits"
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(os.path.join(out_dir, out_name), overwrite=True)
    print(f"Saved photometry for ID {id} as: {out_name}")