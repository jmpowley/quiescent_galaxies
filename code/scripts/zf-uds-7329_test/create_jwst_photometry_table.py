import numpy as np
from astropy.table import Table
import astropy.units as u
import os

# Load filter names
jwst_filter_names = np.array(['f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w'])

# Photometry data used by Turner et al.
phot = np.array([25.567, 24.299, 22.826, 22.154, 21.827, 21.616, 21.546])

# Photometry error used (both errors and noise floor)
noise_floor = True
if noise_floor:
    phot_err = np.array([.05, .05, .05, .05, .05, .05, .05])
else:
    phot_err = np.array([.034, .02, .02, .02, .02, .02, .02]) # min error ~0.02

# Build the table
tb = Table()
tb['FILTER'] = jwst_filter_names
tb['FLUX'] = phot * u.mag  # assign magnitude units
tb['ERR'] = phot_err * u.mag

# Save as an astropy Table
out_dir = '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329_test/photometry'
tb_name = 'zf-uds-7329_nircam_photometry.fits'
out_path = os.path.join(out_dir, tb_name)
tb.write(out_path, overwrite=True)