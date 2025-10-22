import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from astropy.utils.data import download_file
from astropy.io import fits
from astropy.units import UnitBase, Quantity
from astropy.io.fits.card import Undefined

from grizli import utils

from msaexp.spectrum import SpectrumSampler

# -------------------------------
# Functions for cleaning metadata
# -------------------------------
def sanitize_value(v):
    """Cleans a dictionary containing astropy quantities, units or undefined value before being saved to a FITS header
    """
    if isinstance(v, UnitBase):
        return v.to_string()
    if isinstance(v, Quantity):
        return f"{v.value} {v.unit.to_string()}"
    if isinstance(v, Undefined):
        return None
    return v

def clean_meta_dict(meta_dict):
    cleaned = {}
    for k, v in meta_dict.items():
        new_key = str(k).upper()  # upper-case the key
        cleaned[new_key] = sanitize_value(v)
    return cleaned

# ----------------------------
# Functions to write FITS HDUs
# ----------------------------
def return_primary_hdu(spec):
    """Turns cleaned metadata dictionary into FITS header
    """

    # Make FITS header from clean dict
    hdr_dict = clean_meta_dict(spec.meta)
    hdr = fits.Header(hdr_dict)

    # Convert to empty HDU
    pri_hdu = fits.PrimaryHDU(data=None, header=hdr)

    return pri_hdu

def return_data_hdulist(spec):
    """Uses `msaexp.spectrum.SpectrumSampler` output to produce multiple extensions of spectrum data to save as FITS file

    Adds units from metadata to header of each HDU
    """

    # Add units to spectra
    spec['wave'].unit = spec.meta.get('waveunit')
    spec['flux'].unit = spec.meta.get('fluxunit')
    spec['err'].unit = spec.meta.get('fluxunit')

    # Store data in HDUs
    wave_hdu = fits.ImageHDU(data=spec['wave'].value, name='WAVELENGTH')
    wave_hdu.header['BUNIT'] = str(spec['wave'].unit)

    flux_hdu = fits.ImageHDU(data=spec['flux'].value, name='DATA')
    flux_hdu.header['BUNIT'] = str(spec['flux'].unit)

    err_hdu = fits.ImageHDU(data=spec['err'].value, name='ERR')
    err_hdu.header['BUNIT'] = str(spec['wave'].unit)

    mask_hdu = fits.ImageHDU(data=spec['valid'].value.astype(np.uint8), name='VALID')
    mask_hdu.header['COMMENT'] = "Mask stored as uint8 (0=false, 1=true)"

    return [wave_hdu, flux_hdu, err_hdu, mask_hdu]

# -----------------
# Create FITS files
# -----------------
def create_nirspec_fits_for_objid(src_id, src_name, out_dir, table_url, cache_downloads):
    
    # Load table
    dja_tb = utils.read_catalog(download_file(table_url, cache=cache_downloads), format='csv')

    # Load matching spectra
    obj_tb = dja_tb[dja_tb['srcid'] == src_id]
    # -- extract necessary info
    roots = obj_tb['root'].tolist()
    files = obj_tb['file'].tolist()
    gratings = obj_tb['grating'].tolist()
    filters = obj_tb['filter'].tolist()

    # Loop over each spectrum
    for (root, file, grating, filt) in zip(roots, files, gratings, filters):

        # Load spectrum
        fits_url = f"https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}"
        spec = SpectrumSampler(fits_url)

        # Produce FITS HDUs
        pri_hdu = return_primary_hdu(spec)
        data_hdulist = return_data_hdulist(spec)
        hdulist = fits.HDUList([pri_hdu, *data_hdulist])

        # Save file
        out_name = f"{src_id}_{grating.lower()}_{filt.lower()}_1D.fits"
        out_path = os.path.join(out_dir, out_name)
        hdulist.writeto(out_path, overwrite=True)

# -------------
# Main function
# -------------
def main():

    # DJA table info
    version = "v4.4"
    CACHE_DOWNLOADS = True
    # -- use the Zenodo release
    if version == "v4.4":
        URL_PREFIX = "https://zenodo.org/records/15472354/files/"
    TABLE_URL = f"{URL_PREFIX}/dja_msaexp_emission_lines_{version}.csv.gz"

    # Source information
    src_id = 191748  # source ID in DJA
    src_name = 'smiles-gs-191748'  # name of source
    out_dir = f"/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/{src_name}/spectra"  # where to save

    # Create fits files
    try:
        create_nirspec_fits_for_objid(src_id, src_name, out_dir, TABLE_URL, CACHE_DOWNLOADS)
        print("Spectra for {} saved to {}".format(src_name, out_dir))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()