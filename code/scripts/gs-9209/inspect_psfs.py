import numpy as np

import matplotlib.pyplot as plt

import stpsf

from scalpel.loading import load_cube_data

def return_circular_gaussian_psf(wave_um):

    fwhm_as = 2.355 * 0.076 + 2.355 * 0.013 * wave_um * np.exp(-7.6 / wave_um)

    return fwhm_as

def return_major_axis_psf(wave_um):

    fwhm_as = 0.125 + 1.88 * wave_um * np.exp(-24.35 / wave_um)

    return fwhm_as
    
def return_minor_axis_psf(wave_um):

    fwhm_as = 0.094 + 0.20 * wave_um * np.exp(-12.48 / wave_um)

    return fwhm_as

# Load cube data
cube_kwargs = {
    "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/gs-9209/cubes",
    "data_name" : "GS9209-4.67_jw3659_px0.05_dr_ODfde95.0_VSC_AMRC_mMSA_m2ff_xy96_CTX1236_v1.15.1_g235h-f170lp_cgs_s3d.fits",
    "data_ext" : "SCI",
    "wave_from_hdr" : True,
    "in_wave_units" : None,
    "out_wave_units" : None,
    "centre" : None,
    "width" : None,
    "height" : None,
    "wave_min" : None,
    "wave_max" : None,
}
wave, cube, cube_err = load_cube_data(**cube_kwargs)

# Load PSFs
psf_minor = return_minor_axis_psf(wave)
psf_major = return_major_axis_psf(wave)
psf_circ = return_circular_gaussian_psf(wave)

# Load PSFs from STPSF
nrs = stpsf.NIRSpec()
nrs.mode = "IFU"
nrs.disperser = "G235H"
nrs.filter = "F170LP"
print("Band is", nrs.band)

waves = nrs.get_IFU_wavelengths(nlambda=50)

cube = nrs.calc_datacube(waves)

print(cube.info(), print(type(cube)))

# quickcube = nrs.calc_datacube_fast(waves)

index = 20
stpsf.display_psf(cube, ext=3, cube_slice=index,
                    title=f'NRS IFU cube slice {index}, $\\lambda$={cube[0].header["WAVELN"+str(index)]*1e6:.4} micron')

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.plot(wave, psf_circ, label="Circular")
ax.plot(wave, psf_major, label="Major axis")
ax.plot(wave, psf_minor, label="Minor axis")

# Prettify
ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu m]$", size=16)
ax.set_ylabel(r"${\rm FWHM}(\lambda)~[{\rm arcsec}]$", size=16)
ax.legend()

plt.show()

