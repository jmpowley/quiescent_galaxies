import os

from astropy.io import fits

from scalpel.loading import load_cube_data

# Load cube data
# -- full cube
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
    "psf_dir" : None,
    "psf_name" : None,
    "psf_ext" : None,
    "wave_min" : None,
    "wave_max" : None,
}
wave, cube, cube_err = load_cube_data(**cube_kwargs)
# -- cropped cube
# Load cube data
cube_kwargs = {
    "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/gs-9209/cubes",
    "data_name" : "GS9209-4.67_jw3659_px0.05_dr_ODfde95.0_VSC_AMRC_mMSA_m2ff_xy96_CTX1236_v1.15.1_g235h-f170lp_cgs_s3d.fits",
    "data_ext" : "SCI",
    "wave_from_hdr" : True,
    "in_wave_units" : None,
    "out_wave_units" : None,
    "centre" : (44, 40),
    "width" : 20,
    "height" : 20,
    "psf_dir" : None,
    "psf_name" : None,
    "psf_ext" : None,
    "wave_min" : None,
    "wave_max" : None,
}
wave, cube_crop, cube_err_crop = load_cube_data(**cube_kwargs)


# Calculate SNR for cube
snr_cube = cube / cube_err

snr_cube_crop = cube_crop / cube_err_crop

# Save cubes
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/gs-9209/cubes"
# -- full cube
pri_hdu = fits.PrimaryHDU()
cube_hdu = fits.ImageHDU(data=snr_cube, name="DATA")
hdul = fits.HDUList([pri_hdu, cube_hdu])
out_name = "gs-9209_cube_snr.fits"
out_path = os.path.join(out_dir, out_name)
hdul.writeto(out_path, overwrite=True)
# -- cropped cube
pri_hdu = fits.PrimaryHDU()
cube_hdu = fits.ImageHDU(data=snr_cube_crop, name="DATA")
hdul = fits.HDUList([pri_hdu, cube_hdu])
out_name = "gs-9209_cube_cropped_snr.fits"
out_path = os.path.join(out_dir, out_name)
hdul.writeto(out_path, overwrite=True)