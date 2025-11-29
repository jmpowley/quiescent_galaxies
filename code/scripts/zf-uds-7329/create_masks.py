import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

from loading import load_prism_data, load_grating_data
from conversions import convert_wave_A_to_um, convert_wave_um_to_A

def create_1d_wave_mask(wave, wave_units : str, mask_dict : dict, mask_units, default : bool, redshift_lines : bool, zred : float, return_ints : bool):
    """Creates a boolean mask
    
    Parameters
    ----------
    wave : array-like
        array of wavelength vectors to use in the mask
    wave_units : str
        units of the wavelength range
    mask_units : str
        units of the mask dictionary. If different from wave_units then will apply conversion
    default : bool
        boolean value which represents *not* being masked
    return_ints : bool
        return mask as array of integers rather than array of booleans (0 is False, 1 is True)

    Returns
    -------
    mask : array
        mask following the syntax of mask_dict with the default boolean/int as true
    
    mask_dict should have the following keys and values:
    - 'range' is a wavelength range (low, high) or a list of [low, high]
    - 'line' is the central wavelength and then the width each side [line, width] or a list of those
    All items in mask_dict will be assumed to be at the same redshift 
    """

    # Create mask using default argument (True/False)
    wave = np.asarray(wave)
    mask = np.full(wave.shape, default)

    # Redshift lines
    if redshift_lines:
        new_mask = {}
        for key, val in mask_dict.items():
            if key != 'line' or val is None:
                new_mask[key] = val
                continue
            new_mask['line'] = [[entry[0] * (1+zred), entry[1]] for entry in val]  # only redshift line centre
        mask_dict = new_mask

    # Convert values
    new_mask = {}
    # -- no conversion
    if wave_units == mask_units:
        converted = mask_dict.copy()
    # -- microns to angstroms
    elif wave_units == "um" and mask_units == "A":
        converted = {}
        for key, val in mask_dict.items():
            if val is None:
                converted[key] = None
                continue
            if key == 'range':
                converted['range'] = [[convert_wave_A_to_um(lo), convert_wave_A_to_um(hi)] for lo, hi in val]
            elif key == 'line':
                converted['line']  = [[convert_wave_A_to_um(line), convert_wave_A_to_um(width)] for line, width in val]
    # -- angstroms to microns
    elif wave_units == "A" and mask_units == "um":
        converted = {}
        for key, val in mask_dict.items():
            if val is None:
                converted[key] = None
                continue
            if key == 'range':
                converted['range'] = [[convert_wave_um_to_A(lo), convert_wave_um_to_A(hi)] for lo, hi in val]
            elif key == 'line':
                converted['line']  = [[convert_wave_um_to_A(center), width] for center, width in val]
    # -- incorrect units
    else:
        raise ValueError(f"wave_units {wave_units} or mask_units {mask_units} are not accepted. Please use 'um' or 'A'")
    mask_dict = converted

    # Apply mask dictionary to mask array
    eps = 1e-12
    for key, val in mask_dict.items():
        key_mask = np.zeros(wave.shape, dtype=bool)
        # -- range entries
        if key == "range":
            for entry in val:
                lo, hi = entry
                key_mask |= (wave >= lo - eps) & (wave <= hi + eps)
        # -- line entries
        elif key == "line":
            for entry in val:
                line, width = entry
                key_mask |= (np.abs(wave - line) <= width + eps)
        else:
            raise ValueError(f"Unknown mask key: {key}")

        # Combine key_mask with overall mask
        mask[key_mask] = not default

    if return_ints:
        return mask.astype(int)
    else:
        return mask

def save_wave_mask(mask, out_dir : str,out_file : str, mask_ext : str):
    """Save mask as FITS file in a given output name"""

    # Create path
    out_path = os.path.join(out_dir, out_file)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Create FITS file
    # TODO: Make more informative header
    pri_hdu = fits.PrimaryHDU()  # leave empty
    mask_hdu = fits.ImageHDU(data=mask, name=mask_ext)
    hdul = fits.HDUList([pri_hdu, mask_hdu])

    hdul.writeto(out_path, overwrite=True)
    print(f"Mask saved to {out_path}")

def plot_mask(wave, flux, mask, zred=None, color='gray', alpha=0.3, default=True):
    """
    Plots shaded regions for contiguous wavelength regions where the mask is default.
    """
    wave = np.asarray(wave)
    mask = np.asarray(mask).astype(bool)
    N = mask.size

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Optionally invert mask to highlight where mask is not default
    mask_bool = True
    if default is True:
        mask = ~mask
        mask_bool = False

    # Identify transitions between masked/unmasked
    diff = np.diff(mask.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends   = list(np.where(diff == -1)[0] + 1)
    # -- if mask begins True, then the first region starts at index 0
    if mask[0]:
        starts = [0] + starts
    # -- if mask ends True, the last region ends at index N-1
    if mask[-1]:
        ends = ends + [N - 1]

    # Ensure indices are in bounds and ints
    start_indices = [int(max(0, min(N - 1, s))) for s in starts]
    end_indices   = [int(max(0, min(N - 1, e))) for e in ends]

    # Plot spectra
    ax.step(wave, flux, color="black", where="mid")

    # Plot mask
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        ax.axvspan(wave[start], wave[end], color=color, alpha=alpha, label=f"Mask = {mask_bool}" if i == 0 else None)

    # Prettify
    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(1, 6))
    if zred is not None:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        tick_step = 0.2
        start_tick = np.ceil((wave.min()/(1+zred)) / tick_step) * tick_step
        end_tick = np.floor((wave.max()/(1+zred)) / tick_step) * tick_step
        top_ticks_rest = np.arange(start_tick, end_tick+tick_step, tick_step)
        top_ticks_obs = top_ticks_rest * (1 + zred)
        ax_top.set_xticks(top_ticks_obs)
        ax_top.set_xticklabels([f"{t:.1f}" for t in top_ticks_rest])
    ax.set_xlabel(r'Observed Wavelength [$\mu$m]', size=18)
    ax.set_ylabel(r'$f_\lambda~[~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$', size=18)
    ax.legend()

    return fig

# Load spectra
obs_kwargs = {

    "phot_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry",
        "data_name" : "007329_nircam_photometry.fits",
        "data_ext" : "DATA",
        "in_flux_units" : "magnitude",
        "out_flux_units" : "cgs",
        "snr_limit" : 20.0,
        "prefix" : "phot",
    },

    "prism_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_prism_clear_v3.1_extr5_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "uds7329_nirspec_prism_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-19,
        "rescale_factor" : 1.86422,
        "snr_limit" : 20.0,
        "prefix" : "prism",
    },

    "grat1_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_g140m_f100lp_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g140m_disp.fits",
        "in_wave_units" : "um",
        "out_wave_units" : "um",
        "in_flux_units" : "ujy",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-19,
        "rescale_factor" : 1.86422,
        "snr_limit" : 20.0,
        "prefix" : "g140m",
    },

    "grat2_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_g235m_f170lp_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g235m_disp.fits",
        "in_wave_units" : "um",
        "out_wave_units" : "um",
        "in_flux_units" : "ujy",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-19,
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "prefix" : "g235m",
    },

    "grat3_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "data_name" : "007329_g395m_f290lp_1D.fits",
        "data_ext" : "DATA",
        "mask_dir" : None,
        "mask_name" : None,
        "mask_ext" : None,
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g140m_disp.fits",
        "in_wave_units" : "um",
        "out_wave_units" : "um",
        "in_flux_units" : "ujy",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-19,
        "rescale_factor" : 1.86422,
        "snr_limit" : 20.0,
        "prefix" : "g395m",
        "add_jitter" : True,
        "include_outliers" : True,
        "fit_obs" : False,
    },
}

# Redshift information
zred = 3.19

# Load data
prism_kwargs = obs_kwargs['prism_kwargs']
grat1_kwargs = obs_kwargs['grat1_kwargs']
grat2_kwargs = obs_kwargs['grat2_kwargs']
grat3_kwargs = obs_kwargs['grat3_kwargs']

prism_wave_um, prism_flux, prism_err = load_prism_data(**prism_kwargs)
grat1_wave_um, grat1_flux, grat1_err = load_grating_data(**grat1_kwargs)
grat2_wave_um, grat2_flux, grat2_err = load_grating_data(**grat2_kwargs)
grat3_wave_um, grat3_flux, grat3_err = load_grating_data(**grat3_kwargs)

# Create prism mask
prism_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(prism_wave_um.min()), 4000*(1+zred)],  # mask up to 4000A as C3K not accurate
        ],
    'line' : [
        [5900, 400],  # NaD
        [3934, 400],  # CaK
        ],
}
prism_mask = create_1d_wave_mask(wave=prism_wave_um, wave_units="um", mask_dict=prism_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, zred=zred, return_ints=True)

# Create grat1 mask
grat1_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(grat1_wave_um.min()), 4000*(1+zred)],  # mask up to 4000A rest-frame as C3K not accurate
        [convert_wave_um_to_A(grat1_wave_um.min()), convert_wave_um_to_A(1)],  # mask high flux values at start of range
        ],
    'line' : [
        [5900, 400],  # NaD
        [3934, 400],  # CaK
        ],
}
grat1_mask = create_1d_wave_mask(wave=grat1_wave_um, wave_units="um", mask_dict=grat1_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, zred=zred, return_ints=True)

# Create grat2 mask
grat2_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(grat2_wave_um.min()), 4000*(1+zred)],  # mask up to 4000A rest-frame as C3K not accurate
        ],
    'line' : [
        [5900, 400],  # NaD
        [3934, 400],  # CaK
        ],
}
grat2_mask = create_1d_wave_mask(wave=grat2_wave_um, wave_units="um", mask_dict=grat2_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, zred=zred, return_ints=True)

# Create grat3 mask
grat3_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(grat3_wave_um.min()), convert_wave_um_to_A(2.75)],  # mask high flux values at start of range
        [convert_wave_um_to_A(5.4), convert_wave_um_to_A(grat3_wave_um.max())],  # mask high flux values at start of range
        ],
    'line' : [
        ],
}
grat3_mask = create_1d_wave_mask(wave=grat3_wave_um, wave_units="um", mask_dict=grat3_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, zred=zred, return_ints=True)

# Plot masks
# -- prism
fig = plot_mask(prism_wave_um, prism_flux, prism_mask, zred=zred)
# -- grat1
fig = plot_mask(grat1_wave_um, grat1_flux, grat1_mask, zred=zred)
# -- grat2
fig = plot_mask(grat2_wave_um, grat2_flux, grat2_mask, zred=zred)
# -- grat3
fig = plot_mask(grat3_wave_um, grat3_flux, grat3_mask, zred=zred)

plt.show()

# Save masks
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/wave_masks"
# -- prism
out_file = prism_kwargs["data_name"].replace(".fits", "_mask.fits")
save_wave_mask(prism_mask, out_dir, out_file, mask_ext="MASK")
# -- grat1
out_file = grat1_kwargs["data_name"].replace(".fits", "_mask.fits")
save_wave_mask(grat1_mask, out_dir, out_file, mask_ext="MASK")
# -- grat2
out_file = grat2_kwargs["data_name"].replace(".fits", "_mask.fits")
save_wave_mask(grat2_mask, out_dir, out_file, mask_ext="MASK")
# -- grat3
out_file = grat3_kwargs["data_name"].replace(".fits", "_mask.fits")
save_wave_mask(grat3_mask, out_dir, out_file, mask_ext="MASK")