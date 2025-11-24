import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

from loading import load_prism_data, load_grating_data
from conversions import convert_wave_A_to_um, convert_wave_um_to_A

def create_1d_wave_mask(wave, wave_units : str, mask_dict : dict, mask_units, default : bool, redshift_lines : bool, z : float, return_ints : bool):
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
        # -- create temporary dict
        new_mask_dict = {}
        for key, val in mask_dict.items():
            if key != 'line' or val is None:
                new_mask_dict[key] = val
                continue
            new_mask_dict['line'] = [[entry[0] * (1.+z), entry[1]] for entry in val]  # only redshift line centre
        mask_dict = new_mask_dict

    print("After redshifting:")
    print(mask_dict)

    # Convert values
    # -- create temporary dict
    new_mask_dict = {}
    # -- microns to angstroms
    if wave_units == "um" and mask_units == "A":
        new_mask_dict = {
            key: [convert_wave_A_to_um(v) for v in val]
            for key, val in mask_dict.items()
        }
    # -- angstroms to microns
    elif wave_units == "A" and mask_units == "um":
        new_mask_dict = {
            key: [convert_wave_um_to_A(v) for v in val]
            for key, val in mask_dict.items()
        }
    # -- do nothing
    elif wave_units == mask_units:
        pass
    else:
        raise ValueError(f"wave_units {wave_units} or mask_units {mask_units} are not accepted. Please use 'um' or 'A'")
    mask_dict = new_mask_dict

    print("After conversion:")
    print(mask_dict)

    # Apply mask dictionary to mask array
    for key, val in mask_dict.items():
        key_mask = np.zeros(wave.shape, dtype=bool)
        # -- range entries
        if key == "range":
            for entry in val:
                lo, hi = entry
                key_mask |= (wave >= lo) & (wave <= hi)
        # -- line entries
        elif key == "line":
            for entry in val:
                line, width = entry
                key_mask |= (np.abs(wave - line) <= width)
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
    Plots shaded regions (axvspan) for contiguous wavelength regions 
    where the mask is default.
    """
    wave = np.asarray(wave)
    mask = np.asarray(mask)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Optionally invert mask to highlight where mask is not default
    if default is True:
        mask = ~mask

    # Process mask
    # -- identify mask/unmasked transitions
    diff = np.diff(mask.astype(int))
    start_indices = np.where(diff==1)[0] + 1
    end_indices = np.where(diff==-1)[0] + 1
    # -- handle cases where mask starts/ends inside a masked region
    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if mask[-1]:
        end_indices = np.append(end_indices, len(mask) - 1)

    # Plot spectra
    ax.step(wave, flux, color="black", where="mid")

    # Plot mask
    for start, end in zip(start_indices, end_indices):
        ax.axvspan(wave[start], wave[end], color=color, alpha=alpha)

    # Prettify
    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(1, 6))
    if zred is not None:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        top_ticks_rest = np.arange(0.2, 1.4, 0.2)
        top_ticks_obs = top_ticks_rest * (1 + zred)
        ax_top.set_xticks(top_ticks_obs)
        ax_top.set_xticklabels([f"{t:.1f}" for t in top_ticks_rest])
    ax.set_xlabel(r'Observed Wavelength [$\mu$m]', size=18)
    ax.set_ylabel(r'$f_\lambda~[~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$', size=18)

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
        [convert_wave_um_to_A(prism_wave_um.min()), convert_wave_um_to_A(1)],
        [convert_wave_um_to_A(5), convert_wave_um_to_A(prism_wave_um.max())],
        ],
    'line' : [
        [5900, 400]  # sodium doublet
        ],
}
prism_mask = create_1d_wave_mask(wave=prism_wave_um, wave_units="um", mask_dict=prism_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, z=zred, return_ints=True)

print("grat1")
# Create grat1 mask
grat1_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(grat1_wave_um.min()), convert_wave_um_to_A(1)],
        ]
}
grat1_mask = create_1d_wave_mask(wave=grat1_wave_um, wave_units="um", mask_dict=grat1_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, z=zred, return_ints=True)

# Create grat2 mask
grat2_mask_dict_A = {

}

print("grat3")
# Create grat3 mask
grat3_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(grat3_wave_um.min()), convert_wave_um_to_A(2.75)],
        [convert_wave_um_to_A(5.4), convert_wave_um_to_A(grat3_wave_um.max())],
        [convert_wave_um_to_A(prism_wave_um.max()), convert_wave_um_to_A(grat3_wave_um.max())],  # cut off at prism limit
        ]
}
grat3_mask = create_1d_wave_mask(wave=grat3_wave_um, wave_units="um", mask_dict=grat3_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, z=zred, return_ints=True)

# Plot masks
# -- prism
fig = plot_mask(prism_wave_um, prism_flux, prism_mask, zred=zred)
# -- grat1
fig = plot_mask(grat1_wave_um, grat1_flux, grat1_mask, zred=zred)
# -- grat2
# fig = plot_masked_regions(grat2_wave_um, grat2_flux, grat2_mask)
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
# -- grat3
out_file = grat3_kwargs["data_name"].replace(".fits", "_mask.fits")
save_wave_mask(grat3_mask, out_dir, out_file, mask_ext="MASK")