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

    print(mask)

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

def plot_masked_regions(ax, wave, mask, color='gray', alpha=0.3, default=True):
    """
    Plots shaded regions (axvspan) for contiguous wavelength regions 
    where the mask is default.
    """
    wave = np.asarray(wave)
    mask = np.asarray(mask)

    # Optionally invert mask to highlight where mask is not default
    # for Prospector, masked regions are False
    if default is not False:
        mask = ~mask

    # Identify transitions between masked/unmasked regions
    diff = np.diff(mask.astype(int))
    start_indices = np.where(diff == 1)[0] + 1
    end_indices   = np.where(diff == -1)[0] + 1

    # Handle cases where mask starts/ends inside a masked region
    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if mask[-1]:
        end_indices = np.append(end_indices, len(mask) - 1)

    # Plot spans
    for start, end in zip(start_indices, end_indices):
        ax.axvspan(wave[start], wave[end], color=color, alpha=alpha)

def return_file_from_param_dict(param_dict, obs_type):

    # Prism
    if obs_type == 'prism':
        # -- extract variables
        name = param_dict["name"]
        version = param_dict["version"]
        nod = param_dict["nod"]
        # -- version of spectra/reduction
        if version is not None:
            version_str = f"_{version}"
        else:
            version_str = ""
        # -- nod
        if param_dict["nod"] is not None:
            nod_str = f"_{nod}"
        else:
            nod_str = ""
        # -- spectrum dimensions
        dim_str = "_1D"
        # -- combine into file
        extra_str = version_str + nod_str + dim_str
        out_file = f"{name}_prism_clear{version_str}{nod_str}{dim_str}.fits"
    # Grating
    elif obs_type == 'grating':
        # -- extract variables
        name = filter = param_dict["name"]
        grating = param_dict["grating"]
        filter = param_dict["filter"]
        version = param_dict["version"]
        nod = param_dict["nod"]
        # -- version of spectra/reduction
        if version is not None:
            version_str = f"_{version}"
        else:
            version_str = ""
        # -- nod
        if param_dict["nod"] is not None:
            nod_str = f"_{nod}"
        else:
            nod_str = ""
        # -- spectrum dimensions
        dim_str = "_1D"
        # -- combine into file
        extra_str = version_str + nod_str + dim_str
        out_file = f"{name}_{grating}_{filter}{version_str}{nod_str}{dim_str}.fits"
    else:
        raise ValueError("Observation type must be 'prism' or 'grating'")

    return out_file

# Load spectra
obs_kwargs = {

    "prism_params" : {
        "prism_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "name" : "007329",
        "version" : "v3.1",
        "nod" : "extr5",
        "data_ext" : "DATA",
        "mask_ext" : None,
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "return_none" : False,
    },

    "grat1_params" : {
        "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "name" : "007329",
        "grating" : "g140m",
        "filter" : "f100lp",
        "version" : None,
        "nod" : None,
        "data_ext" : "DATA",
        "mask_ext" : "VALID",
        "in_wave_units" : "um",
        "out_wave_units" : "um",
        "in_flux_units" : "ujy",
        "out_flux_units" : "cgs",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "return_none" : False,
    },

    "grat2_params" : {
        "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "name" : "007329",
        "grating" : "g235m",
        "filter" : "f170lp",
        "version" : None,
        "nod" : None,
        "data_ext" : "DATA",
        "mask_ext" : "VALID",
        "in_wave_units" : "um",
        "out_wave_units" : "um",
        "in_flux_units" : "ujy",
        "out_flux_units" : "cgs",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "return_none" : False,
    },

    "grat3_params" : {
        "grating_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
        "name" : "007329",
        "grating" : "g395m",
        "filter" : "f290lp",
        "version" : None,
        "nod" : None,
        "data_ext" : "DATA",
        "mask_ext" : "VALID",
        "in_wave_units" : "um",
        "out_wave_units" : "um",
        "in_flux_units" : "ujy",
        "out_flux_units" : "cgs",
        "rescale_factor" : 1.86422,
        "snr_limit" : 20,
        "return_none" : False,
    },
}

# Redshift information
zred = 3.2

# Load data
prism_params = obs_kwargs['prism_params']
grat1_params = obs_kwargs['grat1_params']
grat2_params = obs_kwargs['grat2_params']
grat3_params = obs_kwargs['grat3_params']

prism_wave, prism_flux, prism_err,_ = load_prism_data(**prism_params)
grat1_wave, grat1_flux, grat1_err,_ = load_grating_data(**grat1_params)
grat2_wave, grat2_flux, grat2_err,_ = load_grating_data(**grat2_params)
grat3_wave, grat3_flux, grat3_err,_ = load_grating_data(**grat3_params)

# Create prism mask
prism_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(prism_wave.min()), convert_wave_um_to_A(1)],
        [convert_wave_um_to_A(5), convert_wave_um_to_A(prism_wave.max())],
        ],
    'line' : [
        [5900, 150]  # sodium doublet
        ],
}
prism_mask = create_1d_wave_mask(wave=prism_wave, wave_units="um", mask_dict=prism_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, z=zred, return_ints=True)

print("grat1")
# Create grat1 mask
grat1_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(grat1_wave.min()), convert_wave_um_to_A(1)],
        ]
}
grat1_mask = create_1d_wave_mask(wave=grat1_wave, wave_units="um", mask_dict=grat1_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, z=zred, return_ints=True)

# Create grat2 mask
grat2_mask_dict_A = {

}

print("grat3")
# Create grat3 mask
grat3_mask_dict_A = {
    'range' : [
        [convert_wave_um_to_A(grat3_wave.min()), convert_wave_um_to_A(2.75)],
        [convert_wave_um_to_A(5.4), convert_wave_um_to_A(grat3_wave.max())],
        [convert_wave_um_to_A(prism_wave.max()), convert_wave_um_to_A(grat3_wave.max())],  # cut off at prism limit
        ]
}
grat3_mask = create_1d_wave_mask(wave=grat3_wave, wave_units="um", mask_dict=grat3_mask_dict_A, mask_units="A",
                              default=True, redshift_lines=True, z=zred, return_ints=True)

# Prism
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# -- plot spectra and mask
ax.step(prism_wave, prism_flux, color="black", where="mid")
plot_masked_regions(ax, prism_wave, prism_mask, default=True)
# -- prettify
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

# Grat1
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# -- plot spectra and mask
ax.step(grat1_wave, grat1_flux, color="black", where="mid")
plot_masked_regions(ax, grat1_wave, grat1_mask, default=True)
# -- prettify
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

# Grat2
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# -- plot spectra and mask
ax.step(grat2_wave, grat2_flux, color="black", where="mid")
# plot_masked_regions(ax, grat2_wave, grat2_mask, mask_default=True)
# -- prettify
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

# Grat3
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# -- plot spectra and mask
ax.step(grat3_wave, grat3_flux, color="black", where="mid")
plot_masked_regions(ax, grat3_wave, grat3_mask, default=True)
# -- prettify
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

plt.show()

# Save masks
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/wave_masks"
# -- prism
prism_file = return_file_from_param_dict(param_dict=prism_params, obs_type="prism")
out_file = prism_file.strip(".fits") + "_mask" + ".fits"  # modify output file name
save_wave_mask(prism_mask, out_dir, out_file, mask_ext="MASK")
# -- grat1
prism_file = return_file_from_param_dict(param_dict=grat1_params, obs_type="grat1")
out_file = prism_file.strip(".fits") + "_mask" + ".fits"  # modify output file name
save_wave_mask(prism_mask, out_dir, out_file, mask_ext="MASK")