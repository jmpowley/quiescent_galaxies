import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sedpy.observate import load_filters

from scalpel.conversions import convert_wave_um_to_m, convert_wave_A_to_um, convert_flux_ujy_to_jy, convert_flux_jy_to_ujy, convert_flux_jy_to_cgs, convert_flux_jy_to_maggie, convert_wave_A_to_m, convert_wave_m_to_A
from scalpel.loading import load_cube_data

# def load_voronoi_bins():

#     # Assume `x_good`, `y_good`, and `bin_num` are already defined from the Voronoi binning process.
#     # `x_good` and `y_good` are the coordinates of the pixels, and `bin_num` contains the bin number for each pixel.

#     # Create a dictionary to store the pixels in each bin
#     pixels_in_bins23 = defaultdict(list)

#     # Loop through the pixels and add each one to the appropriate bin
#     for i, bin_label in enumerate(bin_num23):
#         pixel23 = (round(x_gal23/0.05)+round(x_good23[i]/0.05), round(y_gal23/0.05)+round(y_good23[i]/0.05))
#         pixels_in_bins23[bin_label].append(pixel23)

#     # Convert defaultdict to a regular dictionary (optional)
#     pixels_in_bins23 = dict(pixels_in_bins23)

#     # Now, pixels_in_bins contains each bin as a key, with a list of pixel coordinates as values
#     for bin_label, pixels23 in pixels_in_bins23.items():
#         print(f"Voronoi bin {bin_label} contains pixels: {pixels23}")

def load_voronoi_bin_coords_temp():
    """Load voronoi bin co-ordinates directly from list"""

    bin_coords = [
        [(41, 44), (42, 44)],
        [(41, 43), (42, 43)],
        [(39, 42), (39, 43), (40, 43)],
        [(39, 44), (40, 44)],
        [(38, 39), (39, 39), (40, 39), (37, 40), (38, 40), (39, 40), (40, 40), (38, 41), (39, 41)],
        [(41, 40), (40, 41), (41, 41), (42, 41), (40, 42), (41, 42), (42, 42)],
        [(36, 41), (37, 41), (36, 42), (37, 42), (36, 43), (37, 43)],
        [(38, 44)],
        [(36, 44), (37, 44)],
        [(39, 45)],
        [(38, 45)],
        [(40, 45), (41, 45), (42, 45), (40, 46), (41, 46), (41, 47)],
        [(36, 45), (37, 45), (36, 46)],
        [(39, 46), (39, 47), (40, 47), (39, 48), (40, 48), (39, 49)],
        [(37, 46), (38, 46)],
        [(36, 47), (37, 47), (38, 47), (37, 48), (38, 48), (38, 49)],
    ]

    return bin_coords

def create_voronoi_binned_cube(cube, cube_err, bin_coords):

    # Create binned cube
    nbins = len(bin_coords)
    nlam, ny, nx = cube.shape
    binned_cube = np.full((nlam, nbins), fill_value=np.nan)
    binned_cube_err = np.full((nlam, nbins), fill_value=np.nan)

    # Loop over all bins
    for i, pix_coords in enumerate(bin_coords):

        npix = len(pix_coords)
        binned_spec = np.full((nlam, npix), fill_value=np.nan)
        binned_err = np.full((nlam, npix), fill_value=np.nan)

        # Loop over all spaxels in bin
        for j, pix_coord in enumerate(pix_coords):
            x, y = pix_coord
            # -- save spectra
            binned_spec[:, j] = cube[:, y, x]
            binned_err[:, j] = cube_err[:, y, x]

        # Add binned spectra to cube
        binned_cube[:, i] = np.nansum(binned_spec, axis=1)
        binned_cube_err[:, i] = np.nansum(binned_err, axis=1)

    return binned_cube, binned_cube_err

def plot_voronoi_bin_coords(cube, bin_coords):

    nlam, ny, nx = cube.shape
    spaxels = np.zeros((ny, nx))

    # Loop over all bins
    for i, pix_coords in enumerate(bin_coords):

        # Loop over all spaxels in bin
        for pix_coord in pix_coords:
            x, y = pix_coord
            # -- assign each bin an integer
            spaxels[y, x] = i

    plt.imshow(spaxels, origin="lower", cmap="tab20")

def plot_voronoi_binned_spectra(wave, binned_cube, binned_cube_err):

    nlam, nbins = binned_cube.shape

    fig, axes = plt.subplots(nrows=nbins, ncols=1, figsize=(10, 4*nbins))

    for i in range(nbins):
        ax = axes[i]

        # Load spectra
        binned_spec = binned_cube[:, i]
        binned_err = binned_cube_err[:, i]
        # -- scale for plot
        binned_spec /= 1e-20
        binned_err /= 1e-20

        # Plot spectra
        ax.plot(wave, binned_spec)
        ax.fill_between(wave, binned_spec-binned_err, binned_spec+binned_err, color="k", alpha=0.3)

        # Prettify
        ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu m]$", size=16)
        ax.set_ylabel(r"$f_\lambda~[~10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$", size=16)
        ax.text(0.03, 0.97, f"Voronoi bin {i}", transform=ax.transAxes, va="top", ha="left", size=14)
        ax.tick_params(axis='both' , direction='in')

    plt.tight_layout()

    return fig

def plot_voronoi_binned_spectra_with_filters(wave, binned_cube, binned_cube_err, sedpy_filters):

    nlam, nbins = binned_cube.shape

    fig, axes = plt.subplots(nrows=nbins, ncols=1, figsize=(10, 4*nbins))

    for i in range(nbins):
        ax = axes[i]

        # Load spectra
        binned_spec = binned_cube[:, i]
        binned_err = binned_cube_err[:, i]
        # -- scale for plot
        binned_spec /= 1e-20
        binned_err /= 1e-20

        # Plot spectra
        ax.plot(wave, binned_spec, color="k")
        ax.fill_between(wave, binned_spec-binned_err, binned_spec+binned_err, color="k", alpha=0.3)

        ymax = np.nanpercentile(binned_spec, 99.9)

        # Plot filters
        # colors = [f"C{i}" for i in range(0, len(sedpy_filters))]
        cmap = cm.get_cmap("rainbow")
        colors = [cmap(i/len(sedpy_filters)) for i in range(len(sedpy_filters))]

        for (f, color) in zip(sedpy_filters, colors):
            filter_wave = convert_wave_A_to_um(f.wavelength)
            trans_norm = f.transmission / np.max(f.transmission)  # normalise
            trans_scale = trans_norm * 0.2 * ymax  # scale wrt. ymax
            name = f.nick.lstrip("jwst_").upper()  # nice name (e.g., F444W)
            ax.fill_between(filter_wave, 0, trans_scale, color=color, alpha=0.3)
            ax.fill_between(filter_wave, np.nan, np.nan, color=color, label=name)
            ax.plot(filter_wave, trans_scale, color=color)

        # Prettify
        ax.set_xlim(wave.min(), wave.max())
        ax.set_ylim(0, ymax)
        ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu m]$", size=16)
        ax.set_ylabel(r"$f_\lambda~[~10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$", size=16)
        ax.text(0.03, 0.97, f"Voronoi bin {i}", transform=ax.transAxes, va="top", ha="left", size=14)
        ax.tick_params(axis='both' , direction='in')

        if i == 0:
            ax.legend(ncols=len(sedpy_filters)//2, bbox_to_anchor=[0, 1], loc="lower left")

    plt.tight_layout()

    return fig

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
    "psf_dir" : None,
    "psf_name" : None,
    "psf_ext" : None,
}
wave, cube, cube_err = load_cube_data(**cube_kwargs)

# Load filters
filters = ["f090w", "f115w", "f150w", "f182m", "f200w", "f210m", "f277w", "f356w", "f410m", "f430m", "f444w", "f460m", "f480m"]
sedpy_filters = load_filters(["jwst_" + f for f in filters])

# Load binned data
bin_coords = load_voronoi_bin_coords_temp()
binned_cube, binned_cube_err = create_voronoi_binned_cube(cube, cube_err, bin_coords)

# Make plots
fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/gs-9209/obs_inspection"
# -- bin co-ordinates
plot_voronoi_bin_coords(cube, bin_coords)
# -- voronoi-binned spectra
fig_name = "gs-9209_binned_spectra.pdf"
fig = plot_voronoi_binned_spectra(wave, binned_cube, binned_cube_err)
fig.savefig(os.path.join(fig_dir, fig_name))
# -- voronoi-binned spectra
fig_name = "gs-9209_binned_spectra_allfilters.pdf"
fig = plot_voronoi_binned_spectra_with_filters(wave, binned_cube, binned_cube_err, sedpy_filters)
fig.savefig(os.path.join(fig_dir, fig_name))

plt.show()