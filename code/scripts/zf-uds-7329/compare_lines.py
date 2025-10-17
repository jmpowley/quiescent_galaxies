import os

import numpy as np
import matplotlib.pyplot as plt

from conversions import convert_wave_A_to_um, convert_wave_um_to_m, convert_flux_jy_to_cgs, convert_flux_ujy_to_jy
from loading import load_prism_data, load_grating_data

def plot_spec_for_line(waves, fluxes, errs, masks, lines, complexes, spec_names, widths, colors):

    # Create figure
    nrows = 4
    ncols = 4
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 12))

    # Loop over each line
    for i, (stri, wl) in enumerate(lines.items()):
        r, c = divmod(i, ncols)

        # Loop over each spectrum
        for wave, flux, err, mask, spec_name, color in zip(waves, fluxes, errs, masks, spec_names, colors):

            # Apply NaN mask
            wave[~mask] = np.nan
            flux[~mask] = np.nan
            err[~mask] = np.nan

            # Mask regions outside of lines
            keep_mask = (wave >= (wl - widths[stri])) & (wave <= (wl + widths[stri]))
            idx = np.where(keep_mask)[0]
            # -- expand mask by n indices to include prism
            if len(idx) > 0:
                n_expand = 2
                i_min = max(idx[0] - n_expand, 0)
                i_max = min(idx[-1] + n_expand, len(wave) - 1)
                keep_mask[i_min:i_max+1] = True
            # -- apply keep mask
            wave_plot = wave.copy()
            flux_plot = flux.copy()
            err_plot = err.copy()
            wave_plot[~keep_mask] = np.nan
            flux_plot[~keep_mask] = np.nan
            err_plot[~keep_mask] = np.nan

            # Plot flux and error
            ax[r, c].step(wave_plot, flux_plot, color=color, label=spec_name, where='mid')
            ax[r, c].fill_between(wave_plot, flux_plot-err_plot, flux_plot+err_plot,  step='mid', color=color, alpha=0.25)
            # -- add line info
            if stri in complexes.keys():
                comp = complexes[stri]
                nlines = len(comp)
                for k in range(nlines):
                    ax[r, c].axvline(comp[k], color='gray', ls='--', lw=2)
            else:
                ax[r, c].axvline(wl, color='gray', ls="--", lw=2)
            # -- prettify
            ax[r, c].set_title(f"{stri}", size=16)
            ax[r, c].set_xlim(wl-widths[stri], wl+widths[stri])
            if c == 0:
                ax[r, c].set_ylabel(r'Flux / $1~\times~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
            if r == nrows - 1:
                ax[r, c].set_xlabel(r'Observed Wavelength / $\mu$m')
    
    # Extra prettifying code
    handles = [plt.Line2D([0], [0], color=color, lw=2, label=name) for name, color in zip(spec_names, colors)]
    handles.append(plt.Line2D([0], [0], color='gray', ls='--', lw=2, label=f"line at z = {z}"))
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.05, 1.0), ncol=len(handles), frameon=False, fontsize=14)

    fig_name = "zf-uds-7329_all_spectra_lines_comparison.png"
    plt.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return fig

def plot_spectra(waves, fluxes, errs, masks, lines, complexes, spec_names, colors):

    # Create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8.5))

    # Plot each spectrum
    for wave, flux, err, mask, spec_name, color in zip(waves, fluxes, errs, masks, spec_names, colors):

        # Apply mask
        wave[~mask] = np.nan
        flux[~mask] = np.nan
        err[~mask] = np.nan

        # Plot flux
        ax.step(wave, flux, color=color, label=spec_name, where='mid')
        ax.fill_between(wave, flux-err, flux+err, color=color, alpha=0.25)
        # -- prettify
        ax.set_xlabel(r'Observed Wavelength / $\mu$m')
        ax.set_ylabel(r'Flux / $1~\times~10^{-19}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$')
        ax.set_ylim(0.0, 3.0)
        ax.legend()

    # Add line info
    for (stri, wl) in lines.items():
        if stri in complexes.keys():
            comp = complexes[stri]
            nlines = len(comp)
            for k in range(nlines):
                ax.axvline(comp[k], color='gray', ls='--')
        else:
            ax.axvline(wl, color='gray', ls="--")
        ax.text(wl+0.05, 0.25, stri, color='gray', ha='left')

    plt.tight_layout()

    return fig

# Spectra info
spec_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra"
name = 'zf-uds-7329'
grat_filts = ['g395m-f290lp', 'g235m-f170lp', 'g140m-f100lp']
spec_names = ['prism', *grat_filts]  # combine into one list to loop over

fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/zf-uds-7329"

# Lists to store flux information
spec_waves_um = []
spec_fluxes_ujy = []
spec_fluxes_cgs = []
spec_errs_ujy = []
spec_errs_cgs = []
spec_masks = []

# Loop over each spectrum
for i, spec_name in enumerate(spec_names):

    # Load data
    # -- prism
    if spec_name == 'prism':
        wave_um, flux_ujy, err_ujy, mask = load_prism_data(spec_dir, name, version=3.1, extra_nod='extr5', wave_units='um', flux_units='ujy', return_none=False)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        wave_m = convert_wave_um_to_m(wave_um)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)
    # -- grating
    else:
        grat, filt = spec_name.split('-')
        wave_um, flux_ujy, err_ujy, mask = load_grating_data(spec_dir, name, grating=grat, filter=filt, wave_units='um', flux_units='ujy', return_none=False)
        flux_jy, err_jy = convert_flux_ujy_to_jy(flux_ujy, err_ujy)
        wave_m = convert_wave_um_to_m(wave_um)
        flux_cgs, err_cgs = convert_flux_jy_to_cgs(wave_m, flux_jy, err_jy, cgs_factor=1e-19)

    # Multiply flux by scaling factor
    factor = 1.86422
    flux_jy = flux_jy * factor
    flux_cgs = flux_cgs * factor
    err_jy = err_jy * factor
    err_cgs = err_cgs * factor

    # Append data to list
    spec_waves_um.append(wave_um)
    spec_fluxes_ujy.append(flux_ujy)
    spec_fluxes_cgs.append(flux_cgs)
    spec_errs_ujy.append(err_ujy)
    spec_errs_cgs.append(err_cgs)
    spec_masks.append(mask)

# Lines to compare
lines_A = {  # units in Angstroms
    'MgII' : 2800.000,  # individual lines are 2795.528, 2802.705
    'CaII' : 3950.000,  # individual lines are 3933.663, 3968.469
    'Hdelta' : 4101.742,
    'Hgamma' : 4340.471,
    'Hbeta' : 4861.333,
    'Mgb' : 5200.00,  # individual lines are 2795.528, 2802.705
    'NIII4511?' : 4510.910,
    'FeII4556?' : 4555.893,
    # 'FeII5198?' : 5197.577,
    'FeIII5270?' : 5270.400,
    'FeII8892?' : 8891.910,
    'NaD' : 5892.500,  # individual lines are 5889.950, 5895.924
    'Halpha' : 6562.819,
    'TiO' : 7200.000,  # approx.
    'TiO/ZrO/CN' : 9300.000,  # approx.
    'Unlisted' : 10000.00,  # unlisted line?
    'CN' : 11000.000,  # approx.
}
lines_um = {
    stri : convert_wave_A_to_um(wl) for stri, wl in lines_A.items()
}
# -- complexes (doublets & triplets)
complexes_A = {
    "MgII" : [2795.528, 2802.705],
    'CaII' : [3933.663, 3968.469],
    "NaD" : [5889.950, 5895.924],
}
complexes_um = {
    stri : [convert_wave_A_to_um(complex) for complex in complexes] for stri, complexes in complexes_A.items()
}

# Redshift lines
z = 3.2
zlines_um = {
    stri : wl*(1.+z) for stri, wl in lines_um.items()
    }
zcomplexes_um = {
    stri : [complex * (1.+z) for complex in complexes] for stri, complexes in complexes_um.items()
}

# Widths of lines to plot
widths_A = {
    'MgII' : 100.00,
    'CaII' : 150.00,
    'Hdelta' : 100.00,
    'Hgamma' : 50.00,
    'Hbeta' : 100.00,
    'Mgb' : 100.00,
    'NIII4511?' : 100.00,
    'FeII4556?' : 100.00,
    'FeII5198?' : 100.00,
    'FeIII5270?' : 100.00,
    'FeII8892?' : 100.00,
    'NaD' : 300.00,
    'Halpha' : 100.00,
    'TiO' : 800.00,
    'TiO/ZrO/CN' : 800.00,
    'Unlisted' : 350.00,
    'CN' : 200.00,
}
widths_um = {
    stri : convert_wave_A_to_um(width) for stri, width in widths_A.items()
}

# Plotting kwargs
colors = ['red', 'blue', 'green', 'orange']

# plot_spectra(spec_waves_um, spec_fluxes_cgs, spec_errs_cgs, spec_masks, zlines_um, zcomplexes_um, spec_names, colors)
plot_spec_for_line(spec_waves_um, spec_fluxes_cgs, spec_errs_cgs, spec_masks, zlines_um, zcomplexes_um, spec_names, widths_um, colors)
plt.show()