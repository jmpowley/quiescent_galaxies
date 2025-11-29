import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Import loading function
sys.path.append("/Users/Jonah/PhD/Research/quiescent_galaxies/code/scripts/zf-uds-7329")
from loading import load_grating_data
from conversions import convert_wave_A_to_um

obs_kwargs = {

    "g140m_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g140m_f100lp",
        "data_name" : "000007_g140m_f100lp_v5.1_1D.fits",
        # "data_name" : "000003_g140m_f100lp_v5.1_1D.fits",
        # "data_name" : "000140_g140m_f100lp_v5.1_1D.fits",
        # "data_name" : "000319_g140m_f100lp_v5.1_1D.fits",
        # "data_name" : "001028_g140m_f100lp_v5.1_1D.fits",
        # "data_name" : "002888_g140m_f100lp_v5.1_1D.fits",
        # "data_name" : "000024_g140m_f100lp_v5.1_1D.fits",
        # "data_name" : "000017_g140m_f100lp_v5.1_1D.fits",
        "data_ext" : "DATA",
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g140m_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-20,
        "rescale_factor" : None,
        "snr_limit" : None,
        "prefix" : "g140m",
    },

    "g235m_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g235m_f170lp",
        "data_name" : "000007_g235m_f170lp_v5.1_1D.fits",
        # "data_name" : "000003_g235m_f170lp_v5.1_1D.fits",
        # "data_name" : "000140_g235m_f170lp_v5.1_1D.fits",
        # "data_name" : "000319_g235m_f170lp_v5.1_1D.fits",
        # "data_name" : "001028_g235m_f170lp_v5.1_1D.fits",
        # "data_name" : "002888_g235m_f170lp_v5.1_1D.fits",
        # "data_name" : "000024_g235m_f170lp_v5.1_1D.fits",
        # "data_name" : "000017_g235m_f170lp_v5.1_1D.fits",
        "data_ext" : "DATA",
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g235m_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-20,
        "rescale_factor" : None,
        "snr_limit" : None,
        "prefix" : "g295m",
    },

    "g395m_kwargs" : {
        "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g395m_f290lp",
        "data_name" : "000007_g395m_f290lp_v5.1_1D.fits",
        # "data_name" : "000003_g395m_f290lp_v5.1_1D.fits",
        # "data_name" : "000140_g395m_f290lp_v5.1_1D.fits",
        # "data_name" : "000319_g395m_f290lp_v5.1_1D.fits",
        # "data_name" : "001028_g395m_f290lp_v5.1_1D.fits",
        # "data_name" : "002888_g395m_f290lp_v5.1_1D.fits",
        # "data_name" : "000024_g395m_f170lp_v5.1_1D.fits",
        # "data_name" : "000017_g395m_f290lp_v5.1_1D.fits",
        "data_ext" : "DATA",
        "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
        "disp_name" : "jwst_nirspec_g395m_disp.fits",
        "in_wave_units" : "si",
        "out_wave_units" : "um",
        "in_flux_units" : "si",
        "out_flux_units" : "cgs",
        "cgs_factor" : 1e-20,
        "rescale_factor" : None,
        "snr_limit" : None,
        "prefix" : "g395m",
    },
}
g140m_wave, g140m_spec, g140m_err = load_grating_data(**obs_kwargs["g140m_kwargs"])
g235m_wave, g235m_spec, g235m_err = load_grating_data(**obs_kwargs["g235m_kwargs"])
g395m_wave, g395m_spec, g395m_err = load_grating_data(**obs_kwargs["g395m_kwargs"])

# Define lines
# -- UV
lines_A = {
    "lya" : 1215.67,
    # "NVa" : 1238.821,
    "NVna" : 1238.804,
    # "NV" : (1238.821 + 1242.804) / 2,
    "NVnb" : 1242.795,
    "SiIIa" : 1260.422,
    "SiII" : (1260.422 + 1264.730) / 2,
    "SiIIb" : 1264.730,
    "OI" : 1302.168,
    "SiIIc" : 1304.37,
    "CII" : (1334.532 + 1335.708) / 2,
    "SiVa" : 1393.755,
    "SiV" : (1393.755 + 1402.770) / 2,
    "SiVb" : 1402.770,
    # "SV?" : 1501.760,
    "CIVa" : 1548.187,
    "CIV" : (1548.187 + 1550.772) / 2,
    "CIVb" : 1550.772,
    "HeII" : 1640.420,
    "OIII]a" : 1660.809,
    "OIII]b" : 1666.150,
    "CIIIa:" : 1906.683,
    "CIIIb" : 1908.734,
    "OIII" : 2320.951,
    "MgII]" : (2795.528 + 2802.705) / 2,
    "OIIa" : 3726.032,
    "OII" : (3728.815 + 3726.032) / 2.0,
    "OIIb" : 3728.815,
    "NeIIIa" : 3868.760,
    "NeIIIb" : 3967.470,
    "Heps" : 3970.079,
    "CaII3934" : 3934.777,
    "CaII3968" : 3969.591,
    "Dn4000" : 4000.00,
    "FeV" : 4071.240,
    "Hdelta" : 4101.742,
    "FeII" : 4178.862,
    "HeIa" : 4143.761,
    "FeII" : 4178.862,
    "Hgamma" : 4340.471,
    "ArIV?" : 4711.260,
    "Hbeta": 4861.333,
    "OIII4959" : 4958.911,
    "OIII5007" : 5006.843,
    "NI" : 5200.257,
    "SIIIa" : 6312.060,
    "NII6548" : 6548.050,
    "Halpha" : 6562.819,
    "NII6583" : 6583.460,
    "SII6716" : 6716.440,
    "SII6731" : 6730.810,
    "SIIIb" : 9068.600,
    "Pa9" : 9231.50,
    "SIIIc" : 9531.100,
    "Peps" : 9548.60,
    "Pdelta" : 10052.10,
    "Pgamma" : 10941.10,
    "HeIb" : (10027.730 + 10031.160) / 2,
    "HeIc" : 10830.340,
    "Fe?" : 12134.495,
    "Pbeta" : 12821.60,
    "Palpha" : 18756.10,
}
lines_um = {key : convert_wave_A_to_um(val) for key, val in lines_A.items()}

# Define redshift guess
zred = 9.3133
# zred = 2.387
# zred = 9.95
# zred = 6.1
# zred = 6.5775
# zred = 5.766
# zred = 2.982
# zred = 2.51
# zred = 7.8803
# zred = 9.299

# Plot G140M
fig, ax = plt.subplots(figsize=(12, 6))
# -- plot spectra
ax.step(g140m_wave, g140m_spec, where="mid")
ax.fill_between(g140m_wave, g140m_spec-g140m_err, g140m_spec+g140m_err, color="C0", alpha=0.3, step="mid")
# -- add lines
for i, (key, line) in enumerate(lines_um.items()):
    zline = line * (1+zred)
    if (zline > g140m_wave.min()) & (zline < g140m_wave.max()):
        ax.axvline(line * (1+zred), color=f"C{i+1}", ls="--", label=key)
# -- prettify
ax.set_ylabel(r"$f_\lambda~[10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$", size=16)
ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu{\rm m}]$", size=16)
ax.set_xlim(g140m_wave.min(), g140m_wave.max())
ax.set_title("G140M", size=16)
ax.legend()
plt.tight_layout()

# Plot G235M
# -- create figure
fig, ax = plt.subplots(figsize=(12, 6))
# -- plot spectra
ax.step(g235m_wave, g235m_spec, where="mid")
ax.fill_between(g235m_wave, g235m_spec-g235m_err, g235m_spec+g235m_err, color="C0", alpha=0.3, step="mid")
# -- add lines
for i, (key, line) in enumerate(lines_um.items()):
    zline = line * (1+zred)
    if (zline > g235m_wave.min()) & (zline < g235m_wave.max()):
        ax.axvline(line * (1+zred), color=f"C{i+1}", ls="--", label=key)
# -- prettify
ax.set_ylabel(r"$f_\lambda~[10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$", size=16)
ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu{\rm m}]$", size=16)
ax.set_xlim(g235m_wave.min(), g235m_wave.max())
ax.set_title("G235M", size=16)
ax.legend()
plt.tight_layout()

# Plot G395M
# -- create figure
fig, ax = plt.subplots(figsize=(12, 6))
# -- plot spectra
ax.step(g395m_wave, g395m_spec, where="mid")
ax.fill_between(g395m_wave, g395m_spec-g395m_err, g395m_spec+g395m_err, color="C0", alpha=0.3, step="mid")
# -- add lines
for i, (key, line) in enumerate(lines_um.items()):
    zline = line * (1+zred)
    if (zline > g395m_wave.min()) & (zline < g395m_wave.max()):
        ax.axvline(line * (1+zred), color=f"C{i+1}", ls="--", label=key)
# -- prettify
ax.set_ylabel(r"$f_\lambda~[10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$", size=16)
ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu{\rm m}]$", size=16)
ax.set_xlim(g395m_wave.min(), g395m_wave.max())
ax.set_title("G395M", size=16)
ax.legend()
plt.tight_layout()

# Plot all spectra
# -- create figure
fig, ax = plt.subplots(figsize=(12, 6))
# -- plot spectra
ax.step(g140m_wave, g140m_spec, where="mid", label="G140M")
ax.fill_between(g140m_wave, g140m_spec-g140m_err, g140m_spec+g140m_err, color="C0", alpha=0.3, step="mid")
ax.step(g235m_wave, g235m_spec, where="mid", label="G235M")
ax.fill_between(g235m_wave, g235m_spec-g235m_err, g235m_spec+g235m_err, color="C1", alpha=0.3, step="mid")
ax.step(g395m_wave, g395m_spec, where="mid", label="G395M")
ax.fill_between(g395m_wave, g395m_spec-g395m_err, g395m_spec+g395m_err, color="C2", alpha=0.3, step="mid")
# -- prettify
ax.set_ylabel(r"$f_\lambda~[10^{-20}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}]$", size=16)
ax.set_xlabel(r"$\lambda_{\rm obs}~[\mu{\rm m}]$", size=16)
ax.legend()
plt.tight_layout()

plt.show()