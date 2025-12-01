import os
import sys
from time import time

import numpy as np

from astropy import cosmology
import astropy.units as u
from sedpy.observate import load_filters
import fsps

from prospect.sources.galaxy_basis import FastStepBasis
from prospect.observation import Photometry, Spectrum, PolyOptCal
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.models import priors
from prospect.likelihood import NoiseModel, NoiseModelCov
from prospect.likelihood.kernels import Uncorrelated
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer

# Import functions
sys.path.append("/Users/Jonah/PhD/Research/quiescent_galaxies/code/scripts/zf-uds-7329")
from loading import load_photometry_data, load_spectrum_data, load_mask_data, load_dispersion_data
from preprocessing import crop_bad_spectral_resolution

# ----------------
# Helper functions
# ----------------
def convert_zred_to_agebins(zred=None, nbins_sfh=None, **extras):
    """ 
    Returns age bins going [0, 10Myr, 30Myr, 100Myr, ...] and nbin-2 equally spaced 
    (in logtime) bins from 100Myr to age_universe
    """

    # TODO: Add cosmology as kwarg
    cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)
    zmax = 20

    # Calculate age of the oldest bin
    t_univ = np.squeeze(cosmo.age(zred).to("yr").value)  # age of universe at zred 
    t_zmax = np.squeeze(cosmo.age(zmax).to("yr").value)  # age of universe at zmax
    t_binmax = t_univ - t_zmax

    # Create set of age bins up from t_zmax to t_obs/t0
    n_comp = np.squeeze(nbins_sfh)
    age_lims1 = [0.0, 7.0, 7.4771]  # 0(1), 10, 30 Myr
    n_age_lims3 = 4
    n_age_lims2 = (n_comp + 1) - len(age_lims1) - n_age_lims3
    age_lims2 = np.linspace(8.0, 9.0, n_age_lims2, endpoint=False).tolist()  # 100 Myr, ... 1 Gyr
    age_lims3 = np.log10(np.linspace(1.*10**9, t_binmax, n_age_lims3, endpoint=True)).tolist()  # equally-spaced logtime bins from 1 Gyr to t_binmax
    age_lims = age_lims1 + age_lims2 + age_lims3
    # -- convert to bins
    age_bins = np.array([age_lims[:-1], age_lims[1:]])
    
    return age_bins.T

def convert_zred_to_agebins_finer(zred=None, nbins_sfh=12, **extras):
    """ 
    Returns age bins going [0, 5Myr, 10Myr, 20Myr, 30Myr, 50Myr, 100Myr, ...] and nbin-2 equally spaced 
    (in logtime) bins from 100Myr to age_universe
    """

    # TODO: Add cosmology as kwarg
    cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)
    zmax = 20

    # Calculate age of the oldest bin
    t_univ = np.squeeze(cosmo.age(zred).to("yr").value)  # age of universe at zred 
    t_zmax = np.squeeze(cosmo.age(zmax).to("yr").value)  # age of universe at zmax
    t_binmax = t_univ - t_zmax

    # Create set of age bins up from t_zmax to t_obs/t0
    n_comp = np.squeeze(nbins_sfh)
    age_lims1 = [0.0, 6.6990, 7.0, 7.3010, 7.4771, 7.6990]  # 0(1), 5, 10, 20, 30, 50 Myr
    n_age_lims3 = 4
    n_age_lims2 = (n_comp + 1) - len(age_lims1) - n_age_lims3
    age_lims2 = np.linspace(8.0, 9.0, n_age_lims2, endpoint=False).tolist()  # 100 Myr, ... 1 Gyr
    age_lims3 = np.log10(np.linspace(1.*10**9, t_binmax, n_age_lims3, endpoint=True)).tolist()  # 3 equally-spaced logtime bins from 1 Gyr to t_binmax
    age_lims = age_lims1 + age_lims2 + age_lims3
    # -- convert to bins
    age_bins = np.array([age_lims[:-1], age_lims[1:]])

    # Check age bins match number of sfh bins
    assert np.shape(age_bins)[0] == nbins_sfh
    
    return age_bins.T

def convert_logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
        """ Computes masses formed in each bin from SFR ratios"""

        # Calculate age bins and SFR ratios
        # -- select age bins
        # agebins = convert_zred_to_agebins(zred=zred, **extras)
        agebins = convert_zred_to_agebins_finer(zred=zred, **extras)
        logsfr_ratios = np.clip(logsfr_ratios, -10, 10)  # numerical issues...
        sfr_ratios = 10**logsfr_ratios

        # Compute relative mass contributions to each bin
        nbins = agebins.shape[0]
        dt = (10**agebins[:, 1] - 10**agebins[:, 0])
        coeffs = np.array([(1./np.prod(sfr_ratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
        mass_norm = (10**logmass) / coeffs.sum()  # normalise by stellar mass
        mass_per_bin = mass_norm * coeffs

        return mass_per_bin

def convert_to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    """Convert second component of dust model to first component based on fitted fraction of second dust model"""

    dust1 = dust1_fraction * dust2
    
    return dust1

def convert_kms_offset_to_redshift(zred, vel_kms):
    """Convert offset of velocity in km/s to difference in redshift (plus and minus)"""

    # Calculate approaching/receding factors
    c_kms = 299792.458
    beta = vel_kms / c_kms
    rec_factor = np.sqrt((1 + beta) / (1 - beta))
    app_factor = np.sqrt((1 - beta) / (1 + beta))

    # Apply to redshift
    z_minus = (1 + zred) * app_factor - 1
    z_plus = (1 + zred) * rec_factor - 1

    return z_minus, z_plus

# -----------------
# Build noise model
# -----------------
# def build_noise(noise_kwargs):

#     add_jitter = bool(noise_kwargs.get("add_jitter", False))
#     if add_jitter:
#         jitter = Uncorrelated(parnames=["spec_jitter"])
#         spec_noise = NoiseModelCov(kernels=[jitter], metric_name="unc", weight_by=["unc"])
#         return spec_noise
#     else:
#         return None

# -----------------
# Build noise model
# -----------------
def build_noise(prefix="", add_jitter=False, include_outliers=True, correlated=False, **extras):
    """
    Return a NoiseModel/NoiseModelCov configured for a single observation.

    Parameters
    ----------
    prefix : str
        Arbitrary tag (e.g., "phot", "prism") used for naming per-obs parameters.
    add_jitter : bool
        Include an Uncorrelated jitter kernel tied to e.g. '{prefix}spec_jitter'.
    correlated : bool
        *CURRENTLY UNIMPLEMENETD* If True, return a NoiseModelCov (allows kernels with off-diagonal covariances).

    Returns
    -------
    noise : instance of NoiseModel or NoiseModelCov
    """

    kernels = []

    # Build jitter kernel to reference a model parname
    if add_jitter:
        jitter_par = f"{prefix}_jitter" if prefix else "jitter"
        jitter = Uncorrelated(parnames=[jitter_par])
        kernels.append(jitter)

    # Build outlier NoiseModel to reference a model parname
    if include_outliers:
        frac_name = f"{prefix}_f_outlier" if prefix else "f_outlier"
        nsig_name = f"{prefix}_nsigma_outlier" if prefix else "nsigma_outlier"
        nm = NoiseModel(frac_out_name=frac_name, nsigma_out_name=nsig_name)
        # -- return only outlier model if no jitter
        if len(kernels) == 0:
            return nm
        kernels.append(nm)

    if len(kernels) == 0:
        return None
    
    # Ensure weight_by length matches kernels length
    weight_by = ["unc"] * len(kernels)
    return NoiseModelCov(kernels=kernels, metric_name="unc", weight_by=weight_by)

# ------------------
# Build observations
# ------------------
class PolySpectrum(PolyOptCal, Spectrum):
    pass

def build_obs(obs_kwargs, **extras):
    """Build a set of Prospector observations using the `Prospector.prospect.observation.Observation` class

    Parameters
    ----------
    obs_kwargs : dict
        input arguments needed to extract wavelength, flux and uncertainty information and convert to the correct units

    Returns
    -------
    obs : list
        list of `Prospector.prospect.observation.Observation` classes made up of spectra and photometry
    """

    # Load spectral data
    phot_filters, phot_flux, phot_err = load_photometry_data(**obs_kwargs["phot_kwargs"])
    prism_wave, prism_flux, prism_err = load_spectrum_data(**obs_kwargs["prism_kwargs"])
    g140m_wave, g140m_flux, g140m_err = load_spectrum_data(**obs_kwargs["g140m_kwargs"])
    g235m_wave, g235m_flux, g235m_err = load_spectrum_data(**obs_kwargs["g235m_kwargs"])
    g395m_wave, g395m_flux, g395m_err = load_spectrum_data(**obs_kwargs["g395m_kwargs"])

    # Load resolution data
    prism_sigma = load_dispersion_data(**obs_kwargs["prism_kwargs"])
    g140m_sigma = load_dispersion_data(**obs_kwargs["g140m_kwargs"])
    g235m_sigma = load_dispersion_data(**obs_kwargs["g235m_kwargs"])
    g395m_sigma = load_dispersion_data(**obs_kwargs["g395m_kwargs"])

    # Build noise models
    phot_noise = build_noise(**obs_kwargs["phot_kwargs"])
    prism_noise = build_noise(**obs_kwargs["prism_kwargs"])
    g140m_noise = build_noise(**obs_kwargs["g140m_kwargs"])
    g235m_noise = build_noise(**obs_kwargs["g235m_kwargs"])
    g395m_noise = build_noise(**obs_kwargs["g395m_kwargs"])

    # Load masks
    prism_mask = load_mask_data(**obs_kwargs["prism_kwargs"])
    g140m_mask = load_mask_data(**obs_kwargs["g140m_kwargs"])
    g235m_mask = load_mask_data(**obs_kwargs["g235m_kwargs"])

    # Crop wavelength ranges with spectral resolution high than data
    # -- prism
    # prism_wave, prism_flux, prism_err, prism_mask, prism_sigma = crop_bad_spectral_resolution(prism_wave, prism_flux, prism_err,prism_mask, prism_sigma, zred=3.2, wave_rest_lo=2000, wave_rest_hi=7000)
    # -- medium-grating spectra
    # g140m_wave, g140m_flux, g140m_err, g140m_mask, g140m_sigma = crop_bad_spectral_resolution(g140m_wave, g140m_flux, g140m_err, g140m_mask, g140m_sigma, zred=3.2, wave_rest_lo=3000, wave_rest_hi=7000)
    # g235m_wave, g235m_flux, g235m_err, g235m_mask, g235m_sigma = crop_bad_spectral_resolution(g235m_wave, g235m_flux, g235m_err, g235m_mask, g235m_sigma, zred=3.2, wave_rest_lo=2000, wave_rest_hi=9000)

    # Create Photometry and Spectrum classes
    # -- nircam photometry
    phot = Photometry(filters=phot_filters,
                      flux=phot_flux,
                      uncertainty=phot_err,
                      mask=None,
                      noise=phot_noise,
                      name=obs_kwargs["phot_kwargs"].get("prefix"),
                      )
    # -- prism spectrum
    prism_polyspec = PolySpectrum(wavelength=prism_wave,
                                  flux=prism_flux,
                                  uncertainty=prism_err,
                                  mask=prism_mask,
                                  noise=prism_noise,
                                  resolution=prism_sigma,
                                  polynomial_order=10,
                                  name=obs_kwargs["prism_kwargs"].get("prefix"),
                                  )
    # -- medium-grating spectrum
    g140m_polyspec = PolySpectrum(wavelength=g140m_wave,
                                  flux=g140m_flux,
                                  uncertainty=g140m_err,
                                  mask=g140m_mask,
                                  noise=g140m_noise,
                                  resolution=g140m_sigma,
                                  polynomial_order=10,
                                  name=obs_kwargs["g140m_kwargs"].get("prefix"),
                                  )
    g235m_polyspec = PolySpectrum(wavelength=g235m_wave,
                                  flux=g235m_flux,
                                  uncertainty=g235m_err,
                                  mask=g235m_mask,
                                  noise=g235m_noise,
                                  resolution=g235m_sigma,
                                  polynomial_order=10,
                                  name=obs_kwargs["g235m_kwargs"].get("prefix"),
                                  )
    g395m_polyspec = PolySpectrum(wavelength=g395m_wave,
                                  flux=g395m_flux,
                                  uncertainty=g395m_err,
                                  mask=None,
                                  noise=g395m_noise,
                                  resolution=g395m_sigma,
                                  polynomial_order=10,
                                  name=obs_kwargs["g395m_kwargs"].get("prefix"),
                                  )

    # Build obs from spectrum and photometry
    # -- ensures all required keys are present for fitting
    phot.rectify()
    prism_polyspec.rectify()
    g140m_polyspec.rectify()
    g235m_polyspec.rectify()
    g395m_polyspec.rectify()
    # -- complile observations
    obs = [phot, prism_polyspec, g140m_polyspec, g235m_polyspec, g395m_polyspec]

    return obs

# -----------
# Build model
# -----------
def build_model(model_kwargs, obs_kwargs=None, **extras):
    """Build a `Prospector.models.sedmodel.SpecModel` class using a `ProspectorParams` object

    Parameters
    ----------
    models_kwargs : dict
        kwargs for building the model (e.g., turn nebular meission on/off)

    Returns
    -------
    model : `Prospector.models.sedmodel.SpecModel`
        model to be fit to data
    """

    # Load kwargs
    add_nebular = model_kwargs["add_nebular"]
    smooth_spectra = model_kwargs["smooth_spectra"]
    add_nuisance = model_kwargs["add_nuisance"]
    
    # Continuity SFH
    model_params = TemplateLibrary["continuity_sfh"]

    # Add nebular emission
    if add_nebular:
        model_params.update(TemplateLibrary["nebular"])
        # -- gas parameters
        model_params["gas_logu"]["init"] = -2.0
        model_params["gas_logu"]["isfree"] = True
        model_params["gas_logz"]["isfree"] = True
        # -- adjust for widths of emission lines
        model_params["nebemlineinspec"]["init"] = True
        model_params["eline_sigma"] = dict(N=1, isfree=True, init=100.0, units="km/s",
                                           prior=priors.TopHat(mini=30, maxi=550))
        
    # Set zred
    zred = model_kwargs["zred"]
    z_plus, z_minus = convert_kms_offset_to_redshift(zred=zred, vel_kms=500)  # set bounds +/- 500 km/s from redshift estimate
    model_params["zred"]["isfree"] = True
    model_params["zred"]["init"] = zred
    model_params["zred"]["prior"] = priors.TopHat(mini=z_plus, maxi=z_minus)

    # Set IMF
    model_params["imf_type"]["init"] = 2  # Kroupa IMF

    # Set SFH prior
    # -- fix number of SFH bins
    nbins_sfh = 12
    model_params["nbins_sfh"] = dict(N=1, isfree=False, init=nbins_sfh)
    model_params["agebins"]["N"] = nbins_sfh
    model_params["mass"]["N"] = nbins_sfh
    model_params["logsfr_ratios"]["N"] = nbins_sfh - 1
    # -- create rising SFH ratios
    z0 = zred
    agebins = convert_zred_to_agebins(zred=z0, nbins_sfh=nbins_sfh)
    cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)
    t_lookback_z0 = cosmo.lookback_time(z0).value * 1e9  # lookback time to z0
    t_lookback_bins = t_lookback_z0 + np.mean(10**agebins, axis=1)  # lookback time for agebins
    zbins = cosmology.z_at_value(cosmo.lookback_time, t_lookback_bins*u.yr)  # redshift for agebins
    # -- calculate SFR ratios using rising prior
    alpha = 0.8
    mu = 5/2
    sfr_z = np.exp(-alpha*(zbins - z0)) * (1 + zbins)**(mu)  # base sfr in each bin
    base_logsfr_ratios = np.log10(sfr_z[0:-1]/sfr_z[1::])  # starting logsfr ratios
    # -- set logSFR bin ratios
    # model_params["logsfr_ratios"]["init"] = np.full(nbins_sfh - 1, baseline_sfr_ratios)  # rising SFH bin ratios
    model_params["logsfr_ratios"]["init"] = base_logsfr_ratios  # rising SFH
    model_params["logsfr_ratios"]["prior"] = priors.StudentT(mean=np.full(nbins_sfh-1, 0.0),  
                                                             scale=np.full(nbins_sfh-1, 0.3),
                                                             df=np.full(nbins_sfh-1, 2))  # use Student's-t distribution parameters from Leja et al. (2019)
    # -- scale agebins for redshift such that t_max = t_univ
    model_params["agebins"]["depends_on"] = convert_zred_to_agebins

    # Set total mass formed prior
    model_params["logmass"]["isfree"] = True
    model_params["logmass"]["prior"] = priors.TopHat(mini=7, maxi=12)

    # Set mass formed in each bin
    model_params["mass"]["isfree"] = False
    model_params["mass"]["depends_on"] = convert_logmass_to_masses

    # Set metallicity prior
    model_params["logzsol"]["init"] = np.log10(1.)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2., maxi=0.19)
    model_params["logzsol"]["isfree"] = True

    # Complexify dust attenuation
    # -- switch to Kriek and Conroy (2013) dust attenuation
    model_params["dust_type"]["init"] = 4
    # -- slope of the (diffuse) attenuation curve, expressed as the index of the power-law that modifies the base Kriek & Conroy/Calzetti shape.
    # -- a value of zero is basically Calzetti with a 2175A bump
    model_params["dust_index"] = dict(N=1, isfree=True, init=0.0, 
                                      prior=priors.TopHat(mini=-1.0, maxi=0.2))
    # -- set attenuation of old stellar light (not birth cloud component)
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)
    model_params["dust2"]["isfree"] = True
    # -- set attenuation due to birth clouds (fitted as a fraction of diffuse component)
    model_params["dust1"] = dict(N=1, isfree=False, init=0, prior=None, depends_on=convert_to_dust1)
    model_params["dust1_fraction"] = dict(N=1, isfree=True, init=1.0, 
                                          prior=priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3))

    # Add per-observation noise/outlier parameters
    for key, obs in obs_kwargs.items():
        if obs.get('fit_obs'):
            p = obs.get('prefix')
            model_params[f"{p}_jitter"] = dict(
                N=1, isfree=True, init=1.0,
                prior=priors.TopHat(mini=0.5, maxi=5.0)
            )
            model_params[f"{p}_f_outlier"] = dict(
                N=1, isfree=True, init=1e-3,
                prior=priors.TopHat(mini=1e-5, maxi=1e-2)
            )
            model_params[f"{p}_nsigma_outlier"] = dict(
                N=1, isfree=False, init=50.0
            )
    
    # Add spectral smoothing
    if smooth_spectra:
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["sigma_smooth"] = dict(N=1, isfree=True, init=300.0, units='km/s',
                                            prior= priors.TopHat(mini=10, maxi=1000))  # follow De Graff+25 and set wide prior
        model_params['smoothtype']['init'] = 'vel'
        model_params['fftsmooth'] = {'N':1, 'isfree': False, 'init': True}

    # Add nuiscance parameter to test sampling
    if add_nuisance:
        model_params["bob"] = dict(N=1, isfree=True, init=0.5,
                                prior=priors.TopHat(mini=0, maxi=1))
    
    # Build model
    model = SpecModel(model_params)

    return model

# ----------------------------------
# Build stellar population synthesis
# ----------------------------------
def build_sps(zcontinuous=1, **extras):
    """Build an SPS object
    :param zcontinuous: (default: 1)
        python-fsps parameter controlling how metallicity interpolation of the
        SSPs is acheived.  A value of `1` is recommended.
        * 0: use discrete indices (controlled by parameter "zmet")
        * 1: linearly interpolate in log Z/Z_sun to the target metallicity
             (the parameter "logzsol".)
        * 2: convolve with a metallicity distribution function at each age.
             The MDF is controlled by the parameter "pmetals"
    """
    
    sps = FastStepBasis(zcontinuous=zcontinuous, compute_vega_mags=False)
    
    return sps

# ---------
# Build all
# ---------
# def build_all(obs_kwargs, model_kwargs, **extras):

#     # noise = build_noise(noise_kwargs)
#     obs = build_obs(obs_kwargs)
#     model = build_model(model_kwargs)
#     sps = build_sps()

#     return obs, model, sps

def build_all(obs_kwargs, model_kwargs, **extras):
    """Build all models"""

    obs = build_obs(obs_kwargs)
    model = build_model(model_kwargs, obs_kwargs=obs_kwargs)
    sps = build_sps()
    return obs, model, sps

# -------------
# Main function
# -------------
def main():

    obs_kwargs = {

        # NOTE: No photometry yet
        "phot_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/photometry",
            "data_name" : "000007_nircam_photometry.fits",
            "data_ext" : "DATA",
            "in_flux_units" : "magnitude",
            "out_flux_units" : "maggie",
            "snr_limit" : 20.0,
            "prefix" : "phot",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
        },

        # NOTE: No prism spectra yet
        "prism_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/prism",
            "data_name" : "007329_prism_clear_v3.1_extr5_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" :  "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/spurs/wave_masks",
            "mask_name" : "007329_prism_clear_v3.1_extr5_1D_mask.fits",
            "mask_ext" : "MASK",
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "uds7329_nirspec_prism_disp.fits",
            "in_wave_units" : "si",
            "out_wave_units" : "A",
            "in_flux_units" : "si",
            "out_flux_units" : "maggie",
            "rescale_factor" : None,
            "snr_limit" : 20.0,
            "prefix" : "prism",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : False,
        },

        "g140m_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g140m_f100lp",
            "data_name" : "000007_g140m_f100lp_v5.1_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" : None,
            "mask_name" : None,
            "mask_ext" : None,
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "jwst_nirspec_g140m_disp.fits",
            "in_wave_units" : "m",
            "out_wave_units" : "A",
            "in_flux_units" : "si",
            "out_flux_units" : "maggie",
            "rescale_factor" : None,
            "snr_limit" : 20.0,
            "prefix" : "g140m",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
        },

        "g235m_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g235m_f170lp",
            "data_name" : "000007_g235m_f170lp_v5.1_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" : None,
            "mask_name" : None,
            "mask_ext" : None,
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "jwst_nirspec_g235m_disp.fits",
            "in_wave_units" : "m",
            "out_wave_units" : "A",
            "in_flux_units" : "si",
            "out_flux_units" : "maggie",
            "rescale_factor" : None,
            "snr_limit" : 20.0,
            "prefix" : "g235m",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
        },

        "g395m_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/spurs/spectra/g395m_f290lp",
            "data_name" : "000007_g395m_f290lp_v5.1_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" : None,
            "mask_name" : None,
            "mask_ext" : None,
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "jwst_nirspec_g395m_disp.fits",
            "in_wave_units" : "si",
            "out_wave_units" : "A",
            "in_flux_units" : "si",
            "out_flux_units" : "maggie",
            "rescale_factor" : None,
            "snr_limit" : 20.0,
            "prefix" : "g395m",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
        },
    }

    model_kwargs = {
        "zred" : 9.3133,
        "add_nebular" : True,
        "smooth_spectra" : True,
        "add_nuisance" : False,
        }
    
    # Store all dicts in run_params
    run_params = {}
    run_params["obs_kwargs"] = obs_kwargs
    run_params["model_kwargs"] = model_kwargs

    # Load all
    obs, model, sps = build_all(**run_params)

    # Load FSPS libraries
    sp = fsps.StellarPopulation()
    isoc_lib, spec_lib, dust_lib = sp.libraries
    fsps_libraries = {
        "isoc_lib" : isoc_lib.decode("utf-8"),
        "spec_lib" : spec_lib.decode("utf-8"),
        "dust_lib" : dust_lib.decode("utf-8"),
        }
    print("FSPS libraries used:\n", fsps_libraries)

    # Add extra run kwargs
    # TODO: Change to add some of these from the command line
    # -- general kwargs
    run_params["param_file"] = __file__
    # -- select method
    run_params["emcee"] = False
    run_params["optimize"] = False
    run_params["dynesty"] = False
    run_params["nested_sampler"] = "nautilus"
    # -- nautilus kwargs
    run_params["verbose"] = True
    run_params["n_live"] = 1000  # TODO: change to take as CL argument
    run_params["discard_exploration"] = True
    # -- FSPS libraries
    run_params["fsps_libraries"] = fsps_libraries

    # Map obs list to explicit names
    expected_names = obs_kwargs.keys()
    obs_map = dict(zip(expected_names, obs))

    # Compose new_obs list: include only if 'fit_obs' is True
    new_obs = []
    for key in obs_kwargs.keys():
        if obs_kwargs[key].get("fit_obs", True):
            new_obs.append(obs_map[key])
    print("Observations:\n", obs)
    print("Observations to fit:\n", new_obs)

    # Build descriptive output strings
    # -- file name str
    file_str = __file__.rstrip(".py")
    file_base = os.path.basename(file_str)
    # -- FSPS libraries str
    fsps_str = isoc_lib.decode("utf-8") + spec_lib.decode("utf-8").replace("_", "")
    # -- sampler str
    samp_str = run_params["nested_sampler"]
    # -- obs_str e.g. 'phot_prism_g235m'
    obs_str_list = []
    for key in obs_kwargs.keys():
        if obs_kwargs[key].get("fit_obs", True):
            obs_str_list.append(obs_kwargs[key].get('prefix'))
        obs_str = "_".join(obs_str_list) or "none"
    # -- other key info
    neb_str = "T" if model_kwargs["add_nebular"] else "F"
    smooth_str = "T" if model_kwargs["smooth_spectra"] else "F"
    nuis_str = "T" if model_kwargs["add_nuisance"] else "F"

    # Prepare output
    out_name = f"{file_base}_{fsps_str}_{samp_str}_{obs_str}_neb{neb_str}_smooth{smooth_str}_nuis{nuis_str}.h5"
    out_dir = os.path.join("/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/spurs/prospector_outputs", 
                           f"{file_base}_{fsps_str}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    print("Output path:\n", out_path)

    # Fit model
    start = time()
    output = fit_model(new_obs, model, sps, lnprobfn=lnprobfn, **run_params)
    end = time()
    print(output)
    print(f"Model fit in {end - start:.2f} seconds")
    
    # Save results
    writer.write_hdf5(out_path, run_params, model, new_obs,
                        sampling_result=output["sampling"],
                        sps=sps,
                    )
    print(f"Output saved to {out_path}")

# --------------
# Run prospector
# --------------
if __name__ == "__main__":
    main()