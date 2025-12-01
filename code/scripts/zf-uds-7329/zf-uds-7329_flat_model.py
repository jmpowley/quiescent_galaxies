import os
from time import time

import numpy as np

from astropy import cosmology
from sedpy.observate import load_filters
import fsps

from prospect.sources.galaxy_basis import CSPSpecBasis, FastStepBasis
from prospect.observation import Photometry, Spectrum, PolyOptCal
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.models import priors
from prospect.likelihood import NoiseModel, NoiseModelCov
from prospect.likelihood.kernels import Uncorrelated
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer

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

        tuniv = np.squeeze(cosmo.age(zred).to("yr").value)
        ncomp = np.squeeze(nbins_sfh)
        tbinmax = np.squeeze(cosmo.age(zmax).to("yr").value)
        tbinmax = tuniv - tbinmax
        #agelims = [0.0, 7.4772] + np.linspace(8.0, np.log10(tbinmax), ncomp-1).tolist()
        logtmax = np.log10(2*10**9)
        agelims = [0.0, 7.0, 7.4771] \
                    + np.linspace(8.0, 9.0, ncomp-6, endpoint=False).tolist() \
                    + np.log10(np.linspace(1.*10**9, tbinmax, 4, endpoint=True)).tolist()
        agebins = np.array([agelims[:-1], agelims[1:]])
        
        return agebins.T

def convert_logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
        """ Computes masses formed in each bin from SFR ratios"""

        # Calculate age bins and SFR ratios
        agebins = convert_zred_to_agebins(zred=zred, **extras)
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
    grat1_wave, grat1_flux, grat1_err = load_spectrum_data(**obs_kwargs["grat1_kwargs"])
    grat2_wave, grat2_flux, grat2_err = load_spectrum_data(**obs_kwargs["grat2_kwargs"])
    grat3_wave, grat3_flux, grat3_err = load_spectrum_data(**obs_kwargs["grat3_kwargs"])

    # Load resolution data
    prism_sigma = load_dispersion_data(**obs_kwargs["prism_kwargs"])
    grat1_sigma = load_dispersion_data(**obs_kwargs["grat1_kwargs"])
    grat2_sigma = load_dispersion_data(**obs_kwargs["grat2_kwargs"])
    grat3_sigma = load_dispersion_data(**obs_kwargs["grat3_kwargs"])

    # Build noise models
    phot_noise = build_noise(**obs_kwargs["phot_kwargs"])
    prism_noise = build_noise(**obs_kwargs["prism_kwargs"])
    grat1_noise = build_noise(**obs_kwargs["grat1_kwargs"])
    grat2_noise = build_noise(**obs_kwargs["grat2_kwargs"])
    grat3_noise = build_noise(**obs_kwargs["grat3_kwargs"])

    # Load masks
    prism_mask = load_mask_data(**obs_kwargs["prism_kwargs"])
    grat1_mask = load_mask_data(**obs_kwargs["grat1_kwargs"])
    grat2_mask = load_mask_data(**obs_kwargs["grat2_kwargs"])

    # Crop wavelength ranges with spectral resolution high than data
    # -- prism
    prism_wave, prism_flux, prism_err, prism_mask, prism_sigma = crop_bad_spectral_resolution(prism_wave, prism_flux, prism_err,prism_mask, prism_sigma, zred=3.2, wave_rest_lo=2000, wave_rest_hi=7000)
    # -- medium-grating spectra
    grat1_wave, grat1_flux, grat1_err, grat1_mask, grat1_sigma = crop_bad_spectral_resolution(grat1_wave, grat1_flux, grat1_err, grat1_mask, grat1_sigma, zred=3.2, wave_rest_lo=3000, wave_rest_hi=7000)
    grat2_wave, grat2_flux, grat2_err, grat2_mask, grat2_sigma = crop_bad_spectral_resolution(grat2_wave, grat2_flux, grat2_err, grat2_mask, grat2_sigma, zred=3.2, wave_rest_lo=2000, wave_rest_hi=9000)

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
    grat1_polyspec = PolySpectrum(wavelength=grat1_wave,
                                  flux=grat1_flux,
                                  uncertainty=grat1_err,
                                  mask=grat1_mask,
                                  noise=grat1_noise,
                                  resolution=grat1_sigma,
                                  polynomial_order=10,
                                  name=obs_kwargs["grat1_kwargs"].get("prefix"),
                                  )
    grat2_polyspec = PolySpectrum(wavelength=grat2_wave,
                                  flux=grat2_flux,
                                  uncertainty=grat2_err,
                                  mask=grat2_mask,
                                  noise=grat2_noise,
                                  resolution=grat2_sigma,
                                  polynomial_order=10,
                                  name=obs_kwargs["grat2_kwargs"].get("prefix"),
                                  )
    grat3_polyspec = PolySpectrum(wavelength=grat3_wave,
                                  flux=grat3_flux,
                                  uncertainty=grat3_err,
                                  mask=None,
                                  noise=grat3_noise,
                                  resolution=grat3_sigma,
                                  polynomial_order=10,
                                  name=obs_kwargs["grat3_kwargs"].get("prefix"),
                                  )

    # Build obs from spectrum and photometry
    # -- ensures all required keys are present for fitting
    phot.rectify()
    prism_polyspec.rectify()
    grat1_polyspec.rectify()
    grat2_polyspec.rectify()
    grat3_polyspec.rectify()
    # -- complile observations
    obs = [phot, prism_polyspec, grat1_polyspec, grat2_polyspec, grat3_polyspec]

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
    model_params["zred"]["isfree"] = True
    model_params["zred"]["init"] = model_kwargs["zred"]
    # model_params["zred"]["prior"] = priors.TopHat(mini=3.0, maxi=3.4)
    model_params["zred"]["prior"] = priors.TopHat(mini=3.15, maxi=3.25)

    # Set IMF
    model_params["imf_type"]["init"] = 2  # Kroupa IMF

    # Set SFH prior
    # -- fix number of SFH bins
    nbins_sfh = 9
    model_params["nbins_sfh"] = dict(N=1, isfree=False, init=nbins_sfh)
    model_params["agebins"]["N"] = nbins_sfh
    model_params["mass"]["N"] = nbins_sfh
    model_params["logsfr_ratios"]["N"] = nbins_sfh - 1
    # -- set logSFR bin ratios
    model_params["logsfr_ratios"]["init"] = np.full(nbins_sfh - 1, 0.0)  # logSFR = 0 means constant SFH
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

    # Set spectral calibration polynomial coefficient priors
    # model_params.update(TemplateLibrary["optimize_speccal"])
    # model_params["spec_norm"] = {"N": 1, "isfree": True, "init": 1.0, 
    #                              "units": "f_true/f_obs", "prior": priors.Normal(mean=1.0, sigma=0.1)}
    # model_params["polyorder"]["init"] = 7  # order of polynomial that"s fit to spectrum

    # Set noise priors
    # -- pixel outlier models
    # model_params["nsigma_outlier_spec"] = dict(N=1, isfree=False, init=50.)
    # model_params["f_outlier_spec"] = dict(N=1, isfree=True, init=1e-3, 
    #                                       prior=priors.TopHat(mini=1e-5, maxi=0.01))
    # # -- add multiplicative noise inflation term. Inflates noise in all spectroscopic pixels as necessary to get a statistically acceptable fit.
    # model_params["spec_jitter"] = dict(N=1, isfree=True, init=1.0, 
    #                                    prior=priors.TopHat(mini=0.5, maxi=5.0))

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
                                            prior= priors.TopHat(mini=10, maxi=1000))  # follow methodology of De Graff+25
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

        "phot_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry",
            "data_name" : "007329_nircam_photometry.fits",
            "data_ext" : "DATA",
            "in_flux_units" : "magnitude",
            "out_flux_units" : "maggie",
            "snr_limit" : 20.0,
            "prefix" : "phot",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
        },

        "prism_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "data_name" : "007329_prism_clear_v3.1_extr5_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" :  "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/wave_masks",
            "mask_name" : "007329_prism_clear_v3.1_extr5_1D_mask.fits",
            "mask_ext" : "MASK",
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "uds7329_nirspec_prism_disp.fits",
            "in_wave_units" : "si",
            "out_wave_units" : "A",
            "in_flux_units" : "si",
            "out_flux_units" : "maggie",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20.0,
            "prefix" : "prism",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
        },

        "grat1_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "data_name" : "007329_g140m_f100lp_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/wave_masks",
            "mask_name" : "007329_g140m_f100lp_1D_mask.fits",
            "mask_ext" : "MASK",
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "jwst_nirspec_g140m_disp.fits",
            "in_wave_units" : "um",
            "out_wave_units" : "A",
            "in_flux_units" : "ujy",
            "out_flux_units" : "maggie",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20.0,
            "prefix" : "g140m",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
        },

        "grat2_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra",
            "data_name" : "007329_g235m_f170lp_1D.fits",
            "data_ext" : "DATA",
            "mask_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/wave_masks",
            "mask_name" : "007329_g235m_f170lp_1D_mask.fits",
            "mask_ext" : "MASK",
            "disp_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/dispersion_curves",
            "disp_name" : "jwst_nirspec_g235m_disp.fits",
            "in_wave_units" : "um",
            "out_wave_units" : "A",
            "in_flux_units" : "ujy",
            "out_flux_units" : "maggie",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20,
            "prefix" : "g235m",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : True,
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
            "out_wave_units" : "A",
            "in_flux_units" : "ujy",
            "out_flux_units" : "maggie",
            "rescale_factor" : 1.86422,
            "snr_limit" : 20.0,
            "prefix" : "g395m",
            "add_jitter" : True,
            "include_outliers" : True,
            "fit_obs" : False,
        },
    }

    model_kwargs = {
        "zred" : 3.19,
        "add_nebular" : False,
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
    out_dir = os.path.join("/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/prospector_outputs", 
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