import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import cosmology
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from sedpy.observate import load_filters
import os
import h5py
from time import time
from prospect.sources.galaxy_basis import CSPSpecBasis, FastStepBasis
from prospect.observation import Photometry, Spectrum, PolyOptCal
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SpecModel
from prospect.models import priors
from prospect.likelihood import NoiseModelCov
from prospect.likelihood.kernels import Uncorrelated
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer

from loading import load_photometry_data, load_prism_data, load_grism_data

# ----------------------------
# Functions for building model
# ----------------------------
def zred_to_agebins(zred=None, nbins_sfh=None, **extras):
        """ Returns age bins going [0, 10Myr, 30Myr, 100Myr, ...] and nbin-2 equally spaced (in logtime) bins from 100Myr to age_universe
        """

        # TODO: Add cosmology as kwarg

        cosmo = cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726)

        tuniv = np.squeeze(cosmo.age(zred).to("yr").value)
        ncomp = np.squeeze(nbins_sfh)
        tbinmax = np.squeeze(cosmo.age(20).to("yr").value)
        tbinmax = tuniv - tbinmax
        #agelims = [0.0, 7.4772] + np.linspace(8.0, np.log10(tbinmax), ncomp-1).tolist()
        logtmax = np.log10(2*10**9)
        agelims = [0.0, 7.0, 7.4772] \
                    + np.linspace(8.0, 9.0, ncomp-6, endpoint=False).tolist() \
                    + np.log10(np.linspace(1.*10**9, tbinmax, 4, endpoint=True)).tolist()
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
        """ Computes masses formed in each bin from SFR ratios
        """
        agebins = zred_to_agebins(zred=zred, **extras)
        logsfr_ratios = np.clip(logsfr_ratios, -10, 10)  # numerical issues...
        nbins = agebins.shape[0]
        sfr_ratios = 10**logsfr_ratios
        dt = (10**agebins[:, 1] - 10**agebins[:, 0])
        coeffs = np.array([(1./np.prod(sfr_ratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
        m1 = (10**logmass) / coeffs.sum()
        return m1 * coeffs

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction*dust2

class PolySpectrum(PolyOptCal, Spectrum):
    pass

# ------------------
# Build observations
# ------------------
def build_obs(obs_params):
    """Build a set of Prospector observations using the `Prospector.prospect.observation.Observation` class

    Parameters
    ----------
    obs_params : dict
        input arguments needed to extract wavelength, flux and uncertainty information and convert to the correct units

    Returns
    -------
    obs : list
        list of `Prospector.prospect.observation.Observation` classes made up of spectra and photometry
    """

    prism_params = obs_params['prism_params']
    phot_params = obs_params['phot_params']
    grat1_params = obs_params['grat1_params']
    grat2_params = obs_params['grat2_params']
    grat3_params = obs_params['grat3_params']

    # Load data
    # TODO: Add masks to spectra and photometry
    # -- photometry data
    phot_filters, phot_flux, phot_err = load_photometry_data(**phot_params)
    # -- prism data
    prism_wave, prism_flux, prism_err = load_prism_data(**prism_params)
    # -- grating data
    grat1_wave, grat1_flux, grat1_err = load_grism_data(**grat1_params)
    grat2_wave, grat2_flux, grat2_err = load_grism_data(**grat2_params)
    grat3_wave, grat3_flux, grat3_err = load_grism_data(**grat3_params)

    # Create Photometry and Spectrum classes
    # -- nircam photometry
    phot = Photometry(filters=phot_filters, flux=phot_flux,
                       uncertainty=phot_err, mask=None)
    # -- prism spectrum
    prism_spec = Spectrum(wavelength=prism_wave, flux=prism_flux, uncertainty=prism_err)
    prism_polyspec = PolySpectrum(wavelength=prism_wave, flux=prism_flux, uncertainty=prism_err, polynomial_order=10)  # optimise polynomial spectral calibration (or set as prior in model)
    # -- medium-grating spectrum
    grat1_spec = Spectrum(wavelength=grat1_wave, flux=grat1_flux, uncertainty=grat1_err)
    grat1_polyspec = PolySpectrum(wavelength=grat1_wave, flux=grat1_flux, uncertainty=grat1_err, polynomial_order=10)
    grat2_spec = Spectrum(wavelength=grat2_wave, flux=grat2_flux, uncertainty=grat2_err)
    grat2_polyspec = PolySpectrum(wavelength=grat2_wave, flux=grat2_flux, uncertainty=grat2_err, polynomial_order=10)
    grat3_spec = Spectrum(wavelength=grat3_wave, flux=grat3_flux, uncertainty=grat3_err)
    grat3_polyspec = PolySpectrum(wavelength=grat3_wave, flux=grat3_flux, uncertainty=grat3_err, polynomial_order=10)

    # Build obs from spectrum and photometry
    # -- ensures all required keys are present for fitting
    phot.rectify()
    prism_spec.rectify()
    grat1_spec.rectify()
    grat2_spec.rectify()
    grat3_spec.rectify()
    # -- complile observations
    obs = [phot, prism_spec, grat1_spec, grat2_spec, grat3_spec]

    return obs

# -----------
# Build model
# -----------
def build_model(model_kwargs):
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
    add_nebular = model_kwargs['add_nebular']
    # cosmo = model_kwargs['cosmology']
    
    # Continuity SFH
    model_params = TemplateLibrary["continuity_sfh"]

    # Add nebular emission
    if add_nebular:
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['init'] = -2.0
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True

        # Adjust for widths of emission lines
        model_params["nebemlineinspec"]["init"] = True
        model_params["eline_sigma"] = dict(N=1, isfree=True, init=100.0, units='km/s', 
                                           prior=priors.TopHat(mini=30, maxi=550))
        
    # Set zred to free
    model_params["zred"]["isfree"] = False
    model_params["zred"]["init"] = model_kwargs['zred']
    model_params["zred"]["prior"] = priors.TopHat(mini=3.0, maxi=3.4)

    # Set IMF
    model_params['imf_type']['init'] = 2  # Kroupa IMF

    # Set SFH prior
    # -- fix number of SFH bins
    nbins_sfh = 9
    model_params["nbins_sfh"] = dict(N=1, isfree=False, init=nbins_sfh)
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh - 1
    # -- set logSFR bin ratios
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh - 1, 0.0)  # logSFR = 0 means constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1, 0.0),
                                                             scale=np.full(nbins_sfh-1, 0.3),
                                                             df=np.full(nbins_sfh-1, 2))  # use Student's t-distribution parameters from Leja et al. (2019)

    # Scale agebins for redshift such that t_max = t_univ
    model_params['agebins']['depends_on'] = zred_to_agebins

    # Set total mass formed prior
    model_params["logmass"]["isfree"] = True
    model_params["logmass"]["prior"] = priors.TopHat(mini=7, maxi=12)

    # Set mass formed in each bin ---
    model_params["mass"]["isfree"] = False
    model_params['mass']['depends_on'] = logmass_to_masses

    # Set metallicity prior
    model_params["logzsol"]["init"] = np.log10(1.)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2., maxi=0.19)
    model_params["logzsol"]["isfree"] = True

    # Complexify dust attenuation
    # -- switch to Kriek and Conroy (2013) dust attenuation
    model_params['dust_type']['init'] = 4
    # -- slope of the (diffuse) attenuation curve, expressed as the index of the power-law that modifies the base Kriek & Conroy/Calzetti shape.
    # -- a value of zero is basically calzetti with a 2175AA bump
    model_params["dust_index"] = {'N': 1, 'isfree': True,
                                  'init': 0.0, 'prior': priors.TopHat(mini=-1.0, maxi=0.2)}
    # -- set attenuation of old stellar light (not birth cloud component)
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.3, sigma=1)
    model_params["dust2"]['isfree'] = True

    # set attenuation due to birth clouds (fitted as a fraction of diffuse component)
    model_params['dust1'] = dict(N=1, isfree=False, init=0,
                                prior=None, depends_on=to_dust1)
    model_params['dust1_fraction'] = dict(N=1, isfree=True, init=1.0,
            prior=priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3))

    # Set spectral calibration polynomial coefficient priors
    model_params.update(TemplateLibrary['optimize_speccal'])
    model_params["spec_norm"] = {'N': 1, 'isfree': True, 'init': 1.0, 
                                 'units': 'f_true/f_obs', 'prior': priors.Normal(mean=1.0, sigma=0.1)}
    model_params["polyorder"]["init"] = 7  # order of polynomial that's fit to spectrum

    # Pixel outlier models
    model_params['nsigma_outlier_spec'] = dict(N=1, isfree=False, init=50.)
    model_params['f_outlier_spec'] = dict(N=1, isfree=True, init=1e-3,
                                          prior=priors.TopHat(mini=1e-5, maxi=0.01))

    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a statistically acceptable fit.
    model_params['spec_jitter'] = dict(N=1, isfree=True, init=1.0, 
                                       prior=priors.TopHat(mini=0.5, maxi=5.0))
    
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

# -----------------
# Build noise model
# -----------------
def build_noise(add_jitter=False, **extras):
    if add_jitter:
        jitter = Uncorrelated(parnames=['spec_jitter'])
        spec_noise = NoiseModelCov(kernels=[jitter], metric_name='unc', weight_by=['unc'])
        return spec_noise
    else:
        return None

# ---------
# Build all
# ---------
def build_all(obs_params, model_kwargs, noise_kwargs):

    obs = build_obs(obs_params)
    model = build_model(model_kwargs)
    sps = build_sps()
    noise = build_noise(noise_kwargs)

    return obs, model, sps, noise

# -------------
# Main function
# -------------
def main():

    fit_obs = {
        'phot' : True,
        'prism' : False,
        'grating1' : False,
        'grating2' : False,
    }

    obs_params = {

         'phot_params' : {
            'phot_dir' : '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/photometry',
            'name' : 'zf-uds-7329',
            'flux_units' : 'maggie',
            'return_none' : not fit_obs['phot'],
        },

        'prism_params' : {
            'prism_dir' : '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra',
            'name' : 'zf-uds-7329',
            'version' : 3.1,
            'extra_nod' : 'extr5',
            'flux_units' : 'maggie',
            'return_none' : not fit_obs['prism'],
        },

        'grat1_params' : {
             'grating_dir' : '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra',
             'name' : 'zf-uds-7329',
             'grating' : 'g140m',
             'filter' : 'f100lp',
             'flux_units' : 'maggie',
             'return_none' : not fit_obs['grating1'],
        },

        'grat2_params' : {
             'grating_dir' : '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra',
             'name' : 'zf-uds-7329',
             'grating' : 'g235m',
             'filter' : 'f170lp',
             'flux_units' : 'maggie',
             'return_none' : not fit_obs['grating2'],
        },

        'grat3_params' : {
             'grating_dir' : '/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/spectra',
             'name' : 'zf-uds-7329',
             'grating' : 'g395m',
             'filter' : 'f290lp',
             'flux_units' : 'maggie',
             'return_none' : not fit_obs['grating3'],
        },
    }

    model_kwargs = {
        'zred' : 3.19,
        'add_nebular' : True,
        # 'cosmology' : cosmology.FlatLambdaCDM(H0=67.4, Om0=0.315, Tcmb0=2.726),
        }

    noise_kwargs = {
        'add_jitter' : True
        }

    # Load all
    obs, model, sps, noise = build_all(obs_params, model_kwargs, noise_kwargs)

    print("obs:", obs)
    print("model", model)
    print("sps:", sps)

    # Run kwargs
    run_params = {}
    # -- select method
    run_params["dynesty"] = False
    run_params["emcee"] = True
    run_params["optimize"] = False
    # -- optimize kwargs
    # run_params["min_method"] = 'lm'
    # -- emcee kwargs
    # run_params["nwalkers"] = 128  # numebr of walkers
    # run_params["niter"] = 512  # number of iterations of the MCMC sampling
    # run_params["nburn"] = [16, 32, 64]  # number of iterations in each round of burn-in
    # -- dynesty kwargs
    # -- nautilus kwargs
    # run_params['']

    # Select observations
    new_obs = [o for o, k in zip(obs, fit_obs.keys()) if fit_obs.get(k, True)]

    # Fit model
    start = time()
    # output = fit_model(spec_obs, model, sps, lnprobfn=lnprobfn, **run_params)
    output = fit_model(new_obs, model, sps, lnprobfn=lnprobfn, **run_params)
    end = time()
    print(output)
    print(f"Model fit in {end - start:.2f} seconds")

    out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329"
    out_name = "zf-uds-7329_dynesty_mcmc.h5"
    out_path = os.path.join(out_dir, out_name)
    writer.write_hdf5(out_path, run_params, model, obs,
                        sampling_result=output["sampling"], 
                        # optimize_result_tuple=output["optimization"],
                        sps=sps
                    )
    
    print(f"Output saved to {out_path}")

# --------------
# Run prospector
# --------------
if __name__ == "__main__":
    main()