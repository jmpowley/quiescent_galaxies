import numpy as np

from astropy import cosmology

# ----------------------------
# Functions for building model
# ----------------------------
def convert_zred_to_agebins(zred=None, nbins_sfh=None, **extras):
        """ Returns age bins going [0, 10Myr, 30Myr, 100Myr, ...] and nbin-2 equally spaced (in logtime) bins from 100Myr to age_universe
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
        agelims = [0.0, 7.0, 7.4772] \
                    + np.linspace(8.0, 9.0, ncomp-6, endpoint=False).tolist() \
                    + np.log10(np.linspace(1.*10**9, tbinmax, 4, endpoint=True)).tolist()
        agebins = np.array([agelims[:-1], agelims[1:]])
        
        return agebins.T

def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
        """ Computes masses formed in each bin from SFR ratios
        """

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
        """Convert second component of dust model to first component based on fitted fraction of second dust model
        """

        dust1 = dust1_fraction*dust2
        
        return dust1
