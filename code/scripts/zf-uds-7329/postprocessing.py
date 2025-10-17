import numpy as np

from dynesty.plotting import _quantile as weighted_quantile

def return_sfh(results, theta):
        """Returns the star formation history, or the  mass formed in each age bin, from the inputted theta values"""

        # Extract variables from results
        # -- parameters in chain
        chain_names = getattr(results['chain'].dtype, "names", None)
        # -- all model parameters
        model_params = results['model_params']
        age_bins = model_params['agebins']
        nbins = np.squeeze(model_params['nbins_sfh'])

        # Create bins
        firstindex = chain_names.index('logsfr_ratios')
        sfr_bins = []  # Normalised SFR bins (so s_0 = 1)
        ratio_bins = 10**theta[firstindex:firstindex+nbins-1]  # bins of SFR ratios
        
        # Calculate age bin widths (yr):
        delta_t = np.array([10**(age_bin[1]) - 10**age_bin[0] for age_bin in age_bins])
        for i in range(0, nbins):
            sfr_bins.append((1. / np.prod(ratio_bins[:i])))
        sfr_bins = np.array(sfr_bins)

        # Calculate mass formed in each bins
        log_mass = theta[chain_names.index('logmass')]
        zred = theta[chain_names.index('zred')]
        M_bin = sfr_bins * 10**log_mass / np.sum(delta_t * sfr_bins)
        M_bin = np.squeeze(M_bin).tolist()
        
        return M_bin, log_mass

def return_sfh_for_one_sigma_quantiles(sfh_chain, weights):
    """Returns the 16th, 50th and 84th quantile of a chain of star formation histories using the weights of the results
    """
    
    # Get the weighted quantiles of the SFH chain
    sfh_16, sfh_50, sfh_84 = np.squeeze(np.array([
        [weighted_quantile(sfh, q=quantile, weights=weights) for sfh in sfh_chain.T] for quantile in (.16, .50, .84)]
        ))
    
    return sfh_16, sfh_50, sfh_84

def return_sfh_chain(results):
    """ Return the chain of star formation histories from the `prospector` numeric/unstructured chain of model results
    """

    # Extract variables from results
    numeric_chain = results["unstructured_chain"]
    age_bins = results['model_params']['agebins']

    # Call return_sfh at each point on the chain
    sfh_chain = np.array([return_sfh(results, theta, age_bins)[0] for theta in numeric_chain])

    return sfh_chain