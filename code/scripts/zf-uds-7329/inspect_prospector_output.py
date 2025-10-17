import os

import numpy as np
import matplotlib.pyplot as plt

import prospect.io.read_results as reader

from plotting import call_subcorner
from postprocessing import return_sfh, return_sfh_chain, return_sfh_for_one_sigma_quantiles

def plot_sfh(age_bins, sfh_best, sfh_chain, logscale=False):

        logages = np.array(age_bins).ravel()
        ages = 10**(logages-9) # ages in Gyr

        sfh_16, sfh_50, sfh_84 = return_sfh_for_one_sigma_quantiles(sfh_chain, weights)

        # Create figure
        fig, ax = plt.subplots()

        # Plot SFHs
        # -- best fit
        ax.plot(ages, [val for val in sfh_best for _ in (0,1)], color='blue', label=u'MAP')
        # -- median
        ax.plot(ages, [val for val in sfh_50 for _ in (0,1)], color='red', label='Median')
        # -- 16th and 84th percentiles
        ax.fill_between(ages, 
                        [val for val in sfh_16 for _ in (0,1)],
                        [val for val in sfh_84 for _ in (0,1)], 
                        color="red", alpha=0.2, linewidth=0.)
        
        ax.set_xlabel(r'$t_{\mathrm{obs}} - t$ (Gyr)')
        ax.set_ylabel('SFR '+ u'(M$_\u2609$/yr)')
        if logscale:
            ax.set_yscale('log')
            plt.legend(loc='upper left')
        #     plt.savefig(f'{arbname}_SFR_log.png', bbox_inches='tight', pad_inches=0.1, dpi=800)
        else:
            plt.legend(loc='upper left')
        #     plt.savefig(f'{arbname}_SFR.png', bbox_inches='tight', pad_inches=0.1, dpi=800)

        return fig

# Load results from output file
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329"
# out_file = "zf-uds-7329_flat_model_nautilus.h5"
out_file = "zf-uds-7329_flat_model_nautilus_2.h5"
out_path = os.path.join(out_dir, out_file)
results_type = "nautlius"
results, obs, _ = reader.results_from(out_path.format(results_type), dangerous=False)

# print(results['paramfile_text'])

# sps = reader.get_sps(results)
# model = reader.get_model(results)
# print(model)

# Extract variables from results
model_params = results['model_params']
lnprob = results['lnprobability']
numeric_chain = results['unstructured_chain']
weights = results["weights"]

# Extract truths
imax = np.argmax(lnprob)
theta_best = numeric_chain[imax, :].copy()  # max of log probability
theta_med = np.nanmedian(numeric_chain, axis=0)  # median of posteriors

# Parameters to show in corner plot
showpars=['zred', 'logmass', 'logzsol', 'logsfr_ratios', 'gas_logz', 'gas_logu', 'eline_sigma']

# Extract SFHs
age_bins = np.asarray(model_params['agebins'])
sfh_best, logmass_best = return_sfh(results, theta_best, age_bins)
sfh_chain = return_sfh_chain(results, age_bins)  # chain of SFR vectors (solar masses / year)

# Make plots
fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/zf-uds-7329"
# -- corner plot
fig_name = "zf-uds-7329_prospector_cornerplot.png"
call_subcorner(results, showpars, truths=theta_best, color="purple", fig_dir=fig_dir, fig_name=fig_name, savefig=True)
# -- SFH plot
plot_sfh(age_bins, sfh_best, sfh_chain, logscale=False)

plt.show()