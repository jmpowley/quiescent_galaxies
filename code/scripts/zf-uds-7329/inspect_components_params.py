import os
import glob

import asdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sedpy.observate import load_filters

from scalpel.helpers import return_linked_param, return_const_param
from scalpel.plotting import plot_residuals_grid

def plot_linked_parameters(results, filters, linked_params, wv_to_save, waveffs, ind_meds=None, ind_uncs=None):
    
    # Create figure
    n_linked = len(linked_params)
    fig, axes = plt.subplots(nrows=1, ncols=n_linked, figsize=(4*n_linked, 4))

    # Loop over each linked param
    for i, param in enumerate(linked_params):
        ax = axes[i]
        
        # Plot fit for each param
        param_at_wv, std_at_wv = return_linked_param(results, param=param, return_std=True)
        ax.plot(wv_to_save, param_at_wv, color='crimson', label="Multi-fit")
        ax.fill_between(wv_to_save, param_at_wv-std_at_wv, param_at_wv+std_at_wv, color='crimson', alpha=0.3)

        # Plot value for each cutout
        for j, filter in enumerate(filters):
            param_at_f, std_at_f = results[f"{param}_{filter}"]
            ax.errorbar(waveffs[j], param_at_f, yerr=std_at_f, color="crimson", fmt="o", label=f"Joint" if j == 0 else None)

            # Plot individual fit results
            if ind_meds is not None and ind_uncs is not None:
                med = ind_meds[filter][param]
                unc = ind_uncs[filter][param]
                ax.errorbar(waveffs[j], med, yerr=np.array([[unc[0]], [unc[1]]]), color='k', fmt="o", label="Ind." if j == 0 else None)

        # Prettify
        ax.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', size=16)
        ax.set_ylabel(f"{param}", size=16)
        ax.set_ylabel(f"B/T" if param == "f_1" else f"{param}", size=16)
        axes[0].legend()

    plt.tight_layout()

    return fig

def plot_const_parameters(results, const_params):
    
    # Create figure
    n_const = len(const_params)
    fig, axes = plt.subplots(nrows=1, ncols=n_const, figsize=(14, 6))

    # Loop over each const param
    for i, param in enumerate(const_params):
        ax = axes[i]
        
        # Plot fit for each param
        val, std = return_const_param(results, param, return_std=True)
        ax.axvline(val, color='crimson')
        ax.axvspan(val-std, val+std, color='crimson', alpha=0.3)
        ax.set_title(f"{param}")
    
    plt.tight_layout()

    return fig

def return_individual_fit_posterior(path, params):

    tree = asdf.open(path)
    post_dict = tree['posterior']

    param_posts = {}
    param_meds = {}
    param_uncs = {}

    for p in params:
        param_post = post_dict[p][1]  # use second output ()
        param_post_50 = np.nanpercentile(param_post, q=50)
        param_post_16 = np.nanpercentile(param_post, q=16)
        param_post_84 = np.nanpercentile(param_post, q=84)

        param_posts[p] = param_post
        param_meds[p] = param_post_50
        param_uncs[p] = [param_post_50-param_post_16, param_post_84-param_post_50]

    return param_posts, param_meds, param_uncs

# Load output file
# out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/gs-9209/scalpel_test"
# out_name = "sersic_exp_sim_fit_results.asdf"
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/gs-9209/gs-9209_sersic"
out_name = "sersic_sim_mcmc_poly_fit_results.asdf"
# out_name = "sersic_sim_mcmc_bspline_fit_results.asdf"
# out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/scalpel_test"
# out_name = "sersic_exp_sim_mcmc_bspline_fit_results.asdf"
# out_name = "sersic_exp_sim_mcmc_poly_fit_results.asdf"
tree = asdf.open(os.path.join(out_dir, out_name))

# Load tree data
results = tree['results_dict']
filters = tree['filters']
cutout_kwargs = tree['cutout_kwargs']
prior_dict = tree['prior_dict']
fit_kwargs = tree['fit_kwargs']
profile_type = fit_kwargs["profile_type"]
method = fit_kwargs["method"]
multifitter = fit_kwargs["multifitter"]

# Access fit parameters
linked_params = fit_kwargs['linked_params']
const_params = fit_kwargs['const_params']
params = linked_params + const_params
waveffs = np.asarray([load_filters(['jwst_' + filter])[0].wave_effective / 1e4 for filter in filters])
wv_to_save = tree["wv_to_save"]

# Load individual fits
ind_posts = {}
ind_param_meds = {}
ind_param_uncs = {}
for f in filters:
    ind_path = os.path.join(out_dir, f"{profile_type}_{method}_fit_{f}.asdf")
    param_posts, param_meds, param_uncs = return_individual_fit_posterior(ind_path, params)
    
    ind_posts[f] = param_posts
    ind_param_meds[f] = param_meds
    ind_param_uncs[f] = param_uncs

# Load individual trees
ind_names = f"{profile_type}_{method}_fit*.asdf"
ind_paths = glob.glob(os.path.join(out_dir, ind_names))
ind_trees = [asdf.load(path) for path in ind_paths]

# Make plots
# fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/gs-9209/scalpel_test"
fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/gs-9209/gs-9209_sersic"
# fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/zf-uds-7329/scalpel_test"
# -- linked parameters
fig = plot_linked_parameters(results=results, filters=filters, linked_params=linked_params, wv_to_save=wv_to_save, waveffs=waveffs, ind_meds=ind_param_meds, ind_uncs=ind_param_uncs)
# fig_name = "gs-9209_multifit_bspline_linked_parameters.png"
fig_name = f"multifit_{multifitter}_linked_parameters.png"
fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)
# -- residuals for all filters
fig_name = f"{profile_type}_{method}_allfilters_residual.pdf"
fig = plot_residuals_grid(ind_trees, filters, profile_type=profile_type)
fig.savefig(os.path.join(fig_dir, fig_name))

plt.show()