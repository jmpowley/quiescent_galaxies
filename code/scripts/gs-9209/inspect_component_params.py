import os
import glob

import asdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sedpy.observate import load_filters

from scalpel.helpers import return_linked_param_at_wv, return_const_param, return_median_model_from_individual_tree, return_median_model_from_joint_tree, return_joint_fit_medians_at_filter, return_individual_fit_posteriors
from scalpel.plotting import plot_residuals_grid

def plot_linked_parameters(joint_tree, ind_trees, filters, linked_params, wv_to_save, waveffs):
    
    joint_results = joint_tree['results_dict']

    # Create figure
    n_linked = len(linked_params)
    fig, axes = plt.subplots(nrows=1, ncols=n_linked, figsize=(4*n_linked, 4))

    # Loop over each linked param
    for i, linked_param in enumerate(linked_params):
        ax = axes[i]
        
        # Plot joint fit at each wv_to_save
        param_at_wv, std_at_wv = return_linked_param_at_wv(joint_results, param=linked_param, return_std=True)
        ax.plot(wv_to_save, param_at_wv, color='crimson', label="Multi-fit")
        ax.fill_between(wv_to_save, param_at_wv-std_at_wv, param_at_wv+std_at_wv, color='crimson', alpha=0.3)

        # Plot fit for each cutout
        for j, (filter, ind_tree) in enumerate(zip(filters, ind_trees)):

            # -- joint fit
            param_at_f, std_at_f = return_joint_fit_medians_at_filter(joint_tree, params=[linked_param], filter=filter, return_list=True, return_unc=True)
            ax.errorbar(waveffs[j], param_at_f, yerr=std_at_f, color="crimson", fmt="o", label=f"Joint" if j == 0 else None)
            # -- individual fit
            _, ind_meds, ind_uncs = return_individual_fit_posteriors(ind_tree, params=[linked_param], return_medians=True, return_uncs=True)
            ax.errorbar(waveffs[j], ind_meds, yerr=np.array([[ind_uncs[0][0]], [ind_uncs[0][1]]]), color='k', fmt="o", label="Ind." if j == 0 else None)

        # Prettify
        ax.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', size=16)
        ax.set_ylabel(f"{linked_param}", size=16)
        ax.set_ylabel(f"B/T" if linked_param == "f_1" else f"{linked_param}", size=16)
        # ax.tick_params(axis='both' , direction='in')
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

# Load output files
out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/gs-9209/gs-9209_sersic"
# out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/gs-9209/scalpel_test"
# out_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/gs-9209/gs-9209_sersic_exp"
# -- joint fit
out_name = "sersic_sim_mcmc_poly_fit_results.asdf"
# out_name = "sersic_sim_mcmc_bspline_fit_results.asdf"
# out_name = "sersic_exp_sim_mcmc_bspline_fit_results.asdf"
# out_name = "sersic_exp_sim_mcmc_poly_fit_results.asdf"
joint_tree = asdf.open(os.path.join(out_dir, out_name))

# Load joint tree data
joint_results = joint_tree['results_dict']
filters = joint_tree['filters']
cutout_kwargs = joint_tree['cutout_kwargs']
prior_dict = joint_tree['prior_dict']
fit_kwargs = joint_tree['fit_kwargs']
profile_type = fit_kwargs["profile_type"]
method = fit_kwargs["method"]
multifitter = fit_kwargs["multifitter"]

# Access fit parameters
linked_params = fit_kwargs['linked_params']
const_params = fit_kwargs['const_params']
all_params = linked_params + const_params

print(all_params)

waveffs = np.asarray([load_filters(['jwst_' + filter])[0].wave_effective / 1e4 for filter in filters])
wv_to_save = joint_tree["wv_to_save"]

# Load individual trees
images = []
ind_trees = []
ind_models = []
ind_resids = []
joint_models = []
joint_resids = []
for filter in filters:
    ind_tree_path = os.path.join(out_dir, f"{profile_type}_{method}_fit_{filter}.asdf")
    ind_tree = asdf.load(ind_tree_path)
    # -- individual fit model
    ind_model, image, ind_resid = return_median_model_from_individual_tree(ind_tree, profile_type, return_image=True, return_residual=True)
    # -- joint fit model
    joint_model, image, joint_resid = return_median_model_from_joint_tree(joint_tree, ind_tree, profile_type, filter, return_image=True, return_residual=True)
    # -- add to lists
    # images.append(image)
    ind_trees.append(ind_tree)
    ind_models.append(ind_model)
    ind_resids.append(ind_resid)
    joint_models.append(joint_model)
    joint_resids.append(joint_resid)

# Make plots
# fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/gs-9209/scalpel_test"
fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/gs-9209/gs-9209_sersic"
# fig_dir = "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/gs-9209/gs-9209_sersic_exp"
# -- linked parameters
fig = plot_linked_parameters(joint_tree=joint_tree, ind_trees=ind_trees, filters=filters, linked_params=linked_params, wv_to_save=wv_to_save, waveffs=waveffs)
fig_name = f"multifit_{multifitter}_linked_parameters.png"
fig.savefig(os.path.join(fig_dir, fig_name), dpi=400)
# -- residuals for all filters with individual fit
# fig_name = f"{profile_type}_{method}_indifit_allfilters_residual.pdf"
# fig = plot_residuals_grid(images, masks=None, models=ind_models, residuals=ind_resids, filters=filters)
# fig.savefig(os.path.join(fig_dir, fig_name))
# -- residuals for all filters with joint fit
# fig_name = f"{profile_type}_{method}_jointfit_allfilters_residual.pdf"
# fig = plot_residuals_grid(images, masks=None, models=joint_models, residuals=joint_resids, filters=filters)
# fig.savefig(os.path.join(fig_dir, fig_name))

plt.show()