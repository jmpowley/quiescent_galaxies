import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from pysersic.results import plot_residual
from pysersic.rendering import HybridRenderer

def call_plot_residual(fitter, im, mask, psf, profile_type):

    # Generate best fit model
    summary = fitter.sampling_results.summary()
    dict = {}
    for hdul, b in zip(summary.index, summary["mean"]):
        dict[hdul] = b
    best_model = HybridRenderer(im.shape, jnp.array(psf.astype(np.float32))).render_source(dict, profile_type=profile_type)
    
    fig, ax = plot_residual(im, best_model, mask=mask, vmin=-1, vmax=1)

    return fig

def plot_best_sersic_exp_fit(multi_results, filters, multi_results_ind=None):
    pass

def plot_best_fit_from_posteriors(tree, profile_type, method):

    input_data = tree["input_data"]
    posteriors = tree["posterior"]

   # Convert ASDF arrays to numpy arrays
    image = np.array(input_data["image"])
    mask = np.array(input_data["mask"])
    psf = np.array(input_data["psf"])
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)

    params_median = {}

    if profile_type == "sersic":
        for key in ["ellip", "flux", "n", "r_eff", "theta", "xc", "yc"]:
            # -- load posterior samples
            if method == "mcmc":
                posterior = posteriors[key][1, :]
            else:
                posterior = posteriors[key]
            # -- extract median parameters
            params_median[key] = np.nanmedian(posterior)

    model = renderer.render_source(params_median, profile_type=profile_type)
    model = np.array(model)

    fig, axes = plot_residual(image, model=model, mask=~mask, vmin=-1, vmax=1)

    return fig, axes

def plot_residuals_grid(trees, filters, profile_type="sersic"):
    
    # Create figure
    ncols = 3
    nrows = len(trees)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4*nrows), squeeze=False)

    # Loop over each tree
    for i, (tree, filter) in enumerate(zip(trees, filters)):
        ax = axes[i, :]

        # Load data
        # -- image data
        input_data = tree["input_data"]
        image = np.asarray(input_data["image"])
        mask = np.asarray(input_data["mask"])
        psf_np = np.asarray(input_data["psf"]).astype(np.float32)
        # -- samples
        posterior = tree["posterior"]
        param_medians = {key : np.nanmedian(val) for key, val in posterior.items()}
        
        # Render model and residual
        renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf_np)
        model = np.asarray(renderer.render_source(param_medians, profile_type=profile_type))
        resid = image - model

        # Make plot
        # -- scale limits
        vmin = np.nanmean(image) - (2 * np.nanstd(image, ddof=1))
        vmax = np.nanmean(image) + (2 * np.nanstd(image, ddof=1))
        # -- image
        ax[0].imshow(image, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
        ax[0].text(0.03, 0.97, f"{filter.upper()}", transform=ax[0].transAxes, va="top", ha="left", size=14)
        # -- model
        ax[1].imshow(model, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
        # -- residual
        im_resid = ax[2].imshow(resid, origin="lower", cmap="seismic", vmin=-1, vmax=1)
        ax_divider = make_axes_locatable(ax[2])
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im_resid, cax=cax)

    plt.tight_layout()
    
    return fig