import os

import numpy as np
import matplotlib.pyplot as plt

import corner

from prospect.plotting.corner import allcorner

# -------------------------------
# Functions to plot model results
# -------------------------------
def subcorner(results, showpars=None, truths=None,
              start=0, thin=1, chains=slice(None),
              logify=("mass", "tau"), **kwargs):
    """
    
    """

    import corner as triangle

    # Determine full list of parameter names
    try:
        parnames = getattr(results['chain'].dtype, "names", None)
        parnames = list(parnames)
    except Exception as e:
        raise Exception(f"Error: {e}")
    # -- decide which parameters to show
    if showpars is not None:
        ind_show = np.array([parnames.index(p) for p in showpars])
        parnames = [parnames[i] for i in ind_show]
    else:
        ind_show = slice(None)

    # Convert chain to unstructured ndarray
    numeric = results['unstructured_chain'][..., ind_show]
    # --- select trace
    trace = numeric[chains, start::thin]
    # --- flatten (nwalkers, nsteps, nparams) -> (N, nparams) for corner
    if trace.ndim > 2:
        samples = trace.reshape(-1, trace.shape[-1])
    else:
        samples = trace.copy()
    # -- select weights
    weights = results.get('weights', None)
    if weights is not None:
        weights = weights[start::thin]

    # log-ify some parameters
    xx = samples.copy()
    if truths is not None:
        xx_truth = np.array(truths).copy()[..., ind_show]
    else:
        xx_truth = None
    # for p in logify:
    #     if p in parnames:
    #         idx = parnames.index(p)
    #         xx[:, idx] = np.log10(xx[:, idx])
    #         parnames[idx] = "log({})".format(parnames[idx])
    #         if truths is not None:
    #             xx_truth[idx] = np.log10(xx_truth[idx])

    # -- scale corner plot to show percentiles of data
    p_lo, p_hi = 5, 95
    # p_lo, p_hi = 2.5, 97.5
    # p_lo, p_hi = 0, 100
    ranges = []
    for i in range(xx.shape[1]):
        lo, hi = np.percentile(xx[:, i], [p_lo, p_hi])
        if lo == hi:
            # degenerate: spread a little around the single value
            lo -= 0.5
            hi += 0.5
        pad = 0.05 * (hi - lo)  # 5% padding
        ranges.append((lo - pad, hi + pad))

    # --- corner defaults and call ---
    corner_kwargs = {"plot_datapoints": False, "plot_density": False,
                     "fill_contours": True, "show_titles": True}
    corner_kwargs.update(kwargs)
    corner_kwargs["range"] = ranges
    
    fig = triangle.corner(xx, labels=parnames, truths=xx_truth,
                          quantiles=[0.14, 0.5, 0.86], weights=weights,
                          **corner_kwargs)

    return fig

def call_subcorner(results, showpars, truths, color, fig_dir, fig_name, savefig):
    """ Very small wrapper around modified `subcorner` function in `propsect.io.read_results`
    """

    # Create figure
    npars = len(showpars)
    figscale = 2 * npars
    fig = plt.subplots(npars, npars, figsize=(figscale, figscale))[0]

    # Call subcorner to create plot
    cornerfig = subcorner(results, 
                        showpars=showpars,
                        start=0,
                        thin=1,
                        truths=truths,
                        fig=fig, color=color)
    
    cornerfig.tight_layout(rect=[0, 0, 0.975, 1])

    # Save figure
    if savefig:
        cornerfig.savefig(os.path.join(fig_dir, fig_name), dpi=400)

    return cornerfig

def call_allcorner(samples, labels, weights, fig_dir, fig_name, savefig=True, **kwargs):

    ndim = samples.shape[0]
    assert len(labels) == ndim, (
        f"`labels` (len = {len(labels)}) must match number of model dimensions in `samples` "
        f"(ndim = {ndim})"
    )

    # Create figure with correct shape
    fig, axes = plt.subplots(ndim, ndim, figsize=(2*ndim, 2*ndim))
    
    # Call allcorner
    axes = allcorner(
        samples,
        labels=labels,
        axes=axes,
        weights=weights,
        **kwargs
    )
    # -- extra prettifying
    plt.tight_layout()

    # Save figure
    if savefig:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=300)

    return fig