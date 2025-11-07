import numpy as np
import jax.numpy as jnp

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