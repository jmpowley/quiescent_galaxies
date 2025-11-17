from pysersic.loss import student_t_loss_free_sys

from scalpel import Scalpel

# Set up config dictionary
config = {

    "cutout_kwargs" : {

        "F090W_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F090W_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f090w.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f090w",
        },

        "F115W_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F115W_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f115w.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f115w",
        },

        "F150W_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F150W_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f150w.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f150w",
        },

        "F200W_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F200W_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f200w.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f200w",
        },

        "F277W_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F277W_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f277w.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f277w",
        },

        "F356W_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F356W_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f356w.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f356w",
        },

        "F410M_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F410M_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f410m.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f410m",
        },

        "F444W_kwargs" : {
            "data_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/zf-uds-7329/cutouts",
            "data_name" : "ZF-UDS-7329_F444W_cutout.fits",
            "data_ext" : "SCI",
            "centre" : (215, 239),
            "width" : 43,
            "height" : 43,
            "psf_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/data_processed/model_psfs",
            "psf_name" : "mpsf_f444w.fits",
            "psf_ext" : None,
            "snr_limit" : 10,
            "plot" : False,
            "filter" : "f444w",
        },
    },

    "prior_dict" : {
        "r_eff_1" : (0.5, 20.0),
        "r_eff_2" : (0.5, 20.0),
        "ellip_1" : (0.0, 1.0),
        "ellip_2" : (0.0, 1.0),
        "f_1" : (0.0, 1.0),
        "n" : (2.0, 4.5),  # Sersic index of the central component
    },

    "fit_kwargs" : {
        "fit_type" : "simultaneous",
        "profile_type" : "sersic_exp",
        "sky_type" : "none",
        "loss_func" : "student_t",
        "seed" : 1000,
        "verbose" : True,
        "linked_params" : ["f_1", "n"],
        "const_params" : ["xc", "yc", "theta", "ellip_1", "ellip_2", "r_eff_1", "r_eff_2"],
        "out_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/outputs/zf-uds-7329/scalpel_test",
        "fig_dir" : "/Users/Jonah/PhD/Research/quiescent_galaxies/figures/zf-uds-7329/scalpel_test",
    },
}

# Create Scalpel object
scalpel = Scalpel(config)
scalpel.dissect()