import numpy as np

from prospect.observation import Photometry, Spectrum

def build_mock_obs(filterset=None, wavelength=None):
    """Make a mock Observation class to use when making model predictions.

    Parameters
    ----------
    filterset : 
        Set of `sedpy filters to load into the `prospect.observation.Photometry` class. Either a list or single filter.
    wavelength : array-like
        Wavelength data to load into the `prospect.observation.Spectrum` class. Either a list or single array-like object

    Returns
    -------
    mock_obs : list
        List of `prospect.observation.Observation` objects loaded from the filterset and wavelength variable inputs
    """
    
    # Create observations
    mock_obs = []

    # Add photometry
    if filterset is not None:
        phot = Photometry(filters=filterset,
                    flux=None,
                    uncertainty=None,
                    mask=None,
                    noise=None,
                    )
        mock_obs.append(phot)
    
    # Add spectra
    if wavelength is not None:
        # -- multiple
        if type(wavelength) is list:
            for wave in wavelength:
                spec = Spectrum(wavelength=wave,
                                flux=None,
                                uncertainty=None,
                                mask=None,
                                noise=None,
                                )
                mock_obs.append(spec)
        # -- single
        else:
            spec = Spectrum(wavelength=wavelength,
                            flux=None,
                            uncertainty=None,
                            mask=None,
                            noise=None,
                            )
            mock_obs.append(spec)

    return mock_obs

def predict_mock_obs(model, theta, observations, mock_observations, sps, snr_spec=20.0, snr_phot=20.0, add_noise=False, seed=101, **model_params):

    # Update model params
    params = {}
    for p in model.params.keys():
        if p in model_params:
            params[p] = np.atleast_1d(model_params[p])
    model.params.update(params)

    # Make predictions from mock obs
    mock_pred, mfrac = model.predict(theta=theta,
                                     observations=mock_observations,
                                     sps=sps
                                     )

    # Add in uncertainty/noise
    for pred, obs in zip(mock_pred, observations):
        # -- photometry
        if obs.kind == "photometry" and add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            # -- add noise from SNR to predictions
            phot_noise_sigma = pred / snr_phot
            phot_noise = np.random.normal(0, 1, size=len(obs.wavelength)) * phot_noise_sigma
            pred += phot_noise
        # -- spectra
        if obs.kind == "spectrum" and add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            # -- add noise from SNR to predictions
            spec_noise_sigma = pred / snr_spec
            spec_noise = np.random.normal(0, 1, size=len(obs.wavelength)) * spec_noise_sigma
            pred += spec_noise

    return mock_pred, mfrac