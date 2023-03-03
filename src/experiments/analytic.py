"""Analytic approximations to random search."""

import numpy as np
from scipy import special


def ellipse_volume(cs):
    """Return the volume of an ellipse with ``cs`` as the axes' lengths.

    Parameters
    ----------
    cs : array of floats, required
        A 1D array of floats providing the axes' lengths for the
        ellipse.

    Return
    ------
    float
        The volume of the ellipse.
    """
    cs = np.array(cs)
    if not len(cs.shape) == 1:
        raise ValueError(f'cs must be a 1D array, not have shape {cs.shape}.')

    n_dims, = cs.shape

    return (
        np.pi ** (n_dims / 2)
        / special.gamma(n_dims/2 + 1)
        * np.prod(cs)
    )
