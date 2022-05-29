"""Utilities"""

import numpy as np


def sort_by_first(*args):
    """Return the arrays sorted by the first array.

    Parameters
    ----------
    *args : arrays, required
        The arrays to sort.

    Returns
    -------
    arrays
        The arrays sorted by the first array. Thus, the first array will
        be sorted and the other arrays will have their elements permuted
        the same way as the elements from the first array.
    """
    # Validate the arguments.
    args = tuple(map(np.array, args))
    if any(arg.shape != args[0].shape for arg in args):
        raise ValueError(
            'All argument arrays must have the same shape.'
        )

    # Return sorted copies of the arrays.
    if len(args) == 0:
        return ()

    sorting = np.argsort(args[0])
    return tuple(arg[sorting] for arg in args)


def dkw_epsilon(n, confidence):
    """Return epsilon from the Dvoretzky-Kiefer-Wolfowitz inequaltiy.

    The Dvoretzky-Kiefer-Wolfowitz inequality states that a confidence
    interval for the CDF is given by the empirical CDF plus or minus:

    .. math::

       \\epsilon = \\sqrt{\\frac{\\log \\frac{2}{\\alpha}}{2n}}

    Where :math:`1 - \\alpha` is the coverage.

    Parameters
    ----------
    n : int, required
        The number of samples.
    confidence : float, required
        The desired confidence or coverage.

    Returns
    -------
    float
        The epsilon for the Dvoretzky-Kiefer-Wolfowitz inequality.
    """
    n = np.array(n)
    confidence = np.array(confidence)

    return np.sqrt(
        np.log(2. / (1. - confidence))
        / (2. * n)
    )
