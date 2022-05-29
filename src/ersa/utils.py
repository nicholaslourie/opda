"""Utilities"""

import numpy as np
from scipy import stats


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


def beta_ppf_interval(a, b, coverage):
    """Return an interval containing ``coverage`` of the probability.

    For the beta distribution with parameters ``a`` and ``b``, return
    the interval about the median that contains ``coverage`` of the
    probability mass.

    Parameters
    ----------
    a : float or array of floats, required
        The alpha parameter for the beta distribution.
    b : float or array of floats, required
        The beta parameter for the beta distribution.
    coverage : float, required
        The desired coverage for the returned intervals.

    Returns
    -------
    float or array of floats, float or array of floats
        A pair of floats or arrays of floats with the shape determined
        by broadcasting ``a``, ``b``, and ``coverage`` together. The
        first returned value gives the lower bound and the second the
        upper bound for the intervals.
    """
    a = np.array(a)
    b = np.array(b)
    coverage = np.array(coverage)

    beta = stats.beta(a, b)

    x = beta.ppf((1. - coverage) / 2.)
    y = beta.ppf((1. + coverage) / 2.)

    return x, y


def beta_ppf_coverage(a, b, x):
    """Return the coverage of the smallest interval containing ``x``.

    For the beta distribution with parameters ``a`` and ``b``, return
    the coverage of the smallest ppf interval containing ``x``. See the
    related function: ``beta_ppf_interval``.

    Parameters
    ----------
    a : float or array of floats, required
        The alpha parameter for the beta distribution.
    b : float or array of floats, required
        The beta parameter for the beta distribution.
    x : float or array of floats, required
        The points defining the minimal intervals whose coverage to
        return.

    Returns
    -------
    float or array of floats
        A float or array of floats with shape determined by broadcasting
        ``a``, ``b``, and ``x`` together. The values represent the
        coverage of the minimal ppf interval containing the
        corresponding value from ``x``.
    """
    a = np.array(a)
    b = np.array(b)
    x = np.array(x)

    beta = stats.beta(a, b)

    return 2 * np.abs(0.5 - beta.cdf(x))
