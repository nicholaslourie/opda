"""Utilities"""

import numpy as np
from scipy import stats

from ersa import exceptions


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


def beta_equal_tailed_interval(a, b, coverage):
    """Return an interval containing ``coverage`` of the probability.

    For the beta distribution with parameters ``a`` and ``b``, return
    the equal-tailed interval that contains ``coverage`` of the
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
        upper bound for the equal-tailed intervals.
    """
    a = np.array(a)
    b = np.array(b)
    coverage = np.array(coverage)

    beta = stats.beta(a, b)

    x = beta.ppf((1. - coverage) / 2.)
    y = beta.ppf((1. + coverage) / 2.)

    return x, y


def beta_hpd_interval(a, b, coverage, atol=1e-10):
    """Return an interval containing ``coverage`` of the probability.

    For the beta distribution with parameters ``a`` and ``b``, return
    the shortest interval that contains ``coverage`` of the
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
    # NOTE: Given the lower endpoint of the interval, ``x``, we can immediately
    # compute the upper one as: ``beta.ppf(beta.cdf(x) + coverage)``. Below the
    # interval, the density of the lower endpoint is less than the upper
    # one. Above the interval, it's the reverse. Thus, we can find the lower
    # endpoint via binary search.
    #
    # The beta distribution only has a mode when ``a`` or ``b`` is greater than
    # 1. If both are greater than 1, the mode is in the interior of [0, 1]. If
    # ``a`` or ``b`` is less than or equal to 1, then the mode is on the
    # boundary. If ``a`` and ``b`` are less than or equal to 1, then the mode
    # is not unique and the highest density region is not necessarily an
    # interval.
    a = np.array(a)
    b = np.array(b)
    coverage = np.array(coverage)

    if np.any((a <= 1.) & (b <= 1.)):
        raise ValueError(
            'Either a or b must be greater than one to have an HPD interval.'
        )

    beta = stats.beta(a, b)

    # Initialize bounds.
    x_lo = np.where(b <= 1., beta.ppf(1. - coverage), 0.)
    x_hi = np.where(a <= 1., 0., beta.ppf(1. - coverage))
    # Binary search for the lower endpoint.
    for _ in range(1_000):
        x = (x_lo + x_hi) / 2.
        y = beta.ppf(np.clip(beta.cdf(x) + coverage, 0., 1.))

        x_lo = np.where(beta.pdf(x) < beta.pdf(y), x, x_lo)
        x_hi = np.where(beta.pdf(x) >= beta.pdf(y), x, x_hi)

        if np.all(x_hi - x_lo < atol):
            break
    else:
        raise exceptions.OptimizationException(
            'beta_hpd_interval failed to converge.'
        )

    return x, y


def beta_equal_tailed_coverage(a, b, x):
    """Return the coverage of the smallest interval containing ``x``.

    For the beta distribution with parameters ``a`` and ``b``, return
    the coverage of the smallest equal-tailed interval containing
    ``x``. See the related function: ``beta_equal_tailed_interval``.

    Parameters
    ----------
    a : float or array of floats, required
        The alpha parameter for the beta distribution.
    b : float or array of floats, required
        The beta parameter for the beta distribution.
    x : float or array of floats, required
        The points defining the minimal equal-tailed intervals whose
        coverage to return.

    Returns
    -------
    float or array of floats
        A float or array of floats with shape determined by broadcasting
        ``a``, ``b``, and ``x`` together. The values represent the
        coverage of the minimal equal-tailed interval containing the
        corresponding value from ``x``.
    """
    a = np.array(a)
    b = np.array(b)
    x = np.array(x)

    beta = stats.beta(a, b)

    return 2 * np.abs(0.5 - beta.cdf(x))


def beta_hpd_coverage(a, b, x, atol=1e-10):
    """Return the coverage of the smallest interval containing ``x``.

    For the beta distribution with parameters ``a`` and ``b``, return
    the coverage of the smallest hpd interval containing ``x``. See the
    related function: ``beta_hpd_interval``.

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
        coverage of the minimal hpd interval containing the
        corresponding value from ``x``.
    """
    # Use binary search to find the coverage of the HPD interval
    # containing x.
    a = np.array(a)
    b = np.array(b)
    x = np.array(x)

    if np.any((a <= 1.) & (b <= 1.)):
        raise ValueError(
            'Either a or b must be greater than one to have an HPD interval.'
        )

    beta = stats.beta(a, b)

    mode = (a - 1) / (a + b - 2)
    mode = np.where(a <= 1., 0., mode)
    mode = np.where(b <= 1., 1., mode)
    x_is_lower_end = x < mode

    # Initialize bounds.
    y_lo = np.where(x_is_lower_end, mode, 0.)
    y_hi = np.where(x_is_lower_end, 1., mode)
    # Binary search for the other end.
    for _ in range(1_000):
        y = (y_lo + y_hi) / 2.

        x_is_lower_pdf = beta.pdf(x) < beta.pdf(y)
        y_lo = np.where(x_is_lower_end == x_is_lower_pdf, y, y_lo)
        y_hi = np.where(x_is_lower_end != x_is_lower_pdf, y, y_hi)

        if np.all(y_hi - y_lo < atol):
            break
    else:
        raise exceptions.OptimizationException(
            'beta_hpd_coverage failed to converge.'
        )

    x, y = np.where(x_is_lower_end, x, y), np.where(x_is_lower_end, y, x)

    return beta.cdf(y) - beta.cdf(x)
