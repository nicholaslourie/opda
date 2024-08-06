"""Utilities."""

import numpy as np
from scipy import special, stats


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
            "All argument arrays must have the same shape.",
        )

    # Return sorted copies of the arrays.
    if len(args) == 0:
        return ()

    sorting = np.argsort(args[0])
    return tuple(arg[sorting] for arg in args)


def dkw_epsilon(n, confidence):
    r"""Return epsilon from the Dvoretzky-Kiefer-Wolfowitz inequaltiy.

    The Dvoretzky-Kiefer-Wolfowitz inequality states that a confidence
    interval for the CDF is given by the empirical CDF plus or minus:

    .. math::

       \epsilon = \sqrt{\frac{\log \frac{2}{\alpha}}{2n}}

    Where :math:`1 - \alpha` is the coverage.

    Parameters
    ----------
    n : positive int, required
        The number of samples.
    confidence : float from 0 to 1 inclusive, required
        The desired confidence or coverage.

    Returns
    -------
    non-negative float
        The epsilon for the Dvoretzky-Kiefer-Wolfowitz inequality.
    """
    # Validate the arguments.
    n = np.array(n)[()]
    if not np.isscalar(n):
        raise ValueError("n must be a scalar.")
    if n <= 0:
        raise ValueError("n must be positive.")

    confidence = np.array(confidence)[()]
    if not np.isscalar(confidence):
        raise ValueError("confidence must be a scalar.")
    if confidence < 0. or confidence > 1.:
        raise ValueError(
            "confidence must be between 0 and 1, inclusive.",
        )

    # Compute the DKW epsilon.
    if confidence == 1.:
        return np.inf

    return np.sqrt(
        np.log(2. / (1. - confidence))
        / (2. * n),
    )


def beta_equal_tailed_interval(a, b, coverage):
    """Return an interval containing ``coverage`` of the probability.

    For the beta distribution with parameters ``a`` and ``b``, return
    the equal-tailed interval that contains ``coverage`` of the
    probability mass.

    Parameters
    ----------
    a : finite positive float or array of floats, required
        The alpha parameter for the beta distribution.
    b : finite positive float or array of floats, required
        The beta parameter for the beta distribution.
    coverage : float or array of floats from 0 to 1 inclusive, required
        The desired coverage for the returned intervals.

    Returns
    -------
    pair of floats or arrays of floats from 0 to 1 inclusive
        A pair of floats or arrays of floats with the shape determined
        by broadcasting ``a``, ``b``, and ``coverage`` together. The
        first returned value gives the lower bound and the second the
        upper bound for the equal-tailed intervals.
    """
    # Validate the arguments.
    a = np.array(a)
    if np.any(a <= 0):
        raise ValueError("a must be positive.")
    if np.any(~np.isfinite(a)):
        raise ValueError("a must be finite.")

    b = np.array(b)
    if np.any(b <= 0):
        raise ValueError("b must be positive.")
    if np.any(~np.isfinite(b)):
        raise ValueError("b must be finite.")

    coverage = np.array(coverage)
    if np.any((coverage < 0.) | (coverage > 1.)):
        raise ValueError(
            "coverage must be between 0 and 1, inclusive.",
        )

    # Compute the equal-tailed interval.
    beta = stats.beta(a, b)

    x = beta.ppf((1. - coverage) / 2.)
    y = beta.ppf((1. + coverage) / 2.)

    return x, y


def beta_highest_density_interval(a, b, coverage, *, atol=1e-10):
    """Return an interval containing ``coverage`` of the probability.

    For the beta distribution with parameters ``a`` and ``b``, return
    the shortest interval that contains ``coverage`` of the
    probability mass. Note that the highest density interval only
    exists if at least one of ``a`` or ``b`` is greater than 1.

    Parameters
    ----------
    a : finite positive float or array of floats, required
        The alpha parameter for the beta distribution.
    b : finite positive float or array of floats, required
        The beta parameter for the beta distribution.
    coverage : float or array of floats from 0 to 1 inclusive, required
        The desired coverage for the returned intervals.
    atol : non-negative float, optional
        The absolute tolerance to use for stopping the iteration.

    Returns
    -------
    pair of floats or arrays of floats from 0 to 1 inclusive
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

    # Validate the arguments.
    a = np.array(a)
    if np.any(a <= 0):
        raise ValueError("a must be positive.")
    if np.any(~np.isfinite(a)):
        raise ValueError("a must be finite.")

    b = np.array(b)
    if np.any(b <= 0):
        raise ValueError("b must be positive.")
    if np.any(~np.isfinite(b)):
        raise ValueError("b must be finite.")

    coverage = np.array(coverage)
    if np.any((coverage < 0.) | (coverage > 1.)):
        raise ValueError(
            "coverage must be between 0 and 1, inclusive.",
        )

    if atol < 0.:
        raise ValueError("atol must be non-negative.")

    if np.any((a <= 1.) & (b <= 1.)):
        raise ValueError(
            f"Either a ({a}) or b ({b}) must be greater than one to have"
            f" a highest density interval.",
        )

    # Compute the highest density interval.
    beta = stats.beta(a, b)

    mode = np.clip((a - 1) / (a + b - 2), 0., 1.)

    # Initialize bounds.
    x_lo = beta.ppf(np.maximum(beta.cdf(mode) - coverage, 0.))
    x_hi = np.minimum(mode, beta.ppf(1. - coverage))

    # Binary search for the lower endpoint.
    # NOTE: Each iteration cuts the bracket's length in half, so run
    # enough iterations so that max(x_hi - x_lo) / 2**n_iter < atol.
    n_iter = int(np.ceil(
        # Even when the maximum bracket length is below atol, run at
        # least 1 iteration in order to compute the midpoint and y.
        np.log2(max(2, np.max(x_hi - x_lo) / atol)),
    ))
    for _ in range(n_iter):
        x = (x_lo + x_hi) / 2.
        y = beta.ppf(np.clip(beta.cdf(x) + coverage, 0., 1.))
        # NOTE: For small values of coverage, y (the upper confidence
        # limit) can fall below x (the lower confidence limit) when
        # computed as above due to discretization/rounding errors. In
        # general, y should be at or above the mode, so fix that below.
        y = np.clip(y, mode, 1.)

        # NOTE: Inline the unnormalized beta density rather than using
        # scipy.stats.beta.pdf because:
        #   * scipy.stats.beta.pdf is not monotonic from the
        #     boundaries to the mode. This bug causes the binary
        #     search to fail for small coverages.
        #   * The unnormalized version is significantly faster to
        #     compute.
        # In addition, raise the density to the 1/(b-1) power. This
        # transformation is monotonic, so it doesn't affect the points at
        # which the density is equal; however, it means we can avoid using
        # an expensive power operation on the large arrays.
        with np.errstate(divide="ignore"):
            x_pdf = x**((a-1)/(b-1)) * (1-x)
            y_pdf = y**((a-1)/(b-1)) * (1-y)

        x_lo = np.where(x_pdf <= y_pdf, x, x_lo)
        x_hi = np.where(x_pdf >= y_pdf, x, x_hi)

    return x, y


def beta_equal_tailed_coverage(a, b, x):
    """Return the coverage of the smallest interval containing ``x``.

    For the beta distribution with parameters ``a`` and ``b``, return
    the coverage of the smallest equal-tailed interval containing
    ``x``. See the related function:
    :py:func:`beta_equal_tailed_interval`.

    Parameters
    ----------
    a : finite positive float or array of floats, required
        The alpha parameter for the beta distribution.
    b : finite positive float or array of floats, required
        The beta parameter for the beta distribution.
    x : float or array of floats from 0 to 1 inclusive, required
        The points defining the minimal equal-tailed intervals whose
        coverage to return.

    Returns
    -------
    pair of floats or arrays of floats from 0 to 1 inclusive
        A float or array of floats with shape determined by broadcasting
        ``a``, ``b``, and ``x`` together. The values represent the
        coverage of the minimal equal-tailed interval containing the
        corresponding value from ``x``.
    """
    # Validate the arguments.
    a = np.array(a)
    if np.any(a <= 0):
        raise ValueError("a must be positive.")
    if np.any(~np.isfinite(a)):
        raise ValueError("a must be finite.")

    b = np.array(b)
    if np.any(b <= 0):
        raise ValueError("b must be positive.")
    if np.any(~np.isfinite(b)):
        raise ValueError("b must be finite.")

    x = np.array(x)
    if np.any((x < 0.) | (x > 1.)):
        raise ValueError(
            "x must be between 0 and 1, inclusive.",
        )

    # Compute the equal-tailed coverage.
    beta = stats.beta(a, b)

    return 2 * np.abs(0.5 - beta.cdf(x))


def beta_highest_density_coverage(a, b, x, *, atol=1e-10):
    """Return the coverage of the smallest interval containing ``x``.

    For the beta distribution with parameters ``a`` and ``b``, return
    the coverage of the smallest highest density interval containing
    ``x``. Note that the highest density interval only exists if at
    least one of ``a`` or ``b`` is greater than 1. See the related
    function: :py:func:`beta_highest_density_interval`.

    Parameters
    ----------
    a : finite positive float or array of floats, required
        The alpha parameter for the beta distribution.
    b : finite positive float or array of floats, required
        The beta parameter for the beta distribution.
    x : float or array of floats from 0 to 1 inclusive, required
        The points defining the minimal intervals whose coverage to
        return.
    atol : non-negative float, optional
        The absolute tolerance to use for stopping the iteration.

    Returns
    -------
    pair of floats or arrays of floats from 0 to 1 inclusive
        A float or array of floats with shape determined by broadcasting
        ``a``, ``b``, and ``x`` together. The values represent the
        coverage of the minimal highest density interval containing the
        corresponding value from ``x``.
    """
    # Validate the arguments.
    a = np.array(a)
    if np.any(a <= 0):
        raise ValueError("a must be positive.")
    if np.any(~np.isfinite(a)):
        raise ValueError("a must be finite.")

    b = np.array(b)
    if np.any(b <= 0):
        raise ValueError("b must be positive.")
    if np.any(~np.isfinite(b)):
        raise ValueError("b must be finite.")

    x = np.array(x)
    if np.any((x < 0.) | (x > 1.)):
        raise ValueError(
            "x must be between 0 and 1, inclusive.",
        )

    if atol < 0.:
        raise ValueError("atol must be non-negative.")

    if np.any((a <= 1.) & (b <= 1.)):
        raise ValueError(
            f"Either a ({a}) or b ({b}) must be greater than one to have"
            f" a highest density interval.",
        )

    # Compute the highest density coverage.

    # Use binary search to find the coverage of the highest density interval
    # containing x.
    beta = stats.beta(a, b)

    mode = np.clip((a - 1) / (a + b - 2), 0., 1.)
    x_is_lower_end = x < mode
    # NOTE: Inline the unnormalized beta density rather than using
    # scipy.stats.beta.pdf because:
    #   * scipy.stats.beta.pdf is not monotonic from the
    #     boundaries to the mode. This bug causes the binary
    #     search to fail for small coverages.
    #   * The unnormalized version is significantly faster to
    #     compute.
    # In addition, raise the density to the 1/(b-1) power. This
    # transformation is monotonic, so it doesn't affect the points at
    # which the density is equal; however, it means we can avoid using
    # a power operation on the large array of y's, which makes the
    # function significantly faster.
    with np.errstate(divide="ignore"):
        x_pdf = x**((a-1)/(b-1)) * (1-x)

    # Initialize bounds.
    y_lo = np.where(x_is_lower_end, mode, 0.)
    y_hi = np.where(x_is_lower_end, 1., mode)

    # Binary search for the other end.
    # NOTE: Each iteration cuts the bracket's length in half, so run
    # enough iterations so that max(y_hi - y_lo) / 2**n_iter < atol.
    n_iter = int(np.ceil(
        # Even when the maximum bracket length is below atol, run at
        # least 1 iteration in order to compute the midpoint and figure
        # out if x or y is the lower end.
        np.log2(max(2, np.max(y_hi - y_lo) / atol)),
    ))
    for _ in range(n_iter):
        y = (y_lo + y_hi) / 2.

        with np.errstate(divide="ignore"):
            y_is_lo = x_is_lower_end == (x_pdf < y**((a-1)/(b-1)) * (1-y))

        y_lo = np.where(y_is_lo, y, y_lo)
        y_hi = np.where(~y_is_lo, y, y_hi)

    x, y = np.where(x_is_lower_end, x, y), np.where(x_is_lower_end, y, x)

    return beta.cdf(y) - beta.cdf(x)


def binomial_confidence_interval(n_successes, n_total, confidence):
    """Return a confidence interval for the binomial distribution.

    Given ``n_successes`` out of ``n_total``, return an equal-tailed
    Clopper-Pearson confidence interval with coverage ``confidence``.

    Parameters
    ----------
    n_successes : non-negative int or array of ints, required
        An int or array of ints with each entry denoting the number of
        successes in a sample. Must be broadcastable with ``n_total``.
    n_total : positive int or array of ints, required
        An int or array of ints with each entry denoting the total
        number of observations in a sample. Must be broadcastable with
        ``n_successes``.
    confidence : float or array of floats from 0 to 1 inclusive, required
        A float or array of floats between zero and one denoting the
        desired confidence for each confidence interval. Must be
        broadcastable with ``n_successes`` broadcasted with ``n_total``.

    Returns
    -------
    pair of floats or arrays of floats from 0 to 1 inclusive
        A possibly scalar array of floats representing the lower
        confidence bounds and a possibly scalar array of floats
        representing the upper confidence bounds.

    Notes
    -----
    The Clopper-Pearson interval [1]_ does not account for the
    binomial distribution's discreteness. This lack of correction
    causes Clopper-Pearson intervals to be conservative. In addition,
    this function implements an equal-tailed version of the
    Clopper-Pearson interval which can be very conservative when the
    number of successes is zero or the total number of observations.

    References
    ----------
    .. [1] Clopper, C. and Pearson, E. S., "The Use of Confidence or
       Fiducial Limits Illustrated in the Case of the Binomial"
       (1934). Biometrika. 26 (4): 404-413. doi:10.1093/biomet/26.4.404.
    """
    # Validate the arguments.
    n_successes = np.array(n_successes)
    if not np.all(n_successes % 1 == 0):
        raise ValueError("n_successes must only contain integers.")
    if np.any(n_successes < 0):
        raise ValueError(
            f"n_successes ({n_successes}) must be greater than or equal"
            f" to 0.",
        )

    n_total = np.array(n_total)
    if not np.all(n_total % 1 == 0):
        raise ValueError("n_total must only contain integers.")
    if np.any(n_total < 1):
        raise ValueError(
            f"n_total ({n_total}) must be greater than or equal to 1.",
        )

    confidence = np.array(confidence)
    if np.any((confidence < 0.) | (confidence > 1.)):
        raise ValueError(
            "confidence must be between 0 and 1, inclusive.",
        )

    if np.any(n_successes > n_total):
        raise ValueError(
            f"n_successes ({n_successes}) must be less than or equal to"
            f" n_total ({n_total}).",
        )

    # Compute the binomial confidence interval.
    lo = np.where(
        n_successes == 0,
        0.,
        stats.beta(
            n_successes,
            n_total - n_successes + 1,
        ).ppf((1 - confidence)/2),
    )
    hi = np.where(
        n_successes == n_total,
        1.,
        stats.beta(
            n_successes + 1,
            n_total - n_successes,
        ).ppf(1 - (1 - confidence)/2),
    )

    return lo, hi


def normal_pdf(xs):
    """Evaluate the PDF of the standard normal distribution.

    Parameters
    ----------
    xs : float or array of floats, required
        The points at which to evaluate the standard normal
        distribution's probability density function.

    Returns
    -------
    non-negative float or array of floats
        The standard normal distribution's probability density function
        evaluated at ``xs``.
    """
    # NOTE: In a quick benchmark, this implementation was more than
    # 10-80x faster than scipy.stats.norm.pdf, depending on input size.
    xs = np.array(xs)

    return (
        0.3989422804014327  # 1 / (2 * pi)**0.5 (pre-computed for speed)
        * np.exp(-0.5 * xs**2)
    )


def normal_cdf(xs):
    """Evaluate the CDF of the standard normal distribution.

    Parameters
    ----------
    xs : float or array of floats, required
        The points at which to evaluate the standard normal
        distribution's cumulative distribution function.

    Returns
    -------
    float or array of floats from 0 to 1 inclusive
        The standard normal distribution's cumulative distribution
        function evaluated at ``xs``.
    """
    # NOTE: In a quick benchmark, this implementation was more than
    # 5-95x faster than scipy.stats.norm.cdf, depending on input size.
    xs = np.array(xs)

    return 0.5 * (1. + special.erf(1 / 2**0.5 * xs))


def normal_ppf(qs):
    """Evaluate the PPF of the standard normal distribution.

    Parameters
    ----------
    qs : float or array of floats from 0 to 1 inclusive, required
        The points at which to evaluate the standard normal
        distribution's quantile function.

    Returns
    -------
    float or array of floats
        The standard normal distribution's quantile function evaluated
        at ``qs``.
    """
    # NOTE: In a quick benchmark, this implementation was 2-6x faster
    # than scipy.stats.norm.ppf at moderate input sizes (1-1k) and 1.25x
    # slower at very large sizes (100k-1M).

    # Validate the arguments.
    qs = np.array(qs)
    if np.any((qs < 0. - 1e-10) | (qs > 1. + 1e-10)):
        raise ValueError("qs must be between 0 and 1, inclusive.")
    qs = np.clip(qs, 0., 1.)

    # Compute the quantiles.
    return 2**0.5 * special.erfinv(2. * qs - 1.)
