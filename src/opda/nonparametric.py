"""Nonparametric distributions and tools for optimal design analysis."""

import functools
import multiprocessing
import os
import warnings

import numpy as np
from scipy import special, stats

from opda import utils
import opda.random

# helper functions and classes

def _normalize_pmf(ws):
    return ws / np.sum(ws)


def _normalize_cdf(ws_cumsum):
    return ws_cumsum / ws_cumsum[-1]


@functools.cache
def _dkw_band_weights(n, confidence):
    ws_cumsum = np.arange(n + 1) / n
    epsilon = utils.dkw_epsilon(n, confidence)
    return (
        np.clip(ws_cumsum - epsilon, 0., 1.),
        np.clip(ws_cumsum + epsilon, 0., 1.),
    )


@functools.cache
def _ks_band_weights(n, confidence):
    ws_cumsum = np.arange(n + 1) / n
    epsilon = stats.kstwo(n).ppf(confidence)
    return (
        np.clip(ws_cumsum - epsilon, 0., 1.),
        np.clip(ws_cumsum + epsilon, 0., 1.),
    )


@functools.cache
def _ld_band_weights(
        n,
        confidence,
        kind,
        generator,
        *,
        n_trials = 100_000,
        n_jobs = None,
):
    # NOTE: For i.i.d. samples, the i'th order statistic's quantile is
    # beta(i, n + 1 - i) distributed. Thus, an interval covering
    # ``confidence`` probability from the beta(i, n + 1 - i)
    # distribution provides a confidence interval for the CDF at the
    # i'th order statistic. We extend these pointwise confidence
    # intervals until they hold simultaneously. Thus, if the j'th order
    # statistic is the largest order statistic smaller than x, we can
    # bound the CDF at x between the lower bound for the j'th and the
    # upper bound for the j+1'th order statistics' quantiles.
    #
    # Instead of searching for the coverage level that makes the
    # pointwise intervals hold simultaneously, it's faster to interpret
    # the confidence band as a statistical test and simulate the test
    # statistic's distribution in order to find the critical value.
    #
    # The default number of trials strikes a trade off between
    # speed and precision. How close the simulated sample's quantile is
    # to the true critical value determines the precision. To explore
    # the precision, we can examine a confidence interval for the
    # quantile of the sample's order statistics (again, using the beta
    # distribution). For example, see the following code snippet:
    #
    #     >>> import numpy as np
    #     ... from opda.utils import beta_highest_density_interval
    #     ...
    #     ... n_trials = 100_000
    #     ... quantiles = np.array([0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    #     ... ks = (quantiles * n_trials).astype(int)
    #     ... lo, hi = beta_highest_density_interval(
    #     ...     ks, n_trials + 1 - ks, 0.999
    #     ... )
    #     ... rel_err_lo = (1. - hi) / (1. - quantiles) - 1.
    #     ... rel_err_hi = (1. - lo) / (1. - quantiles) - 1.
    #     ... fmt_qnt = lambda q: f"{q: .3f}"
    #     ... fmt_err = lambda e: f"{e:+.3f}"
    #     ... print(
    #     ...     f'99.9% Confidence Interval for Error of (1 - quantile)\n'
    #     ...     f'-----------------------------------------------------\n'
    #     ...     f'quantile     : {", ".join(map(fmt_qnt, quantiles))}\n'
    #     ...     f'rel err (lo) : {", ".join(map(fmt_err, rel_err_lo))}\n'
    #     ...     f'rel err (hi) : {", ".join(map(fmt_err, rel_err_hi))}\n'
    #     ... )
    #     99.9% Confidence Interval for Error of (1 - quantile)
    #     -----------------------------------------------------
    #     quantile     :  0.500,  0.750,  0.900,  0.950,  0.990,  0.999
    #     rel err (lo) : -0.010, -0.018, -0.031, -0.045, -0.100, -0.294
    #     rel err (hi) : +0.010, +0.018, +0.032, +0.046, +0.107, +0.366
    #
    # From the above execution trace, it can be seen that with 100,000
    # samples, ``1 - confidence`` will be within about +/- 10% of the
    # true value, for typical usage (i.e., confidence up to 99%).

    if kind == "equal_tailed":
        interval = utils.beta_equal_tailed_interval
        coverage = utils.beta_equal_tailed_coverage
    elif kind == "highest_density":
        interval = utils.beta_highest_density_interval
        coverage = utils.beta_highest_density_coverage
    else:
        raise ValueError(
            f'kind must be one of "equal_tailed" or "highest_density",'
            f' not {kind}.',
        )

    ns = np.arange(1, n + 1)

    # Compute the critical value of the test statistic.
    us = generator.random((n_trials, n))
    us.sort(kind="quicksort", axis=-1)
    if n_jobs == 1:
        ts = 0.
        for i, (a, b) in enumerate(zip(ns, n + 1 - ns)):
            ts = np.maximum(ts, coverage(a, b, us[:, i]))
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            ts = pool.starmap(
                coverage,
                ((a, b, us[:, i]) for i, (a, b) in enumerate(zip(ns, n+1-ns))),
            )
            ts = np.max(np.stack(ts, axis=1), axis=1)
    critical_value = np.quantile(ts, confidence)

    lo, hi = interval(ns, n + 1 - ns, critical_value)
    # NOTE: If the j'th order statistic is the largest one smaller than
    # x, then the lower bound for the j'th and the upper bound for the
    # j+1'th provide the bounds for the CDF at x.
    lo = np.concatenate([[0.], lo])
    hi = np.concatenate([hi, [1.]])

    return (
        np.clip(lo, 0., 1.),
        np.clip(hi, 0., 1.),
    )


# main functions and classes

class EmpiricalDistribution:
    r"""The empirical distribution.

    Parameters
    ----------
    ys : 1D array of floats, required
        The sample for which to create an empirical distribution.
    ws : 1D array of non-negative floats or None, optional
        Weights, or the probability masses, to assign to each value in
        the sample, ``ys``. Weights must be non-negative and sum to 1.
        ``ws`` should have the same shape as ``ys``. If ``None``, then
        each sample will be assigned equal weight.
    a : float, optional
        The minimum of the support of the underlying distribution.
    b : float, optional
        The maximum of the support of the underlying distribution.

    Attributes
    ----------
    mean : float
        The distribution's mean.
    variance : float
        The distribution's variance.

    Notes
    -----
    :py:class:`EmpiricalDistribution` provides confidence bands for the
    CDF which can then be translated into confidence bands for the
    tuning curve. See the examples section for how to accomplish this
    task or [1]_ for more background.

    References
    ----------
    .. [1] Lourie, Nicholas, Kyunghyun Cho, and He He. "Show Your Work
       with Confidence: Confidence Bands for Tuning Curves." arXiv
       preprint arXiv:2311.09480 (2023).

    Examples
    --------
    To produce confidence bands for tuning curves, first create
    confidence bands for the CDF of the score distribution:

    .. code:: python

        >>> ns = [1, 2, 3, 4, 5]
        >>> lower_cdf, point_cdf, upper_cdf =\
        ...   EmpiricalDistribution.confidence_bands(
        ...     ys=[0.1, 0.8, 0.5, 0.4, 0.6],
        ...     confidence=0.80,
        ...   )
        >>> tuning_curve_lower = upper_cdf.quantile_tuning_curve(ns)
        >>> tuning_curve_point = point_cdf.quantile_tuning_curve(ns)
        >>> tuning_curve_upper = lower_cdf.quantile_tuning_curve(ns)

    Note that the upper CDF band gives the lower tuning curve band and
    vice versa.
    """

    def __init__(
            self,
            ys,
            ws = None,
            a = -np.inf,
            b = np.inf,
    ):
        # Validate the arguments.
        ys = np.array(ys)
        if len(ys.shape) != 1:
            raise ValueError(f"ys must be a 1D array, not {len(ys.shape)}D.")
        if len(ys) == 0:
            raise ValueError("ys must be non-empty.")

        if ws is not None:
            ws = np.array(ws)
            if ws.shape != ys.shape:
                raise ValueError(
                    f"ws must have the same shape as ys: {ys.shape},"
                    f" not {ws.shape}.",
                )
            if np.any(ws < 0):
                raise ValueError("ws must be non-negative.")
            if np.abs(np.sum(ws) - 1.) > 1e-10:
                raise ValueError("ws must sum to 1.")

        a = np.array(a)[()]
        if not np.isscalar(a):
            raise ValueError("a must be a scalar.")
        if a > np.min(ys):
            raise ValueError(
                f"a ({a}) cannot be greater than the min of ys ({np.min(ys)}).",
            )

        b = np.array(b)[()]
        if not np.isscalar(b):
            raise ValueError("b must be a scalar.")
        if b < np.max(ys):
            raise ValueError(
                f"b ({b}) cannot be less than the max of ys ({np.max(ys)}).",
            )

        # Bind arguments to the instance as attributes.
        self.ys = ys
        self.ws = ws
        self.a = a
        self.b = b

        # Bind other attributes to the instance.
        with np.errstate(invalid="ignore"):
            self.mean = (
                np.mean(ys)
                if ws is None else
                np.sum(ws * ys, where=ws > 0.)
            )
            self.variance = (
                np.var(ys)
                if ws is None else
                np.sum(ws * (ys - self.mean)**2, where=ws > 0.)
            )

        # Handle duplicate values in ys and default value for ws.
        _ys, locations, counts = np.unique(
            self.ys,
            return_inverse=True,
            return_counts=True,
        )
        equal_weights = _normalize_pmf(counts)
        if self.ws is None:
            _ws = equal_weights
        elif np.array_equal(self.ws, equal_weights):
            warnings.warn(
                "ws gives equal weight to each sample. Setting ws"
                " to None, which gives equivalent behavior. Consider"
                " passing None for ws instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            # NOTE: It is important to normalize the representation so
            # that equality checks work and equal instances have equal
            # representations when printed with the __repr__ method.
            self.ws = ws = None
            _ws = equal_weights
        else:
            _ws = _normalize_pmf(np.bincount(locations, weights=self.ws))

        # Handle bounds on the distribution's support.
        #   lower bound
        prepend = []
        if -np.inf != self.a:
            prepend.append(-np.inf)
        if self.a not in _ys:
            prepend.append(self.a)
        #   upper bound
        postpend = []
        if self.b not in _ys:
            postpend.append(self.b)
        if np.inf != self.b:
            postpend.append(np.inf)

        self._ys, self._ws = utils.sort_by_first(
            np.concatenate([prepend, _ys, postpend]),
            np.concatenate([[0.] * len(prepend), _ws, [0.] * len(postpend)]),
        )

        # Initialize useful private attributes.
        self._ws_cumsum = _normalize_cdf(np.cumsum(self._ws))
        self._ws_cumsum_prev = np.concatenate([[0.], self._ws_cumsum[:-1]])

        self._has_ws = self.ws is not None

        self._n = len(self.ys)
        self._ns = np.arange(1, self._n + 1)

        self._original_ys_cummin = np.minimum.accumulate(self.ys)
        self._original_ys_cummax = np.maximum.accumulate(self.ys)

        self._original_ys_sorted = np.sort(self.ys)
        self._original_ys_reverse_sorted = self._original_ys_sorted[::-1]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                np.array_equal(self.ys, other.ys)
                and np.array_equal(self.ws, other.ws)
                and self.a == other.a
                and self.b == other.b
            )
        return NotImplemented

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"ys={self.ys!s},"
            f" ws={self.ws!s},"
            f" a={self.a!s},"
            f" b={self.b!s}"
            f")"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ys={self.ys!r},"
            f" ws={self.ws!r},"
            f" a={self.a!r},"
            f" b={self.b!r}"
            f")"
        )

    def sample(self, size=None, *, generator=None):
        """Return a sample from the empirical distribution.

        Parameters
        ----------
        size : None, int, or tuple of ints, optional
            The desired shape of the returned sample. If ``None``,
            then the sample is a scalar.
        generator : np.random.Generator or None, optional
            The random number generator to use. If ``None``, then the
            global default random number generator is used. See
            :py:mod:`opda.random` for more information.

        Returns
        -------
        float or array of floats
            The sample from the distribution.
        """
        # Validate arguments.
        generator = (
            generator
            if generator is not None else
            opda.random.DEFAULT_GENERATOR
        )

        # Compute the sample.
        return generator.choice(
            self.ys,
            p=self.ws,
            size=size,
        )

    def pmf(self, ys):
        """Return the probability mass at ``ys``.

        Parameters
        ----------
        ys : float or array of floats, required
            The points at which to evaluate the probability mass.

        Returns
        -------
        float or array of floats from 0 to 1 inclusive
            The probability mass at ``ys``.
        """
        indices = np.searchsorted(self._ys, ys, side="left")
        return np.where(
            self._ys[indices] == ys,
            self._ws[indices],
            0,
        )[()]

    def cdf(self, ys):
        r"""Return the cumulative probability at ``ys``.

        We define the cumulative distribution function, :math:`F`, using
        less than or equal to:

        .. math::

           F(y) = \mathbb{P}(Y \leq y)

        Parameters
        ----------
        ys : float or array of floats, required
            The points at which to evaluate the cumulative probability.

        Returns
        -------
        float or array of floats from 0 to 1 inclusive
            The cumulative probability at ``ys``.
        """
        indices = np.searchsorted(self._ys, ys, side="right") - 1
        return self._ws_cumsum[indices]

    def ppf(self, qs):
        r"""Return the quantile at ``qs``.

        Since the empirical distribution is discrete, its exact
        quantiles are ambiguous. We use the following definition of
        the quantile function, :math:`Q`:

        .. math::

           Q(p) = \inf \{y\in[a, b]\mid p\leq F(y)\}

        where :math:`F` is the cumulative distribution function and
        :math:`a` and :math:`b` are the optional bounds provided for the
        distribution's support. Note that this definition is different
        from the most standard one in which :math:`y` is quantified over
        the whole real line; however, quantifying over the reals leads
        to counterintuitive behavior at zero, which then always
        evaluates to negative infinity. Instead, the above definition
        will have zero evaluate to the lower bound on the support.

        Parameters
        ----------
        qs : float or array of floats from 0 to 1 inclusive, required
            The points at which to evaluate the quantiles.

        Returns
        -------
        float or array of floats
            The quantiles at ``qs``.
        """
        # Validate the arguments.
        qs = np.array(qs)
        if np.any((qs < 0. - 1e-10) | (qs > 1. + 1e-10)):
            raise ValueError("qs must be between 0 and 1, inclusive.")
        qs = np.clip(qs, 0., 1.)

        # Compute the quantiles.
        return np.maximum(
            self._ys[np.argmax(qs[..., None] <= self._ws_cumsum, axis=-1)],
            self.a,
        )

    def quantile_tuning_curve(self, ns, q=0.5, minimize=False):
        """Return the quantile tuning curve evaluated at ``ns``.

        Since the empirical distribution is discrete, its exact
        quantiles are ambiguous. See the :py:meth:`ppf` method for the
        definition of the quantile function we use.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        q : float from 0 to 1 inclusive, optional
            The quantile at which to evaluate the tuning curve.
        minimize : bool, optional
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it.

        Returns
        -------
        float or array of floats
            The quantile tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        if q < 0. or q > 1.:
            raise ValueError("q must be between 0 and 1, inclusive.")

        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the quantile tuning curve.
        return self.ppf(
            1 - (1 - q)**(1/ns)
            if minimize else  # maximize
            q**(1/ns),
        )

    def average_tuning_curve(self, ns, minimize=False):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        minimize : bool, optional
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it.

        Returns
        -------
        float or array of floats
            The average tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the average tuning curve.
        if minimize:
            ws = (
                (1 - self._ws_cumsum_prev)**ns[..., None]
                - (1 - self._ws_cumsum)**ns[..., None]
            )
        else:  # maximize
            ws = (
                self._ws_cumsum**ns[..., None]
                - self._ws_cumsum_prev**ns[..., None]
            )

        return np.sum(
            ws * np.where(ws != 0., self._ys, 0.),
            axis=-1,
        )

    def naive_tuning_curve(self, ns, minimize=False):
        """Return the naive estimate for the tuning curve at ``ns``.

        The naive tuning curve estimate assigns to n the maximum value
        seen in the first n samples. The estimate assumes each sample
        has identical weight, so this method cannot be called when
        ``ws`` is not ``None``.

        Parameters
        ----------
        ns : positive int or array of ints, required
            The values at which to evaluate the naive tuning curve
            estimate.
        minimize : bool, optional
            Whether or not to estimate the tuning curve for minimizing a
            metric as opposed to maximizing it.

        Returns
        -------
        float or array of floats
            The values of the naive tuning curve estimate.
        """
        # Validate the instance and arguments.
        if self._has_ws:
            raise ValueError(
                "naive_tuning_curve cannot be called when ws is not None.",
            )

        ns = np.array(ns)
        if not np.all(ns % 1 == 0):
            raise ValueError("ns must only contain integers.")
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")
        ns = ns.astype(int)

        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the naive tuning curve estimate.
        ns = np.clip(ns, None, self._n)

        return (
            self._original_ys_cummin
            if minimize else  # maximize
            self._original_ys_cummax
        )[ns - 1]

    def v_tuning_curve(self, ns, minimize=False):
        """Return the v estimate for the tuning curve at ``ns``.

        The v statistic tuning curve estimate assigns to n the average
        value of the maximum after n observations when resampling with
        replacement. The estimate is consistent but biased.

        Parameters
        ----------
        ns : positive int or array of ints, required
            The values at which to evaluate the v statistic tuning curve
            estimate.
        minimize : bool, optional
            Whether or not to estimate the tuning curve for minimizing a
            metric as opposed to maximizing it.

        Returns
        -------
        float or array of floats
            The values of the v statistic tuning curve estimate.
        """
        # Validate the instance and arguments.
        if self._has_ws:
            raise ValueError(
                "v_tuning_curve cannot be called when ws is not None.",
            )

        ns = np.array(ns)
        if not np.all(ns % 1 == 0):
            raise ValueError("ns must only contain integers.")
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")
        ns = ns.astype(int)

        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the v statistic tuning curve estimate.
        return np.sum(
            (
                (self._ns / self._n)**ns[..., None]
                - ((self._ns - 1) / self._n)**ns[..., None]
            ) * (
                self._original_ys_reverse_sorted
                if minimize else  # maximize
                self._original_ys_sorted
            ),
            axis=-1,
        )

    def u_tuning_curve(self, ns, minimize=False):
        """Return the u estimate for the tuning curve at ``ns``.

        The u statistic tuning curve estimate assigns to n the average
        value of the maximum after n observations when resampling
        without replacement. The estimate is unbiased for n less than or
        equal to the original sample size. For larger n, we return the
        maximum value from the original sample.

        Parameters
        ----------
        ns : positive int or array of ints, required
            The values at which to evaluate the u statistic tuning curve
            estimate.
        minimize : bool, optional
            Whether or not to estimate the tuning curve for minimizing a
            metric as opposed to maximizing it.

        Returns
        -------
        float or array of floats
            The values of the u statistic tuning curve estimate.
        """
        # Validate the instance and arguments.
        if self._has_ws:
            raise ValueError(
                "u_tuning_curve cannot be called when ws is not None.",
            )

        ns = np.array(ns)
        if not np.all(ns % 1 == 0):
            raise ValueError("ns must only contain integers.")
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")
        ns = ns.astype(int)

        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the u statistic tuning curve estimate.
        ns = np.clip(ns, None, self._n)

        return np.sum(
            (
                special.comb(self._ns, ns[..., None])
                - special.comb(self._ns - 1, ns[..., None])
            ) / special.comb(self._n, ns[..., None])
            * (
                self._original_ys_reverse_sorted
                if minimize else  # maximize
                self._original_ys_sorted
            ),
            axis=-1,
        )

    @classmethod
    def confidence_bands(
            cls,
            ys,
            confidence,
            a = -np.inf,
            b = np.inf,
            *,
            generator = None,
            method = "ld_highest_density",
            n_jobs = None,
    ):
        """Return confidence bands for the CDF.

        Return three instances of :py:class:`EmpiricalDistribution`,
        offering a lower confidence band, point estimate, and upper
        confidence band for the CDF of the distribution that generated
        ``ys``.

        The properties of the CDF bands depend on the method used to
        construct them, as set by the ``method`` parameter.

        Parameters
        ----------
        ys : 1D array of floats, required
            The sample from the distribution.
        confidence : float from 0 to 1 inclusive, required
            The coverage or confidence level for the bands.
        a : float, optional
            The minimum of the support of the underlying distribution.
        b : float, optional
            The maximum of the support of the underlying distribution.
        generator : np.random.Generator or None, optional
            The random number generator to use. If ``None``, then the
            global default random number generator is used. See
            :py:mod:`opda.random` for more information.
        method : str, optional
            One of the strings: "dkw", "ks", "ld_equal_tailed", or
            "ld_highest_density". The ``method`` parameter determines
            the kind of confidence band and thus its properties. See
            the notes section for details on the different methods.
        n_jobs : positive int or None, optional
            Set the maximum number of parallel processes to use when
            constructing the confidence bands. If ``None`` then
            ``n_jobs`` will be set to the number of CPUs returned by
            :py:func:`os.cpu_count`. Only some methods (e.g.
            ``"ld_highest_density"``) can leverage parallel
            computation. If the method can't use parallelism, it'll
            just use the current process instead.

        Returns
        -------
        EmpiricalDistribution
            The lower confidence band for the distribution's CDF.
        EmpiricalDistribution
            The point estimate for the distribution's CDF.
        EmpiricalDistribution
            The upper confidence band for the distribution's CDF.

        Notes
        -----
        There are four built-in methods for generating confidence bands:
        dkw, ks, ld_equal_tailed, and ld_highest_density. All four
        methods provide simultaneous confidence bands.

        The dkw method uses the Dvoretzky-Kiefer-Wolfowitz inequality
        which is fast to compute but fairly conservative for smaller
        samples.

        The ks method inverts the Kolmogorov-Smirnov test to provide a
        confidence band with exact coverage and which is uniformly
        spaced above and below the empirical cumulative
        distribution. Because the band has uniform width, it is
        relatively looser at the ends than in the middle, and most
        violations of the confidence band tend to occur near the
        median. The Kolmogorov-Smirnov bands require that the underlying
        distribution is continuous to achieve exact coverage.

        The ld (Learned-Miller-DeStefano) methods expand pointwise
        confidence bands for the order statistics, based on the beta
        distribution, until they hold simultaneously with exact
        coverage. These pointwise bands may either use the equal-tailed
        interval (ld_equal_tailed) or the highest density interval
        (ld_highest_density) from the beta distribution. The highest
        density interval yields the tightest bands; however, the
        equal-tailed intervals are almost the same size and
        significantly faster to compute. The ld bands do not have
        uniform width and are tighter near the end points. They're
        violated equally often across the whole range. The
        Learned-Miller-DeStefano bands require that the underlying
        distribution is continuous to achieve exact coverage. See "A
        Probabilistic Upper Bound on Differential Entropy" [1]_ for
        details.

        References
        ----------
        .. [1] Learned-Miller, E and DeStefano, J, "A Probabilistic
           Upper Bound on Differential Entropy" (2008). IEEE
           TRANSACTIONS ON INFORMATION THEORY. 732.
        """
        # Validate arguments and handle defaults.
        ys = np.array(ys)
        if len(ys.shape) != 1:
            raise ValueError(f"ys must be a 1D array, not {len(ys.shape)}D.")
        if len(ys) == 0:
            raise ValueError("ys must be non-empty.")
        if (
                len(np.unique(ys)) != len(ys)
                and method in ["ks", "ld_equal_tailed", "ld_highest_density"]
        ):
            warnings.warn(
                "Duplicates detected in ys. confidence_bands with the"
                " ks, ld_equal_tailed, or ld_highest_density methods"
                " requires the underlying distribution to be continuous"
                " in order to achieve exact coverage.",
                RuntimeWarning,
                stacklevel=2,
            )

        confidence = np.array(confidence)[()]
        if not np.isscalar(confidence):
            raise ValueError("confidence must be a scalar.")
        if confidence < 0. or confidence > 1.:
            raise ValueError(
                "confidence must be between 0 and 1, inclusive.",
            )

        a = np.array(a)[()]
        if not np.isscalar(a):
            raise ValueError("a must be a scalar.")
        if a > np.min(ys):
            raise ValueError(
                f"a ({a}) cannot be greater than the min of ys ({np.min(ys)}).",
            )

        b = np.array(b)[()]
        if not np.isscalar(b):
            raise ValueError("b must be a scalar.")
        if b < np.max(ys):
            raise ValueError(
                f"b ({b}) cannot be less than the max of ys ({np.max(ys)}).",
            )

        generator = (
            generator
            if generator is not None else
            opda.random.DEFAULT_GENERATOR
        )

        n_jobs = (
            n_jobs
            if n_jobs is not None else
            os.cpu_count()
        )
        if n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer.")

        # Compute the confidence bands.
        n = len(ys)
        ys_extended = np.concatenate([[a], ys, [b]])
        unsorting = np.argsort(np.argsort(ys_extended))

        if method == "dkw":
            ws_lo_cumsum, ws_hi_cumsum = _dkw_band_weights(
                n, confidence,
            )
        elif method == "ks":
            ws_lo_cumsum, ws_hi_cumsum = _ks_band_weights(
                n, confidence,
            )
        elif method == "ld_equal_tailed":
            ws_lo_cumsum, ws_hi_cumsum = _ld_band_weights(
                n,
                confidence,
                kind="equal_tailed",
                generator=generator,
                n_jobs=n_jobs,
            )
        elif method == "ld_highest_density":
            ws_lo_cumsum, ws_hi_cumsum = _ld_band_weights(
                n,
                confidence,
                kind="highest_density",
                generator=generator,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(
                'method must be one of "dkw", "ks", "ld_equal_tailed",'
                ' or "ld_highest_density".',
            )

        ws_lo = np.diff(ws_lo_cumsum, prepend=[0.], append=[1.])[unsorting]
        ws_hi = np.diff(ws_hi_cumsum, prepend=[0.], append=[1.])[unsorting]

        return (
            cls(ys_extended, ws=ws_lo, a=a, b=b),
            cls(ys, ws=None, a=a, b=b),
            cls(ys_extended, ws=ws_hi, a=a, b=b),
        )
