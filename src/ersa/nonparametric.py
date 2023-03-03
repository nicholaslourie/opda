"""Nonparametric ERSA."""

import functools

import numpy as np
from scipy import special, stats

from ersa import utils


# helper functions and classes

@functools.cache
def _dkw_band_weights(n, confidence):
    ws_cumsum = (1. + np.arange(n)) / n
    epsilon = utils.dkw_epsilon(n, confidence)
    return (
        np.clip(ws_cumsum - epsilon, 0., 1.),
        np.clip(ws_cumsum + epsilon, 0., 1.),
    )


@functools.cache
def _ks_band_weights(n, confidence):
    ws_cumsum = (1. + np.arange(n)) / n
    epsilon = stats.kstwo(n).ppf(confidence)
    return (
        np.clip(ws_cumsum - epsilon, 0., 1.),
        np.clip(ws_cumsum + epsilon, 0., 1.),
    )


@functools.cache
def _beta_band_weights(n, confidence, kind, n_trials=100_000):
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
    #     ... from ersa.utils import beta_hpd_interval
    #     ...
    #     ... n_trials = 100_000
    #     ... quantiles = np.array([0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
    #     ... ks = (quantiles * n_trials).astype(int)
    #     ... lo, hi = beta_hpd_interval(ks, n_trials + 1 - ks, 0.999)
    #     ... rel_err_lo = (1. - hi) / (1. - quantiles) - 1.
    #     ... rel_err_hi = (1. - lo) / (1. - quantiles) - 1.
    #     ... fmt_qnt = lambda q: f'{q: .3f}'
    #     ... fmt_err = lambda e: f'{e:+.3f}'
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

    if kind == 'equal_tailed':
        interval = utils.beta_equal_tailed_interval
        coverage = utils.beta_equal_tailed_coverage
    elif kind == 'hpd':
        interval = utils.beta_hpd_interval
        coverage = utils.beta_hpd_coverage
    else:
        raise ValueError(
            f'kind must be one of "equal_tailed" or "hpd",'
            f' not {kind}.'
        )

    ns = np.arange(1, n + 1)
    a = ns
    b = n + 1 - ns

    # Compute the critical value of the test statistic.
    ts = np.random.rand(n_trials, n)
    ts.sort(kind='quicksort', axis=-1)
    ts = np.max(coverage(a, b, ts), axis=-1)
    critical_value = np.quantile(ts, confidence)

    lo, hi = interval(a, b, critical_value)
    # NOTE: If the j'th order statistic is the largest one smaller than
    # x, then the lower bound for the j'th and the upper bound for the
    # j+1'th provide the bounds for the CDF at x.
    hi = np.concatenate([hi[1:], [1.]])

    return (
        np.clip(lo, 0., 1.),
        np.clip(hi, 0., 1.),
    )


# main functions and classes

class EmpiricalDistribution:
    """The empirical distribution.

    Parameters
    ----------
    ys : 1D array of floats, required
        The sample for which to create an empirical distribution. Each
        value in the sample should be unique. Use the ``ws`` argument to
        handle duplicate values.
    ws : 1D array of floats or None, optional (default=None)
        Weights, or the probability masses, to assign to each value in
        the sample. Weights must be non-negative and sum to 1. ``ws``
        should have the same shape as ``ys``. If ``None``, then each
        sample will be assigned equal weight.
    a : float or None, optional (default=None)
        The minimum of the support of the underlying distribution. If
        ``None``, then it will be set to ``-np.inf``.
    b : float or None, optional (default=None)
        The maximum of the support of the underlying distribution. If
        ``None``, then it will be set to ``np.inf``.
    """
    def __init__(
            self,
            ys,
            *,
            ws = None,
            a = None,
            b = None,
    ):
        # Validate the arguments.
        ys = np.array(ys)
        if len(ys.shape) != 1:
            raise ValueError(f'ys must be a 1D array, not {len(ys.shape)}D.')
        if len(ys) == 0:
            raise ValueError('ys must be non-empty.')
        if len(np.unique(ys)) != len(ys):
            raise ValueError(
                'The values of ys must be unique.'
                ' Use the ws argument to handle repeats.'
            )

        if ws is not None:
            ws = np.array(ws)
            if ws.shape != ys.shape:
                raise ValueError(
                    f'ws must have the same shape as ys: {ys.shape},'
                    f' not {ws.shape}.'
                )
            if np.any(ws < 0):
                raise ValueError('ws must be non-negative.')
            if np.abs(np.sum(ws) - 1.) > 1e-10:
                raise ValueError('ws must sum to 1.')

        if a is not None and a > np.min(ys):
            raise ValueError(
                f'a ({a}) cannot be greater than the min of ys ({np.min(ys)}).'
            )

        if b is not None and b < np.max(ys):
            raise ValueError(
                f'b ({b}) cannot be less than the max of ys ({np.max(ys)}).'
            )

        # Bind arguments to attributes.
        self.ys = ys
        self.ws = (
            ws / np.sum(ws)
            if ws is not None else
            np.ones_like(ys, dtype=float)
            / np.sum(np.ones_like(ys, dtype=float))
        )
        self.a = a if a is not None else -np.inf
        self.b = b if b is not None else np.inf

        # Initialize other useful attributes.
        self._has_ws = ws is not None

        self._n = len(self.ys)
        self._ns = np.arange(1, self._n + 1)

        self._original_ys_cummax = np.maximum.accumulate(self.ys)
        self._original_ys_sorted = np.sort(self.ys)

        prepend = []
        if -np.inf != self.a:
            prepend.append(-np.inf)
        if self.a not in self.ys:
            prepend.append(self.a)
        postpend = []
        if self.b not in self.ys:
            postpend.append(self.b)
        if np.inf != self.b:
            postpend.append(np.inf)
        self._ys, self._ws = utils.sort_by_first(
            np.concatenate([prepend, self.ys, postpend]),
            np.concatenate([[0.] * len(prepend), self.ws, [0.] * len(postpend)]),
        )
        self._ws_cumsum = np.cumsum(self._ws)
        self._ws_cumsum_prev = np.concatenate([[0.], self._ws_cumsum[:-1]])

    def sample(self, size):
        """Return a sample from the empirical distribution.

        Parameters
        ----------
        size : int or tuple of ints, required
            The desired shape of the returned sample.

        Returns
        -------
        array of floats
            The sample from the distribution.
        """
        return np.random.choice(
            self.ys,
            p=self.ws,
            size=size,
        )

    def pmf(self, ys):
        """Return the probability mass at ``ys``.

        Parameters
        ----------
        ys : array of floats, required
            The points at which to evaluate the probability mass.

        Returns
        -------
        array of floats
            The probability mass at ``ys``.
        """
        indices = np.searchsorted(self._ys, ys, side='left'),
        return np.where(
            self._ys[indices] == ys,
            self._ws[indices],
            0,
        )

    def cdf(self, ys):
        """Return the cumulative probability at ``ys``.

        We define the cumulative distribution function, F, using less
        than or equal to:

        .. math::

           F(y) = \\mathbb{P}(Y \\leq y)


        Parameters
        ----------
        ys : array of float, required
            The points at which to evaluate the cumulative probability.

        Returns
        -------
        array of floats
            The cumulative probability at ``ys``.
        """
        indices = np.searchsorted(self._ys, ys, side='right') - 1
        return self._ws_cumsum[indices]

    def ppf(self, qs):
        """Return the quantile at ``qs``.

        Since the empirical distribution is discrete, its exact
        quantiles are ambiguous. We use the following common definition
        of the quantile function, Q:

        .. math::

           Q(p) = \\inf \\{y\\in\\mathbb{R}\\mid p\\leq F(y)\\}

        where F is the cumulative distribution function.

        Parameters
        ----------
        qs : array of floats, required
            The points at which to evaluate the quantiles.

        Returns
        -------
        array of floats
            The quantiles at ``qs``.
        """
        # Validate the arguments.
        qs = np.array(qs)
        if np.any((qs < 0. - 1e-10) | (qs > 1. + 1e-10)):
            raise ValueError('qs must be between 0 and 1, inclusive.')

        # Compute the quantiles.
        qs = np.clip(qs, 0., 1.)

        return self._ys[np.argmax(qs[..., None] <= self._ws_cumsum, axis=-1)]

    def quantile_tuning_curve(self, ns, q=0.5):
        """Return the quantile tuning curve evaluated at ``ns``.

        Since the empirical distribution is discrete, its exact
        quantiles are ambiguous. See the ``ppf`` method for the
        definition of the quantile function we use.

        Parameters
        ----------
        ns : array of ints, required
            The integers at which to evaluate the tuning curve.
        q : float, optional (default=0.5)
            The quantile at which to evaluate the tuning curve.

        Returns
        -------
        array of floats
            The quantile tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError('ns must be positive.')

        if q < 0. or 1. < q:
            raise ValueError(
                f'q must be between 0 and 1 inclusive, not {q}.'
            )

        # Compute the quantile tuning curve.
        return self.ppf(q**(1/ns))

    def average_tuning_curve(self, ns):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : array of ints, required
            The integers at which to evaluate the tuning curve.

        Returns
        -------
        array of floats
            The average tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError('ns must be positive.')

        # Compute the average tuning curve.
        ws = (
            self._ws_cumsum**ns[..., None]
            - self._ws_cumsum_prev**ns[..., None]
        )
        return np.sum(
            ws * np.where(ws != 0., self._ys, 0.),
            axis=-1,
        )

    def naive_tuning_curve(self, ns):
        """Return the naive estimate for the tuning curve at ``ns``.

        The naive tuning curve estimate assigns to n the maximum value
        seen in the first n samples. The estimate assumes each sample
        has identical weight, so this method cannot be called when
        ``ws`` is not ``None``.

        Parameters
        ----------
        ns : array of ints, required
            The values at which to evaluate the naive tuning curve. The
            values must be positive integers.

        Returns
        -------
        array of floats
            The values of the naive tuning curve.
        """
        # Validate the instance and arguments.
        if self._has_ws:
            raise ValueError(
                'naive_tuning_curve cannot be called when ws is not None.'
            )

        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError('ns must be positive.')

        # Compute the naive tuning curve estimate.
        ns = np.clip(ns, None, self._n)

        return self._original_ys_cummax[ns - 1]

    def v_tuning_curve(self, ns):
        """Return the v estimate for the tuning curve at ``ns``.

        The v statistic tuning curve estimate assigns to n the average
        value of the maximum after n observations when resampling with
        replacement. The estimate is consistent but biased.

        Parameters
        ----------
        ns : array of ints, required
            The values at which to evaluate the v tuning curve. The
            values must be positive integers.

        Returns
        -------
        array of floats
            The values of the v tuning curve estimate.
        """
        # Validate the instance and arguments.
        if self._has_ws:
            raise ValueError(
                'v_tuning_curve cannot be called when ws is not None.'
            )

        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError('ns must be positive.')

        # Compute the v statistic tuning curve estimate.
        return np.sum(
            (
                (self._ns / self._n)**ns[..., None]
                - ((self._ns - 1) / self._n)**ns[..., None]
            ) * self._original_ys_sorted,
            axis=-1,
        )

    def u_tuning_curve(self, ns):
        """Return the u estimate for the tuning curve at ``ns``.

        The u statistic tuning curve estimate assigns to n the average
        value of the maximum after n observations when resampling
        without replacement. The estimate is unbiased for n less than or
        equal to the original sample size. For larger n, we return the
        maximum value from the original sample.

        Parameters
        ----------
        ns : array of ints, required
            The values at which to evaluate the u tuning curve. The
            values must be positive integers.

        Returns
        -------
        array of floats
            The values of the u tuning curve estimate.
        """
        # Validate the instance and arguments.
        if self._has_ws:
            raise ValueError(
                'u_tuning_curve cannot be called when ws is not None.'
            )

        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError('ns must be positive.')

        # Compute the u statistic tuning curve estimate.
        ns = np.clip(ns, None, self._n)

        return np.sum(
            (
                special.comb(self._ns, ns[..., None])
                - special.comb(self._ns - 1, ns[..., None])
            ) / special.comb(self._n, ns[..., None])
            * self._original_ys_sorted,
            axis=-1,
        )

    @classmethod
    def confidence_bands(
            cls,
            ys,
            confidence,
            method = 'beta_equal_tailed',
            *,
            a = None,
            b = None,
    ):
        """Return confidence bands for the CDF.

        Return three instances of ``EmpiricalDistribution``, offering
        a lower confidence band, point estimate, and upper confidence
        band for the CDF of the distribution that generated ``ys``.

        The properties of the CDF bands depend on the method used to
        construct them, as set by the ``method`` parameter.

        Parameters
        ----------
        ys : 1D array of floats, required
            The sample from the distribution.
        confidence : float, required
            The coverage or confidence level for the bands.
        method : str, optional (default='beta_equal_tailed')
            One of the strings 'dkw', 'ks', 'beta_equal_tailed', or
            'beta_hpd'. The ``method`` parameter determines the kind of
            confidence band and thus its properties. See `Notes`_ for
            details on the different methods.
        a : float or None, optional (default=None)
            The minimum of the support of the underlying distribution.
            If ``None``, then it will be set to ``-np.inf``.
        b : float or None, optional (default=None)
            The maximum of the support of the underlying distribution.
            If ``None``, then it will be set to ``np.inf``.

        Returns
        -------
        EmpiricalDistribution, EmpiricalDistribution, EmpiricalDistribution
            A lower confidence band, point estimate, and upper
            confidence band for the distribution's CDF.

        Notes
        -----
        There are four built-in methods for generating confidence bands:
        dkw, ks, beta_equal_tailed, and beta_hpd. All three methods provide
        simultaneous confidence bands.

        The dkw method uses the Dvoretzky-Kiefer-Wolfowitz inequality
        which is fast to compute but overly conservative.

        The ks method inverts the Kolmogorov-Smirnov test to provide a
        confidence band with exact coverage and which is uniformly
        spaced above and below the empirical cumulative
        distribution. Because the band has uniform width, it is much
        looser at the ends than in the middle, and most violations of
        the confidence band tend to occur near the median.

        The beta methods expand pointwise confidence bands for the order
        statistics, based on the beta distribution, until they hold
        simultaneously with exact coverage. These pointwise bands may
        either use the equal-tailed interval (beta_equal_tailed) or the highest
        density interval (beta_hpd) from the beta distribution. The
        highest density interval yields the tightest bands; however, the
        equal-tailed intervals are almost the same size and
        significantly faster to compute. The beta bands do not have
        uniform width and are tighter near the end points. They're
        violated equally often across the whole range. See "A
        Probabilistic Upper Bound on Differential Entropy"
        (Learned-Miller and DeStefano, 2008) for details.

        References
        ----------
        Learned-Miller, E and DeStefano, J, "A Probabilistic Upper
        Bound on Differential Entropy" (2008). IEEE TRANSACTIONS ON
        INFORMATION THEORY. 732.
        """
        a = a if a is not None else -np.inf
        b = b if b is not None else np.inf

        n = len(ys)
        ys_extended = np.concatenate([[a], ys, [b]])
        unsorting = np.argsort(np.argsort(ys_extended))

        if method == 'dkw':
            ws_lo_cumsum, ws_hi_cumsum = _dkw_band_weights(
                n, confidence,
            )
        elif method == 'ks':
            ws_lo_cumsum, ws_hi_cumsum = _ks_band_weights(
                n, confidence,
            )
        elif method == 'beta_equal_tailed':
            ws_lo_cumsum, ws_hi_cumsum = _beta_band_weights(
                n, confidence, kind='equal_tailed',
            )
        elif method == 'beta_hpd':
            ws_lo_cumsum, ws_hi_cumsum = _beta_band_weights(
                n, confidence, kind='hpd',
            )
        else:
            raise ValueError(
                'method must be one of "dkw", "ks", "beta_equal_tailed",'
                ' or "beta_hpd".'
            )

        ws_lo = np.diff(
            np.concatenate([[0.], ws_lo_cumsum, [1.]]),
            prepend=[0.],
        )[unsorting]
        ws_hi = np.diff(
            np.concatenate([[ws_hi_cumsum[0]], ws_hi_cumsum, [1.]]),
            prepend=[0.],
        )[unsorting]

        return (
            cls(ys_extended, ws=ws_lo, a=a, b=b),
            cls(ys, ws=None, a=a, b=b),
            cls(ys_extended, ws=ws_hi, a=a, b=b),
        )
