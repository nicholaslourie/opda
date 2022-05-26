"""Nonparametric ERSA."""

import numpy as np
from scipy import special

from ersa import utils


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
        return np.sum(
            (
                self._ws_cumsum[1:-1]**ns[..., None]
                - self._ws_cumsum[:-2]**ns[..., None]
            ) * self._ys[1:-1],
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
