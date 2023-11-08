"""Parametric OPDA."""

import numpy as np
from scipy import special


class QuadraticDistribution:
    """The Quadratic distribution.

    When using random search to optimize a deterministic smooth
    function, the best score asymptotically approaches a quadratic
    distribution.

    Parameters
    ----------
    a : float, required
        The minimum value that the distribution can take.
    b : float, required
        The maximum value that the distribution can take.
    c : float, required
        Half the dimension of distribution's search space.
    convex : bool, optional (default=False)
        Whether or not to use the convex form of the quadratic
        distribution. The convex form should be used for minimization
        while the concave form should be used for maximization.
    """
    def __init__(
            self,
            a,
            b,
            c,
            *,
            convex=False,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.convex = convex

    def sample(self, size):
        """Return a sample from the quadratic distribution.

        Parameters
        ----------
        size : int or tuple of ints, required
            The desired shape of the returned sample.

        Returns
        -------
        array of floats
            The sample from the distribution.
        """
        return self.ppf(np.random.uniform(0, 1, size=size))

    def pdf(self, ys):
        """Return the probability density at ``ys``.

        Parameters
        ----------
        ys : array of floats, required
            The points at which to evaluate the probability density.

        Returns
        -------
        array of floats
            The probability density at ``ys``.
        """
        ys = np.array(ys)

        a, b, c = self.a, self.b, self.c

        if self.convex:
            ps = (c / (b - a)) * ((ys - a) / (b - a))**(c - 1)
        else:  # concave
            ps = (c / (b - a)) * ((b - ys) / (b - a))**(c - 1)

        ps = np.where((ys < a) | (ys > b), 0., ps)

        return ps

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
        ys = np.array(ys)

        a, b, c = self.a, self.b, self.c

        if self.convex:
            qs = ((ys - a) / (b - a))**c
        else:  # concave
            qs = 1 - ((b - ys) / (b - a))**c

        qs = np.where(ys <= a, 0., qs)
        qs = np.where(ys >= b, 1., qs)

        return qs

    def ppf(self, qs):
        """Return the quantile at ``qs``.

        We define the quantile function, Q, as:

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
        a, b, c = self.a, self.b, self.c

        if self.convex:
            ys = a + (b - a) * qs**(1/c)
        else:  # concave
            ys = b - (b - a) * (1 - qs)**(1/c)

        ys = np.where(qs >= 1, b, ys)
        ys = np.where(qs <= 0, a, ys)

        return ys

    def quantile_tuning_curve(self, ns, q=0.5):
        """Return the quantile tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : array of positive floats, required
            The points at which to evaluate the tuning curve.
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

        a, b, c = self.a, self.b, self.c

        if self.convex:
            ys = a + (b - a) * qs**(1/(c * ns))
        else:  # concave
            ys = b - (b - a) * (1 - q**(1/ns))**(1/c)

        return ys

    def average_tuning_curve(self, ns):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : array of positive floats, required
            The points at which to evaluate the tuning curve.

        Returns
        -------
        array of floats
            The average tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError('ns must be positive.')

        a, b, c = self.a, self.b, self.c

        if self.convex:
            ys = a + (b - a) * np.exp(
                special.loggamma(ns + 1)
                + special.loggamma((c + 1) / c)
                - special.loggamma(ns + (c + 1) / c)
            )
        else:  # concave
            ys = b - (b - a) * np.exp(
                special.loggamma(ns + 1)
                + special.loggamma((c + 1) / c)
                - special.loggamma(ns + (c + 1) / c)
            )

        return ys

    @classmethod
    def estimate_initial_parameters_and_bounds(
            cls,
            ys,
            fraction=1.,
            convex=False,
    ):
        """Return initial parameter estimates and bounds.

        Use the initial parameter estimates and bounds returned by
        this method to initialize parameters for optimization based
        estimates like MLE.

        Parameters
        ----------
        ys : 1D array of floats, required
            The sample from which to estimate the initial parameters and
            bounds.
        fraction : float (0. <= fraction <= 1.), optional (default=1.)
            The fraction of the sample to use. If ``convex`` is
            ``False``, the greatest ``fraction`` numbers are
            retained. If ``convex`` is ``True``, the least ``fraction``
            numbers are retained.
        convex : bool, optional (default=False)
            Whether to estimate initial parameters and bounds for the
            convex or concave form of the distribution.

        Returns
        -------
        3 array of floats, 3 x 2 array of floats
            An array of initial parameter estimates and an array of
            bounds.
        """
        ys_fraction = (
            np.sort(ys)[:int(fraction * len(ys))]
            if convex else
            np.sort(ys)[-int(fraction * len(ys)):]
        )

        a = ys_fraction[0]
        b = ys_fraction[-1]
        # Initialize c with its MLE assuming a and b are known.
        c = (
            1. / np.mean(np.log((b - a) / (ys_fraction[1:-1] - a)))
            if convex else
            1. / np.mean(np.log((b - a) / (b - ys_fraction[1:-1])))
        )

        # Correct a and b for the fact that we're using only a fraction.
        if convex:
            # Push a a bit lower than min(ys).
            a = a - 0.05 * (b - a)
            # Set b so that P(y <= ys_fraction[-1]) = fraction.
            b = a + (b - a) / fraction**(1/c)
            # Push b a little higher.
            b = b + 0.05 * (b - a)
        else:
            # Push b a bit higher than max(ys).
            b = b + 0.05 * (b - a)
            # Set a so that P(y > ys_fraction[0]) = fraction.
            a = b - (b - a) / fraction**(1/c)
            # Push a a little lower.
            a = a - 0.05 * (b - a)

        params = np.array([a, b, c])
        bounds = np.array([
            (-np.inf, ys_fraction[0]),
            (ys_fraction[-1], np.inf),
            (0, np.inf),
        ])

        return params, bounds
