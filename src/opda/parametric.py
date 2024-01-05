"""Parametric distributions and tools for optimal design analysis."""

import numpy as np
from scipy import special

import opda.random


class QuadraticDistribution:
    """The Quadratic distribution.

    When using random search to optimize a deterministic smooth
    function, the best score asymptotically approaches a quadratic
    distribution. In particular, if the search distribution is a
    continuous distribution and the function is well-approximated by a
    second-order Taylor expansion near the optimum, then the tail of the
    score distribution will approach a quadratic distribution.

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
        distribution, as opposed to the concave form. When optimizing
        via random search, the tail of the score distribution approaches
        the convex form when minimizing and the concave when maximizing.
    """

    def __init__(
            self,
            a,
            b,
            c,
            convex = False,
    ):
        # Validate the arguments.
        if not np.isscalar(a):
            raise ValueError("a must be a scalar.")

        if not np.isscalar(b):
            raise ValueError("b must be a scalar.")

        if not np.isscalar(c):
            raise ValueError("c must be a scalar.")

        if not isinstance(convex, bool):
            raise TypeError("convex must be a boolean.")

        if a > b:
            raise ValueError("a must be less than or equal to b.")

        # Bind attributes to the instance.
        self.a = a
        self.b = b
        self.c = c
        self.convex = convex

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.a == other.a
                and self.b == other.b
                and self.c == other.c
                and self.convex == other.convex
            )
        return NotImplemented

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"a={self.a!s},"
            f" b={self.b!s},"
            f" c={self.c!s},"
            f" convex={self.convex!s}"
            f")"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"a={self.a!r},"
            f" b={self.b!r},"
            f" c={self.c!r},"
            f" convex={self.convex!r}"
            f")"
        )

    def sample(self, size=None, *, generator=None):
        """Return a sample from the quadratic distribution.

        Parameters
        ----------
        size : None, int, or tuple of ints, optional (default=None)
            The desired shape of the returned sample. If ``None``,
            then the sample is a scalar.
        generator : None or np.random.Generator, optional (default=None)
            The random number generator to use. If ``None``, then the
            global default random number generator is used. See
            :py:mod:`opda.random` for more information.

        Returns
        -------
        array of floats
            The sample from the distribution.
        """
        # Validate arguments.
        generator = (
            generator
            if generator is not None else
            opda.random.DEFAULT_GENERATOR
        )

        # Compute the sample.
        return self.ppf(generator.uniform(0., 1., size=size))

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

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.convex:
                ps = (c / (b - a)) * ((ys - a) / (b - a))**(c - 1)
            else:  # concave
                ps = (c / (b - a)) * ((b - ys) / (b - a))**(c - 1)

        ps = np.where(
            (ys < a) | (ys > b),
            0.,
            ps,
        )[()]  # If the result is a 0d array, convert to scalar.

        return ps

    def cdf(self, ys):
        r"""Return the cumulative probability at ``ys``.

        We define the cumulative distribution function, F, using less
        than or equal to:

        .. math::

           F(y) = \mathbb{P}(Y \leq y)

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

        # Handle values of y outside the support by clipping to a and b
        # since the CDF is 0 when y is below a and 1 when y is above b.
        ys = np.clip(ys, a, b)

        if self.convex:
            qs = ((ys - a) / (b - a))**c
        else:  # concave
            qs = 1 - ((b - ys) / (b - a))**c

        return qs

    def ppf(self, qs):
        r"""Return the quantile at ``qs``.

        We define the quantile function, Q, as:

        .. math::

           Q(p) = \inf \{y\in\mathbb{R}\mid p\leq F(y)\}

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
            raise ValueError("qs must be between 0 and 1, inclusive.")
        qs = np.clip(qs, 0., 1.)

        # Compute the quantiles.
        a, b, c = self.a, self.b, self.c

        if self.convex:
            ys = a + (b - a) * qs**(1/c)
        else:  # concave
            ys = b - (b - a) * (1 - qs)**(1/c)

        return ys

    def quantile_tuning_curve(self, ns, q=0.5, minimize=None):
        """Return the quantile tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : array of positive floats, required
            The points at which to evaluate the tuning curve.
        q : float between 0 and 1, optional (default=0.5)
            The quantile at which to evaluate the tuning curve.
        minimize : bool or None, optional (default=None)
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it. Defaults to
            ``None``, in which case it is taken to be the same as
            ``self.convex``, so convex quadratic distributions will
            minimize and concave ones will maximize.

        Returns
        -------
        array of floats
            The quantile tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        if not np.isscalar(q):
            raise ValueError("q must be a scalar.")
        if q < 0. or q > 1.:
            raise ValueError("q must be between 0 and 1, inclusive.")

        minimize = minimize if minimize is not None else self.convex
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        a, b, c = self.a, self.b, self.c

        if self.convex:
            if minimize:
                ys = a + (b - a) * (1 - q**(1/ns))**(1/c)
            else:  # maximize
                ys = a + (b - a) * q**(1/(c * ns))
        else:  # concave
            if minimize:
                ys = b - (b - a) * q**(1/(c * ns))
            else:  # maximize
                ys = b - (b - a) * (1 - q**(1/ns))**(1/c)

        return ys

    def average_tuning_curve(self, ns, minimize=None):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : array of positive floats, required
            The points at which to evaluate the tuning curve.
        minimize : bool or None, optional (default=None)
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it. Defaults to
            ``None``, in which case it is taken to be the same as
            ``self.convex``, so convex quadratic distributions will
            minimize and concave ones will maximize.

        Returns
        -------
        array of floats
            The average tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        minimize = minimize if minimize is not None else self.convex
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        a, b, c = self.a, self.b, self.c

        if self.convex:
            if minimize:
                ys = a + (b - a) * np.exp(
                    special.loggamma(ns + 1)
                    + special.loggamma((c + 1) / c)
                    - special.loggamma(ns + (c + 1) / c),
                )
            else:  # maximize
                ys = a + (b - a) * ns / (ns + 1/c)
        else:  # concave
            if minimize:
                ys = b - (b - a) * ns / (ns + 1/c)
            else:  # maximize
                ys = b - (b - a) * np.exp(
                    special.loggamma(ns + 1)
                    + special.loggamma((c + 1) / c)
                    - special.loggamma(ns + (c + 1) / c),
                )

        return ys

    @classmethod
    def estimate_initial_parameters_and_bounds(
            cls,
            ys,
            fraction = 1.,
            convex = False,
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
        # Validate arguments.
        ys = np.array(ys)
        if len(ys.shape) != 1:
            raise ValueError(f"ys must be a 1D array, not {len(ys.shape)}D.")
        if len(ys) == 0:
            raise ValueError("ys must be non-empty.")

        if not np.isscalar(fraction):
            raise ValueError("fraction must be a scalar.")
        if fraction < 0. or fraction > 1.:
            raise ValueError(
                "fraction must be between 0 and 1, inclusive.",
            )

        if not isinstance(convex, bool):
            raise TypeError("convex must be a boolean.")

        # Compute the initial parameters and bounds.
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
