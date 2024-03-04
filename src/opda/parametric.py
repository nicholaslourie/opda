"""Parametric distributions and tools for optimal design analysis."""

import importlib.resources
import json

import numpy as np
from scipy import special

from opda import exceptions, utils
import opda.random

# backwards compatibility (Python < 3.9)

import sys  # ruff: isort: skip
# NOTE: This import is for checking the Python version for backwards
# compatibility in computing the _APPROXIMATIONS constant below.


class QuadraticDistribution:
    r"""The Quadratic distribution.

    When using random search to optimize a deterministic smooth
    function, the best score asymptotically approaches a quadratic
    distribution. In particular, if the search distribution is a
    continuous distribution and the function is well-approximated by a
    second-order Taylor expansion near the optimum, then the tail of the
    score distribution will approach a quadratic distribution.

    Parameters
    ----------
    a : finite float, required
        The minimum value that the distribution can take.
    b : finite float, required
        The maximum value that the distribution can take.
    c : positive int, required
        The *effective* number of hyperparameters.
    convex : bool, optional (default=False)
        Whether or not to use the convex form of the quadratic
        distribution, as opposed to the concave form. When optimizing
        via random search, the tail of the score distribution approaches
        the convex form when minimizing and the concave when maximizing.

    Attributes
    ----------
    mean : float
        The distribution's mean.
    variance : float
        The distribution's variance.

    See Also
    --------
    NoisyQuadraticDistribution :
        The quadratic distribution with additive normal noise.

    Notes
    -----
    The quadratic distribution, :math:`\mathcal{Q}(\alpha, \beta,
    \gamma)`, has a dual relationship to itself:

    .. math::

       Y \sim \mathcal{Q}_{\max}(\alpha, \beta, \gamma)
       \iff
       -Y \sim \mathcal{Q}_{\min}(-\beta, -\alpha, \gamma)

    Where :math:`\mathcal{Q}_{\max}` and :math:`\mathcal{Q}_{\min}` are
    the concave and convex quadratic distributions, respectively.

    The :math:`\alpha` and :math:`\beta` parameters can also be seen as
    defining a location-scale family:

    .. math::

       Y \sim \mathcal{Q}(0, 1, \gamma)
       \iff
       \alpha + (\beta - \alpha) Y \sim \mathcal{Q}(
         \alpha, \beta, \gamma
       )

    In other words, :math:`\alpha` and :math:`\beta` act on the
    distribution by linearly changing its support.
    """

    def __init__(
            self,
            a,
            b,
            c,
            convex = False,
    ):
        # Validate the arguments.
        a = np.array(a)[()]
        if not np.isscalar(a):
            raise ValueError("a must be a scalar.")
        if not np.isfinite(a):
            raise ValueError("a must be finite.")

        b = np.array(b)[()]
        if not np.isscalar(b):
            raise ValueError("b must be a scalar.")
        if not np.isfinite(b):
            raise ValueError("b must be finite.")

        c = np.array(c)[()]
        if not np.isscalar(c):
            raise ValueError("c must be a scalar.")
        if c % 1 != 0:
            raise ValueError("c must be an integer.")
        if c <= 0:
            raise ValueError("c must be positive.")

        if not isinstance(convex, bool):
            raise TypeError("convex must be a boolean.")

        if a > b:
            raise ValueError("a must be less than or equal to b.")

        # Bind arguments to the instance as attributes.
        self.a = a
        self.b = b
        self.c = c
        self.convex = convex

        # Bind other attributes to the instance.
        self.mean = (
            a + (b - a) * c / (c + 2)
            if convex else
            a + (b - a) * 2 / (c + 2)
        )
        self.variance = (b - a)**2 * 4 * c / ((c + 2)**2 * (c + 4))

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
        float or array of floats from a to b inclusive
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
        ys : float or array of floats, required
            The points at which to evaluate the probability density.

        Returns
        -------
        non-negative float or array of floats
            The probability density at ``ys``.
        """
        ys = np.array(ys)

        a, b, c = self.a, self.b, self.c

        if a == b:
            return np.where(ys == a, np.inf, 0.)[()]

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.convex:
                ps = (c / (2*(b - a))) * ((ys - a) / (b - a))**(c/2 - 1)
            else:  # concave
                ps = (c / (2*(b - a))) * ((b - ys) / (b - a))**(c/2 - 1)

        ps = np.where(
            (ys < a) | (ys > b),
            0.,
            ps,
        )[()]

        return ps

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
        ys = np.array(ys)

        a, b, c = self.a, self.b, self.c

        if a == b:
            return np.where(ys < a, 0., 1.)[()]

        # Handle values of y outside the support by clipping to a and b
        # since the CDF is 0 when y is below a and 1 when y is above b.
        ys = np.clip(ys, a, b)

        if self.convex:
            qs = ((ys - a) / (b - a))**(c/2)
        else:  # concave
            qs = 1 - ((b - ys) / (b - a))**(c/2)

        return qs

    def ppf(self, qs):
        r"""Return the quantile at ``qs``.

        We define the quantile function, :math:`Q`, as:

        .. math::

           Q(p) = \inf \{y\in[a, b]\mid p\leq F(y)\}

        where :math:`F` is the cumulative distribution function.

        Parameters
        ----------
        qs : float or array of floats from 0 to 1 inclusive, required
            The points at which to evaluate the quantiles.

        Returns
        -------
        float or array of floats from a to b inclusive
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
            ys = a + (b - a) * qs**(2/c)
        else:  # concave
            ys = b - (b - a) * (1 - qs)**(2/c)

        return ys

    def quantile_tuning_curve(self, ns, q=0.5, minimize=None):
        """Return the quantile tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        q : float from 0 to 1 inclusive, optional (default=0.5)
            The quantile at which to evaluate the tuning curve.
        minimize : bool or None, optional (default=None)
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it. Defaults to
            ``None``, in which case it is taken to be the same as
            ``self.convex``, so convex quadratic distributions will
            minimize and concave ones will maximize.

        Returns
        -------
        float or array of floats from a to b inclusive
            The quantile tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        q = np.array(q)[()]
        if not np.isscalar(q):
            raise ValueError("q must be a scalar.")
        if q < 0. or q > 1.:
            raise ValueError("q must be between 0 and 1, inclusive.")

        minimize = minimize if minimize is not None else self.convex
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the quantile tuning curve.
        a, b, c = self.a, self.b, self.c

        if self.convex:
            if minimize:
                ys = a + (b - a) * (1 - q**(1/ns))**(2/c)
            else:  # maximize
                ys = a + (b - a) * q**(2/(c * ns))
        else:  # concave
            if minimize:
                ys = b - (b - a) * q**(2/(c * ns))
            else:  # maximize
                ys = b - (b - a) * (1 - q**(1/ns))**(2/c)

        return ys

    def average_tuning_curve(self, ns, minimize=None):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        minimize : bool or None, optional (default=None)
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it. Defaults to
            ``None``, in which case it is taken to be the same as
            ``self.convex``, so convex quadratic distributions will
            minimize and concave ones will maximize.

        Returns
        -------
        float or array of floats from a to b inclusive
            The average tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        minimize = minimize if minimize is not None else self.convex
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the average tuning curve.
        a, b, c = self.a, self.b, self.c

        if self.convex:
            if minimize:
                ys = a + (b - a) * np.exp(
                    special.loggamma(ns + 1)
                    + special.loggamma((c + 2) / c)
                    - special.loggamma(ns + (c + 2) / c),
                )
            else:  # maximize
                ys = a + (b - a) * ns / (ns + 2/c)
        else:  # concave
            if minimize:
                ys = b - (b - a) * ns / (ns + 2/c)
            else:  # maximize
                ys = b - (b - a) * np.exp(
                    special.loggamma(ns + 1)
                    + special.loggamma((c + 2) / c)
                    - special.loggamma(ns + (c + 2) / c),
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
        fraction : float from 0 to 1 inclusive of 1, optional (default=1.)
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

        fraction = np.array(fraction)[()]
        if not np.isscalar(fraction):
            raise ValueError("fraction must be a scalar.")
        if fraction <= 0. or fraction > 1.:
            raise ValueError(
                "fraction must be between 0 and 1, inclusive of 1.",
            )

        if not isinstance(convex, bool):
            raise TypeError("convex must be a boolean.")

        # Compute the initial parameters and bounds.
        n_fraction = int(fraction * len(ys))
        if n_fraction == 0:
            raise ValueError(
                "Taking fraction from ys makes an empty list. Raise"
                " fraction or use a larger sample for ys.",
            )
        ys_fraction = (
            np.sort(ys)[:n_fraction]
            if convex else
            np.sort(ys)[-n_fraction:]
        )

        a = ys_fraction[0]
        b = ys_fraction[-1]
        if a == b:
            return (
                np.array([a, b, np.nan]),
                np.array([(-np.inf, a), (b, np.inf), (0, np.inf)]),
            )

        # Initialize c with its MLE assuming a and b are known.
        c = 2 * (
            1. / np.mean(np.log((b - a) / (ys_fraction[1:-1] - a)))
            if convex else
            1. / np.mean(np.log((b - a) / (b - ys_fraction[1:-1])))
        )

        # Correct a and b for the fact that we're using only a fraction.
        if convex:
            # Push a a bit lower than min(ys).
            a = a - 0.05 * (b - a)
            # Set b so that P(y <= ys_fraction[-1]) = fraction.
            b = a + (b - a) / fraction**(2/c)
            # Push b a little higher.
            b = b + 0.05 * (b - a)
        else:  # concave
            # Push b a bit higher than max(ys).
            b = b + 0.05 * (b - a)
            # Set a so that P(y > ys_fraction[0]) = fraction.
            a = b - (b - a) / fraction**(2/c)
            # Push a a little lower.
            a = a - 0.05 * (b - a)

        params = np.array([a, b, c])
        bounds = np.array([
            (-np.inf, ys_fraction[0]),
            (ys_fraction[-1], np.inf),
            (0, np.inf),
        ])

        return params, bounds


class NoisyQuadraticDistribution:
    r"""The Noisy Quadratic distribution.

    When using random search to optimize a smooth function with additive
    normal noise, the best score asymptotically approaches a noisy
    quadratic distribution. In particular, if the search distribution is
    a continuous distribution, the function is well-approximated by a
    second-order Taylor expansion near the optimum, and the observed
    score is the result of the function plus additive normal noise, then
    the tail of the score distribution will approach a noisy quadratic
    distribution.

    Parameters
    ----------
    a : finite float, required
        The minimum value that the distribution can take, without
        accounting for the additive noise.
    b : finite float, required
        The maximum value that the distribution can take, without
        accounting for the additive noise.
    c : positive int, required
        The *effective* number of hyperparameters. Values of ``c``
        greater than 10 are not supported.
    o : finite non-negative float, required
        The standard deviation of the additive noise.
    convex : bool, optional (default=False)
        Whether or not to use the convex form of the noisy quadratic
        distribution, as opposed to the concave form. When optimizing
        via random search, the tail of the score distribution approaches
        the convex form when minimizing and the concave when maximizing.

    Attributes
    ----------
    mean : float
        The distribution's mean.
    variance : float
        The distribution's variance.

    See Also
    --------
    QuadraticDistribution :
        A *noiseless* version of the noisy quadratic distribution.

    Notes
    -----
    The noisy quadratic distribution, :math:`\mathcal{Q}(\alpha, \beta,
    \gamma, \sigma)`, has a dual relationship to itself:

    .. math::

       Y \sim \mathcal{Q}_{\max}(\alpha, \beta, \gamma, \sigma)
       \iff
       -Y \sim \mathcal{Q}_{\min}(-\beta, -\alpha, \gamma, \sigma)

    Where :math:`\mathcal{Q}_{\max}` and :math:`\mathcal{Q}_{\min}` are
    the concave and convex noisy quadratic distributions, respectively.

    The :math:`\alpha` and :math:`\beta` parameters can also be seen as
    defining a location-scale family:

    .. math::

       Y \sim \mathcal{Q}(0, 1, \gamma, \sigma)
       \iff
       \alpha + (\beta - \alpha) Y \sim \mathcal{Q}(
         \alpha, \beta, \gamma, (\beta - \alpha)\sigma
       )

    In other words, :math:`\alpha` and :math:`\beta` act on the
    distribution by linearly changing its support and residual standard
    deviation, :math:`\sigma`.
    """

    def __init__(
            self,
            a,
            b,
            c,
            o,
            convex = False,
    ):
        # Validate the arguments.
        a = np.array(a)[()]
        if not np.isscalar(a):
            raise ValueError("a must be a scalar.")
        if not np.isfinite(a):
            raise ValueError("a must be finite.")

        b = np.array(b)[()]
        if not np.isscalar(b):
            raise ValueError("b must be a scalar.")
        if not np.isfinite(b):
            raise ValueError("b must be finite.")

        c = np.array(c)[()]
        if not np.isscalar(c):
            raise ValueError("c must be a scalar.")
        if c % 1 != 0:
            raise ValueError("c must be an integer.")
        if c <= 0:
            raise ValueError("c must be positive.")
        if c > 10:
            raise ValueError(
                "Values of c greater than 10 are not supported.",
            )

        o = np.array(o)[()]
        if not np.isscalar(o):
            raise ValueError("o must be a scalar.")
        if not np.isfinite(o):
            raise ValueError("o must be finite.")
        if o < 0:
            raise ValueError("o must be non-negative.")

        if not isinstance(convex, bool):
            raise TypeError("convex must be a boolean.")

        if a > b:
            raise ValueError("a must be less than or equal to b.")

        # Bind arguments to the instance as attributes.
        self.a = a
        self.b = b
        self.c = c
        self.o = o
        self.convex = convex

        # Bind other attributes to the instance.
        self.mean = (
            a + (b - a) * c / (c + 2)
            if convex else
            a + (b - a) * 2 / (c + 2)
        )
        self.variance = o**2 + (b - a)**2 * 4 * c / ((c + 2)**2 * (c + 4))

        # Initialize useful private attributes.
        self._approximate_with = (
            "noiseless" if o < 1e-6 * (b - a) else
            "nothing"   if o < 1e+1 * (b - a) else
            "normal"
        )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.a == other.a
                and self.b == other.b
                and self.c == other.c
                and self.o == other.o
                and self.convex == other.convex
            )
        return NotImplemented

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"a={self.a!s},"
            f" b={self.b!s},"
            f" c={self.c!s},"
            f" o={self.o!s},"
            f" convex={self.convex!s}"
            f")"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"a={self.a!r},"
            f" b={self.b!r},"
            f" c={self.c!r},"
            f" o={self.o!r},"
            f" convex={self.convex!r}"
            f")"
        )

    def sample(self, size=None, *, generator=None):
        """Return a sample from the noisy quadratic distribution.

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
        a, b, c, o = self.a, self.b, self.c, self.o
        # Sample the quadratic distribution via inverse transform sampling.
        qs = generator.uniform(0., 1., size=size)
        if self.convex:
            ys = a + (b - a) * qs**(2/c)
        else:  # concave
            ys = b - (b - a) * (1 - qs)**(2/c)
        # Add the normally distributed noise.
        ys += generator.normal(0, o, size=size)

        return ys

    def pdf(self, ys):
        """Return the probability density at ``ys``.

        Parameters
        ----------
        ys : float or array of floats, required
            The points at which to evaluate the probability density.

        Returns
        -------
        non-negative float or array of floats
            The probability density at ``ys``.
        """
        ys = np.array(ys)

        a, b, c, o = self.a, self.b, self.c, self.o

        # Use approximations if appropriate.

        if a == b and o == 0.:
            return np.where(ys == a, np.inf, 0.)[()]

        if self._approximate_with == "noiseless":
            with np.errstate(divide="ignore", invalid="ignore"):
                if self.convex:
                    ps = (c / (2*(b - a))) * ((ys - a) / (b - a))**(c/2 - 1)
                else:  # concave
                    ps = (c / (2*(b - a))) * ((b - ys) / (b - a))**(c/2 - 1)

            ps = np.where(
                (ys < a) | (ys > b),
                0.,
                ps,
            )[()]

            return ps

        if self._approximate_with == "normal":
            return utils.normal_pdf(
                (ys - self.mean) / self.variance**0.5,
            ) / self.variance**0.5

        # Compute the PDF.

        if self.convex:
            loc = (ys - a) / (b - a)
            scale = -o / (b - a)
        else:  # concave
            loc = (b - ys) / (b - a)
            scale = o / (b - a)

        ps = (scale / o) * (0.5 * c) * self._partial_normal_moment(
            loc=loc,
            scale=scale,
            k=(c-2)/2,
        )

        ps = np.clip(ps, 0., None)

        return ps

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
        ys = np.array(ys)

        a, b, c, o = self.a, self.b, self.c, self.o

        # Use approximations if appropriate.

        if a == b and o == 0.:
            return np.where(ys < a, 0., 1.)[()]

        if self._approximate_with == "noiseless":
            # Handle values of y outside the support by clipping to a and b
            # since the CDF is 0 when y is below a and 1 when y is above b.
            ys = np.clip(ys, a, b)

            if self.convex:
                qs = ((ys - a) / (b - a))**(c/2)
            else:  # concave
                qs = 1 - ((b - ys) / (b - a))**(c/2)

            return qs

        if self._approximate_with == "normal":
            return utils.normal_cdf(
                (ys - self.mean) / self.variance**0.5,
            )

        # Compute the CDF.

        if self.convex:
            point = (ys - b) / o
            loc = (ys - a) / (b - a)
            scale = -o / (b - a)
        else:  # concave
            point = (ys - a) / o
            loc = (b - ys) / (b - a)
            scale = o / (b - a)

        qs = (
            utils.normal_cdf(point)
            - self._partial_normal_moment(
                loc=loc,
                scale=scale,
                k=c/2,
            )
        )

        qs = np.clip(qs, 0., 1.)

        return qs

    def ppf(self, qs):
        r"""Return the quantile at ``qs``.

        We define the quantile function, :math:`Q`, as:

        .. math::

           Q(p) = \inf \{y\in\mathbb{R}\mid p\leq F(y)\}

        where :math:`F` is the cumulative distribution function.

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

        a, b, o = self.a, self.b, self.o

        # Numerically invert the CDF with the bisection method.
        ys_lo = np.full_like(qs, a - 6 * o)
        ys_hi = np.full_like(qs, b + 6 * o)
        ys = (ys_lo + ys_hi) / 2
        for _ in range(30):
            y_is_lo = self.cdf(ys) < qs

            ys_lo[y_is_lo] = ys[y_is_lo]
            ys_hi[~y_is_lo] = ys[~y_is_lo]

            ys = (ys_lo + ys_hi) / 2

        ys = np.array(ys)  # Make scalars 0d arrays.
        ys[qs == 0.] = -np.inf
        ys[qs == 1.] = np.inf
        if o == 0.:
            ys = np.clip(ys, a, b)

        return ys[()]

    def quantile_tuning_curve(self, ns, q=0.5, minimize=None):
        """Return the quantile tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        q : float from 0 to 1 inclusive, optional (default=0.5)
            The quantile at which to evaluate the tuning curve.
        minimize : bool or None, optional (default=None)
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it. Defaults to
            ``None``, in which case it is taken to be the same as
            ``self.convex``, so convex noisy quadratic distributions
            will minimize and concave ones will maximize.

        Returns
        -------
        float or array of floats
            The quantile tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        q = np.array(q)[()]
        if not np.isscalar(q):
            raise ValueError("q must be a scalar.")
        if q < 0. or q > 1.:
            raise ValueError("q must be between 0 and 1, inclusive.")

        minimize = minimize if minimize is not None else self.convex
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        # Compute the quantile tuning curve.
        if minimize:
            ys = self.ppf(1 - q**(1/ns))
        else:  # maximize
            ys = self.ppf(q**(1/ns))

        return ys

    def average_tuning_curve(self, ns, minimize=None, *, atol=None):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        minimize : bool or None, optional (default=None)
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it. Defaults to
            ``None``, in which case it is taken to be the same as
            ``self.convex``, so convex noisy quadratic distributions
            will minimize and concave ones will maximize.
        atol : non-negative float or None, optional (default=None)
            The absolute tolerance to use for stopping the computation.
            The average tuning curve is computed via numerical
            integration. The computation stops when the error estimate
            is less than ``atol``. If ``None``, then ``atol`` is set
            automatically based on the distribution's parameters.

        Returns
        -------
        float or array of floats
            The average tuning curve evaluated at ``ns``.
        """
        # Validate the arguments.
        ns = np.array(ns)
        if np.any(ns <= 0):
            raise ValueError("ns must be positive.")

        minimize = minimize if minimize is not None else self.convex
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean.")

        if atol is not None and atol < 0:
            raise ValueError("atol must be non-negative.")

        # Compute the average tuning curve.
        a, b, o = self.a, self.b, self.o

        # NOTE: We compute the average tuning curve via numerical
        # integration. Let T_n be the max of the first n samples. If the
        # CDF for one sample is F and the PDF is f, then the CDF for the
        # max of n samples is F(y)^n and the PDF is n F(y)^(n-1) f(y).
        #
        # We could evaluate the expected value directly using its
        # definition:
        #
        #   E[T_n] = \int_{-\infty}^{\infty} y n F(y)^(n-1) f(y) dy
        #
        # but, in general, the expected value can also be expressed in
        # terms of the CDF:
        #
        #   E[Y] = \int_0^{\infty} (1 - F(y)^n) dy - \int_{-\infty}^0 F(y)^n dy
        #        = \int_{-\infty}^{\infty} 1[y > 0] - F(y)^n dy
        #
        # This alternative expression has the advantage that we only
        # need evaluate the CDF rather than the CDF and PDF for each
        # point. Using this expression, we can compute the integral via
        # the composite trapezoidal rule [1]. Since the CDF is
        # essentially 0 below a - 6o and 1 above b + 6o, the interesting
        # part of the integral is within that range. Instead of
        # integrating from negative to positive infinity, we integrate
        # from a - 6o to b + 6o, then we add back the omitted parts of
        # the integral. In particular, if a - 6o > 0 then from 0 to a -
        # 6o the integrand, 1[y > 0] - F(y)^n, is essentially 1, and if
        # b + 6o < 0 then from b + 6o to 0 the integrand, 1[y > 0] -
        # F(y)^n, is essentially -1. Thus, the omitted parts of the
        # integral evaluate to max(0, a - 6o) and min(0, b + 6o).
        #
        # Romberg's method runs in essentially the same time as the
        # composite trapezoidal rule and often converges faster;
        # however, we found empirically that it does not converge any
        # faster for the integrals of interest. Thus, we use the simpler
        # composite trapezoidal rule.
        #
        # [1]: https://en.wikipedia.org/wiki/Trapezoidal_rule
        lo, hi = a - 6 * o, b + 6 * o
        atol = atol if atol is not None else 1e-6 * (hi - lo)
        h = hi - lo
        xs = np.array([lo, hi])
        ys = 0.5 * h * np.sum(
            (xs > 0) - (1 - (1 - self.cdf(xs))**ns[..., None])
            if minimize else
            (xs > 0) - self.cdf(xs)**ns[..., None],
            axis=-1,
        )
        for i in range(1, 31):
            h *= 0.5
            xs = lo + np.arange(1, 2**i, 2) * h
            ys_prev, ys = ys, (
                0.5 * ys + h * np.sum(
                    (xs > 0) - (1 - (1 - self.cdf(xs))**ns[..., None])
                    if minimize else
                    (xs > 0) - self.cdf(xs)**ns[..., None],
                    axis=-1,
                )
            )

            # As the composite trapezoid rule has error O(N^{-2}),
            # doubling the number of points reduces the error by about a
            # factor of 4. Thus, if "I" is the integral's value and "e"
            # is the error, we have approximately:
            #
            #   ys - ys_prev = (I - e) - (I - 4e) = 3e
            #
            # So, the current error is about (ys - ys_prev) / 3.
            ##
            err = np.max(np.abs(ys - ys_prev)) / 3
            if i > 3 and err < atol:
                break
        else:
            raise exceptions.IntegrationError(
                "Convergence failed in the allocated number of iterations.",
            )

        return max(0., lo) + min(0., hi) + ys

    @np.errstate(invalid="ignore")
    def _partial_normal_moment(self, loc, scale, k):
        # NOTE: This function returns the kth partial moment from 0 to 1
        # for a normal distribution with mean ``loc`` and scale
        # ``scale``: E_0^1[X^k | loc, scale]. It handles both integer
        # and fractional partial moments. Use this private method
        # whenever a partial normal moment is required by one of the
        # other (public) methods.
        #
        # The partial moments are computed from base partial moments via
        # a recursive formula (based on Equation 3.4 in [1]):
        #
        #     E_0^1[X^n] = E_{-\infty}^1[X^n] - E_{-\infty}^0[X^n]
        #                = o^2 0^{n-1} f(0) - o^2 f(1)
        #                  + u E_0^1[X^{n-1}]
        #                  + (n-1) o^2 E_0^1[X^{n-2}]
        #
        # Where f is the PDF of the normal distribution with mean
        # ``loc`` (u) and scale ``scale`` (o). While [1] provides a
        # derivation only for integer moments, the formula is also valid
        # for fractional moments. The recursive formula can be used to
        # step up or down to the desired moment from a pair of base
        # moments.
        #
        # [1]: Robert L. Winkler, Gary M. Roodman, Robert R. Britney,
        # (1972) The Determination of Partial Moments. Management
        # Science 19(3):290-296.

        # Validate arguments.
        if k < -0.5:
            raise ValueError(f"k (k={k}) must be at least -0.5.")
        if (2 * k) % 1 != 0:
            raise ValueError(f"2k (k={k}) must be an integer.")

        # base case
        i, moment_prev, moment_curr = self._base_partial_normal_moments(
            loc, scale, k,
        )

        # recursive case
        var = scale**2
        term = -scale * utils.normal_pdf((1-loc)/scale)
        #   Step up to the desired moment if i < k.
        if i < k:
            for j in range(round(k - i)):
                moment_prev, moment_curr = moment_curr, (
                    loc * moment_curr
                    + (i + j) * var * moment_prev
                    + term
                )
        #   Step down to the desired moment if i > k.
        if i > k:
            # NOTE: The step down formula is not very numerically
            # stable. As it iterates, it removes a number of digits of
            # precision that depends on the scale parameter. Thus,
            # only use the formula for a few iterations at most.
            i, moment_prev, moment_curr = i-1, moment_curr, moment_prev
            for j in range(round(i - k)):
                moment_prev, moment_curr = moment_curr, (
                    moment_prev
                    - loc * moment_curr
                    - term
                ) / ((i - j) * var)

        # NOTE: The limit of the partial moment from 0 to 1 as loc
        # goes to infinity is 0.
        if np.isscalar(moment_curr) and np.isinf(loc):
            moment_curr = 0.
        if not np.isscalar(moment_curr):
            moment_curr[np.isinf(loc)] = 0.

        return moment_curr

    def _base_partial_normal_moments(self, loc, scale, k):
        if k % 1 == 0:
            # When k is an integer, the base partial moments are the 0th
            # and 1st partial moments.
            moment_prev = (
                utils.normal_cdf((1-loc)/scale) - utils.normal_cdf(-loc/scale)
            )
            if k == 0:
                return (0, None, moment_prev)

            moment_curr = loc * moment_prev + scale * (
                utils.normal_pdf(-loc/scale) - utils.normal_pdf((1-loc)/scale)
            )
            if k == 1:
                return (1, None, moment_curr)

            return (1, moment_prev, moment_curr)
        if k % 1 == 0.5:
            # When k is not an integer, the best base partial moments
            # depend on scale and k.
            if k == -0.5 and abs(scale) < 5e-2:
                # For small scales, compute the -0.5 partial moment
                # directly (using the Chebyshev approximation).
                return (
                    -0.5,
                    None,
                    self._partial_fractional_normal_moment(loc, scale, -0.5),
                )
            if k == -0.5 and abs(scale) >= 5e-2:
                # For large scales, step down to the -0.5 partial moment
                # from the 0.5 and 1.5 partial moments.
                return (
                    1.5,
                    self._partial_fractional_normal_moment(loc, scale, 0.5),
                    self._partial_fractional_normal_moment(loc, scale, 1.5),
                )

            # When k > -0.5, compute the partial moment directly using
            # its minimax piecewise polynomial approximation.
            return (
                k,
                None,
                self._partial_fractional_normal_moment(loc, scale, k),
            )

        raise ValueError(
            f"No base case found for kth partial normal moment (k={k}).",
        )

    def _partial_fractional_normal_moment(self, loc, scale, k):
        # NOTE: This function returns the kth partial fractional moment
        # from 0 to 1 for a normal distribution with mean ``loc`` and
        # scale ``scale``. In other words, it computes:
        # E_0^1[X^k | loc, scale]. k should be an odd natural number
        # divided by two (e.g., 0.5, 1.5, etc).
        #
        # The moment is computed by approximating x^k with a polynomial
        # to produce a weighted sum of partial integer normal moments:
        #
        #     E_0^1[X^k] ~ E_0^1[c_n X^n + ... + c_0 X^0]
        #                = c_n E_0^1[X^n] + ... + c_0 E_0^1[X^0]
        #
        # The partial integer moments can be computed via a recursive
        # formula (based on Equation 3.4 in [1]):
        #
        #     E_0^1[X^n] = E_{-\infty}^1[X^n] - E_{-\infty}^0[X^n]
        #                = o^2 0^{n-1} f(0) - o^2 f(1)
        #                  + u E_0^1[X^{n-1}]
        #                  + (n-1) o^2 E_0^1[X^{n-2}]
        #
        # Where f is the PDF of the normal distribution with mean
        # ``loc`` (u) and scale ``scale`` (o).
        #
        # Sufficiently accurate polynomial approximations are not always
        # feasible, so we use piecewise polynomial approximations with
        # the following identity (assuming a <= b <= c):
        #
        #     E_a^c[X^k] = E_a^b[X^k] + E_b^c[X^k]
        #
        # and we adapt the recursive formula above to work on these
        # pieces.
        #
        # [1]: Robert L. Winkler, Gary M. Roodman, Robert R. Britney,
        # (1972) The Determination of Partial Moments. Management
        # Science 19(3):290-296.

        knots, coefficients =\
            self._get_approximation_coefficients(loc, scale, k)

        fractional_moment = 0
        var = scale**2
        for a, b, cs in zip(knots[:-1], knots[1:], coefficients):
            term0 = scale * utils.normal_pdf((a-loc)/scale)
            term1 = -scale * utils.normal_pdf((b-loc)/scale)

            moment_prev, moment_curr = 0, (
                utils.normal_cdf((b-loc)/scale)
                - utils.normal_cdf((a-loc)/scale)
            )

            fractional_moment += cs[0] * moment_curr
            for i, c in enumerate(cs[1:]):
                moment_prev, moment_curr = moment_curr, (
                    loc * moment_curr
                    + i * var * moment_prev
                    + term0
                    + term1
                )
                fractional_moment += c * moment_curr

                term0 *= a
                term1 *= b

        return fractional_moment

    def _get_approximation_coefficients(self, loc, scale, k):
        # NOTE: This function returns the knots and coefficients for an
        # appropriate piecewise polynomial approximation to x^k. The
        # knots and coefficients can then be used to compute partial
        # fractional normal moments. Depending on k and scale, different
        # approximations work best.

        scale_abs = abs(scale)

        # Check for a piecewise minimax polynomial approximation.
        for approximation in _APPROXIMATIONS.get(k, []):
            if scale_abs < approximation["min_scale"]:
                continue

            return (approximation["knots"], approximation["coefficients"])

        # Since there's no piecewise minimax polynomial approximation,
        # compute a Chebyshev approximation.
        #
        # NOTE: Take the Chebyshev approximation over loc - 6 scale to
        # loc + 6 scale intersected with the interval from 0 to 1,
        # because we want to approximate x^k on 0 to 1 where most of the
        # normal's probability mass is. If these intervals don't overlap
        # (e.g., loc is much smaller than 0 or much greater than 1),
        # then pick a small interval near the closest endpoint of [0, 1]
        # in order to ensure the interval is not empty or a single
        # point, as that would produce nonsensical coefficients.
        lo = np.clip(loc - 6 * scale_abs, 0., 1. - scale_abs)
        hi = np.clip(loc + 6 * scale_abs, scale_abs, 1.)
        # NOTE: Set the midpoint closer to the lower endpoint (0), as
        # x^k (e.g., x^0.5) is harder to approximate near 0 than 1.
        md = (3 * lo + hi) / 4
        n_l, n_r = next(
            (n_l, n_r)
            for min_scale, n_l, n_r in [
                (1e-2, 5, 5),
                (3e-3, 4, 4),
                (6e-4, 3, 3),
                (3e-4, 2, 3),
                (  0., 2, 2),
            ]
            if scale_abs >= min_scale
        )

        return (
            [lo, md, hi],
            [
                self._chebyshev_coefficients(lo, md, k, n_l),
                self._chebyshev_coefficients(md, hi, k, n_r),
            ],
        )

    def _chebyshev_coefficients(self, lo, hi, k, n):
        # NOTE: This function returns the coefficients of the nth degree
        # Chebyshev approximation to x^k from lo to hi. The Chebyshev
        # approximation is just the Lagrange interpolating polynomial on
        # the Chebyshev nodes.
        #
        # The Lagrange polynomial is constructed in terms of its
        # roots. For example, consider a single basis polynomial (see
        # https://en.wikipedia.org/wiki/Lagrange_polynomial):
        #
        #     l_j(x) = \prod_{m=0..n\\ m\not=j} \frac{x - x_m}{x_j - x_m}
        #
        # However, we need the polynomial's *coefficients* to compute
        # the partial normal moment. If we view the Lagrange polynomial
        # as a linear combination of monic polynomials:
        #
        #     \prod_{m=0..n\\ m\not=j} x - x_m
        #
        # with weights:
        #
        #     y_j \prod_{m=0..n\\ m\not=j} \frac{1}{x_j - x_m}
        #
        # Then we could compute the coefficients of these monic
        # polynomials, and sum them up to get the coefficients of the
        # Lagrange polynomial. Vieta's formulas give a polynomial's
        # coefficients in terms of its roots (see
        # https://en.wikipedia.org/wiki/Vieta%27s_formulas):
        #
        #     (-1)^k \frac{a_{n-k}}{a_n} = e_k(r_1, \dots, r_n)
        #
        # Where e_k is the kth elementary symmetric polynomial. We can
        # recursively compute the elementary symmetric polynomials using
        # Newton's identities (see "Application to the Roots of a
        # Polynomial" in
        # https://en.wikipedia.org/wiki/Newton's_identities):
        #
        #     e_0 = 1
        #     (-1)^k e_k = 1/k \sum_{i=1..k} (-1)^i e_{k-i} p_i
        #
        # Where p_k is the sum of kth powers.

        # Compute the Chebyshev nodes.
        ns = np.arange(n+1)
        # NOTE: Use Chebyshev nodes that don't include the end points,
        # since x^-0.5 -> infinity as x -> 0.
        xs = lo[..., None] + (hi - lo)[..., None] * 0.5 * (
            1 - np.cos(np.pi * (2 * ns + 1) / (2 * (n + 1)))
        )

        # Compute the Lagrange interpolation weights.
        ws = xs[..., :, None] - xs[..., None, :]
        ws[..., ns, ns] = 1.
        ws = xs**k / np.prod(ws, axis=-1)

        # Compute the coefficients of the interpolating polynomial.
        es = [np.ones_like(xs)]
        ps = []
        cs = [np.sum(ws, axis=-1)]
        for i in range(1, n+1):
            p = np.sum(xs**i, axis=-1, keepdims=True) - xs**i
            ps.append(p)

            e = np.sum(
                (-1)**np.arange(i)
                * np.stack(es[::-1], axis=-1)
                * np.stack(ps, axis=-1),
                axis=-1,
            ) / i
            es.append(e)

            cs.append((-1)**i * np.sum(ws * e, axis=-1))

        return cs[::-1]


_APPROXIMATIONS = {
    float(exponent): [
        {
            "min_scale": approximation["min_scale"],
            "knots": approximation["knots"],
            "coefficients": [
                np.array(cs)
                for cs in approximation["coefficients"]
            ],
            "max_error": approximation["max_error"],
        }
        for approximation in approximations
    ]
    for exponent, approximations in json.loads(
            # backwards compatibility (Python < 3.9)
            importlib.resources.read_text("opda", "_approximations.json")
            if sys.version_info < (3, 9, 0) else
            importlib.resources
                .files("opda").joinpath("_approximations.json")
                .read_text(),
    ).items()
}
"""Metadata for computing partial fractional normal moments.

The :py:class:`NoisyQuadraticDistribution` class uses this constant to
compute partial fractional normal moments (to then compute the CDF and
PDF of the noisy quadratic distribution).

The constant describes piecewise polynomial approximations to functions
of the form :math:`x^k` for various non-integer k. It has the following
(meta)data:

``exponent``
  The exponent, k, being approximated.
``min_scale``
  The minimum scale (multiplier for the standard normal, i.e. the
  standard deviation with a positive or negative sign) for which to use
  the approximation.
``knots``
  The knots defining the polynomial pieces.
``coefficients``
  The list of polynomial coefficients for each piece, starting with the
  constant term then in increasing order.
``max_error``
  The maximum error of the approximation.

Using these coefficients to compute the partial moments, the error is no
greater than the error of the corresponding approximation (up to
rounding errors), and it can be much less.
"""
###
# NOTE: This constant was generated by the following code:
#
#     import json
#     import numpy as np
#     from opda.approximation import (
#       minimax_polynomial_coefficients,
#       piecewise_polynomial_knots,
#     )
#     from opda.exceptions import NumericalError
#     exponents = [
#       (0.5, 6.5e-1, (1, 1, 2, 2, 3, 4)),
#       (0.5, 1.2e-1, (1, 2, 2, 3, 3, 4, 5, 7)),
#       (0.5, 6.5e-2, (1, 2, 2, 3, 3, 4, 4, 5, 6, 7)),
#       (0.5, 3.0e-2, (2, 2, 2, 3, 3, 4, 5, 6, 7, 8)),
#       (0.5, 0.0e-0, (2, 2, 2, 3, 3, 4, 5, 6, 7, 11)),
#       (1.5, 5.0e-1, (5,)),
#       (1.5, 8.5e-2, (11,)),
#       (1.5, 2.0e-2, (19,)),
#       (1.5, 0.0e+0, (11, 17)),
#       (2.5, 2.0e-1, (5,)),
#       (2.5, 0.0e-0, (9,)),
#       (3.5, 0.0e-0, (6,)),
#       (4.5, 0.0e-0, (6,)),
#     ]
#     approximations = {exponent: [] for exponent, _, _ in exponents}
#     for exponent, min_scale, ns in exponents:
#       knots, err = piecewise_polynomial_knots(
#         f=lambda x: x**exponent, a=0., b=1., ns=ns,
#       )
#       coefficients = []
#       for a, b, n in zip(knots[:-1], knots[1:], ns):
#         for transform in [(-1., 1.), (0., 1.), (a, 1.)]:
#           try:
#             cs, _ = minimax_polynomial_coefficients(
#               f=lambda x: x**exponent,
#               a=a,
#               b=b,
#               n=n,
#               transform=transform,
#             )
#           except NumericalError:
#             continue
#           coefficients.append(cs)
#           break
#         else:
#           raise NumericalError("Failed to find coefficients.")
#       approximations[exponent].append({
#         "min_scale": min_scale,
#         "knots": knots.tolist(),
#         "coefficients": [cs.tolist() for cs in coefficients],
#         "max_error": err,
#       })
#     with open("_approximations.json", "w") as fout:
#       json.dump(approximations, fout)
#
# The code may take a few minutes to run, since it has to compute the
# best piecewise polynomial approximations.
##
