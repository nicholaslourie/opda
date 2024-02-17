"""Parametric distributions and tools for optimal design analysis."""

import importlib.resources
import json

import numpy as np
from scipy import special

from opda import utils
import opda.random

# backwards compatibility (Python < 3.9)

import sys  # ruff: isort: skip
# NOTE: This import is for checking the Python version for backwards
# compatibility in computing the _APPROXIMATIONS constant below.


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

        b = np.array(b)[()]
        if not np.isscalar(b):
            raise ValueError("b must be a scalar.")

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
        ys : array of float, required
            The points at which to evaluate the cumulative probability.

        Returns
        -------
        array of floats
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
            ys = a + (b - a) * qs**(2/c)
        else:  # concave
            ys = b - (b - a) * (1 - qs)**(2/c)

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

        fraction = np.array(fraction)[()]
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
    """The Noisy Quadratic distribution.

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
    a : float, required
        The minimum value that the distribution can take, without
        accounting for the additive noise.
    b : float, required
        The maximum value that the distribution can take, without
        accounting for the additive noise.
    c : positive int, required
        The *effective* number of hyperparameters. Values of ``c``
        greater than 10 are not supported.
    o : non-negative float, required
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

        b = np.array(b)[()]
        if not np.isscalar(b):
            raise ValueError("b must be a scalar.")

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
