"""Parametric distributions and tools for optimal design analysis."""

import collections
import importlib.resources
import itertools
import json
import warnings

import numpy as np
from scipy import optimize, special, stats

from opda import exceptions, utils
import opda.random

# backwards compatibility (scipy < 1.11)

import importlib.metadata  # ruff: isort: skip
# NOTE: This import is for checking the scipy version for backwards
# compatibility in the fit methods below.

scipy_version = tuple(map(int, importlib.metadata.version("scipy").split(".")))


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
        The *effective* number of hyperparameters. Values of ``c``
        greater than 10 are not supported.
    convex : bool, optional
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
    C_MIN : int
        A class attribute giving the minimum supported value of ``c``.
    C_MAX : int
        A class attribute giving the maximum supported value of ``c``.

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

    C_MIN, C_MAX = 1, 10

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
        if not np.isfinite(c):
            raise ValueError("c must be finite.")
        if c % 1 != 0:
            raise ValueError("c must be an integer.")
        if c <= 0:
            raise ValueError("c must be positive.")
        if c < self.C_MIN or c > self.C_MAX:
            raise ValueError(
                f"Values of c less than {self.C_MIN} or greater than"
                f" {self.C_MAX} are not supported.",
            )

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
            if convex else  # concave
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
        size : None, int, or tuple of ints, optional
            The desired shape of the returned sample. If ``None``,
            then the sample is a scalar.
        generator : np.random.Generator or None, optional
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

           Q(p) = \inf \{y\in\operatorname{supp} f(y)\mid p\leq F(y)\}

        where :math:`f` is the probability density, :math:`F` is the
        cumulative distribution function, and
        :math:`\operatorname{supp} f(y)` is the support.

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
        q : float from 0 to 1 inclusive, optional
            The quantile at which to evaluate the tuning curve.
        minimize : bool or None, optional
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
        return self.ppf(
            1 - (1 - q)**(1/ns)
            if minimize else  # maximize
            q**(1/ns),
        )

    def average_tuning_curve(self, ns, minimize=None):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        minimize : bool or None, optional
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
    def fit(
            cls,
            ys,
            limits = (-np.inf, np.inf),
            constraints = None,
            *,
            generator = None,
            method = "maximum_spacing",
    ):
        r"""Return a quadratic distribution fitted to ``ys``.

        Parameters
        ----------
        ys : 1D array of floats, required
            The sample from the distribution.
        limits : pair of floats, optional
            The left-open interval over which to fit the distribution.
            Observations outside the interval are censored (the fit
            considers only that an observation occured above or below
            the interval, not its exact value).
        constraints : mapping or None, optional
            A mapping from strings (the class's parameter names) to
            constraints. For ``a``, ``b``, and ``c``, the constraint can
            be either a float (fixing the parameter's value) or a pair
            of floats (restricting the parameter to that closed
            interval). For ``convex``, the constraint can be either a
            boolean (fixing the parameter's value) or a sequence of
            booleans (restricting the parameter to that set). The
            mapping can be empty and need not include all the
            parameters.
        generator : np.random.Generator or None, optional
            The random number generator to use. If ``None``, then the
            global default random number generator is used. See
            :py:mod:`opda.random` for more information.
        method : str, optional
            One of the strings: "maximum_spacing". The ``method``
            parameter determines how the distribution will be estimated.
            See the notes section for details on different methods.

        Returns
        -------
        QuadraticDistribution
            The fitted quadratic distribution.

        Notes
        -----
        We fit the distribution via maximum product spacing estimation
        (MPS) [1]_ [2]_. MPS maximizes the product of the spacings:
        :math:`F\left(Y_{(i)}; \theta\right) - F\left(Y_{(i-1)};
        \theta\right)`, where :math:`F` is the CDF at the parameters
        :math:`\theta` and :math:`Y_{(i)}` is the i'th order statistic.

        To compute the maximum, we optimize the spacing function with a
        generic algorithm that might fail to find the maximum; however,
        such failures should be rare.

        **Fitting Part of the Data**

        The quadratic distribution approximates the *tail* of the score
        distribution from random search. Thus, you'll typically fit just
        the tail. You can accomplish this using the ``limits`` parameter
        which censors all observations outside a left-open interval.
        Thus, ``limits = (-np.inf, threshold)`` will censor everything
        above ``threshold`` and ``limits = (threshold, np.inf)`` will
        censor everything less than or equal to ``threshold``.

        **Why Use Maximum Spacing Estimation?**

        Often, maximum spacing still works where maximum likelihood
        breaks down. In our case, the quadratic distribution has
        unbounded probability density when :math:`\gamma = 1`. Since the
        parameters :math:`\alpha` and :math:`\beta` control the
        distribution's support, this unbounded density leads to the
        *unbounded likelihood problem* [3]_ which makes the maximum
        likelihood estimator inconsistent. Unlike maximum likelihood,
        maximum spacing remains consistent even with an unbounded
        likelihood.

        **A Variant of Maximum Spacing Estimation**

        Standard MPS can't handle tied data points. In order to handle
        ties and censoring, we use a variant of MPS. For a detailed
        description, see the notes section of
        :py:class:`NoisyQuadraticDistribution`.

        References
        ----------
        .. [1] Cheng, R. C. H., & Amin, N. A. K. (1983). Estimating
           Parameters in Continuous Univariate Distributions with a
           Shifted Origin. Journal of the Royal Statistical Society.
           Series B (Methodological), 45(3), 394-403.

        .. [2] Ranneby, B. (1984). The Maximum Spacing Method. An
           Estimation Method Related to the Maximum Likelihood Method.
           Scandinavian Journal of Statistics, 11(2), 93-112.

        .. [3] Cheng, R. C. H., & Traylor, L. (1995). Non-Regular
           Maximum Likelihood Problems. Journal of the Royal Statistical
           Society. Series B (Methodological), 57(1), 3-44.

        Examples
        --------
        Use :py:meth:`fit` to estimate the distribution from data:

        .. code:: python

           >>> QuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ... )
           QuadraticDistribution(...)

        When fitting a quadratic distribution to the results from random
        search, you'll typically restrict it to be *convex* when
        *minimizing* and *concave* when *maximizing*. You can accomplish
        this with ``constraints``:

        .. code:: python

           >>> minimize = True  # If random search minimized / maximized
           >>> QuadraticDistribution.fit(
           ...   ys=[5.0, 2.9, 5.1, 6.8, 3.2],
           ...   constraints={"convex": True if minimize else False},
           ... )
           QuadraticDistribution(...)

        The quadratic distribution approximates the score distribution's
        left tail when minimizing and right tail when maximizing. You
        can fit to only the tail of your data using ``limits``:

        .. code:: python

           >>> threshold = 5.  # The maximum cross-entropy to consider
           >>> QuadraticDistribution.fit(
           ...   ys=[5.0, 2.9, 5.1, 6.8, 3.2],  # 5.1 & 6.8 are censored
           ...   limits=(
           ...     (-np.inf, threshold)  # minimize: fit the left tail
           ...     if minimize else
           ...     (threshold, np.inf)   # maximize: fit the right tail
           ...   ),
           ...   constraints={"convex": True if minimize else False},
           ... )
           QuadraticDistribution(...)

        You could also censor both tails if necessary:

        .. code:: python

           >>> QuadraticDistribution.fit(
           ...   ys=[5.0, 2.9, 5.1, 6.8, 3.2],
           ...   limits=(1., 10.),
           ... )
           QuadraticDistribution(...)

        Finally, you can use ``constraints`` to bound any of the
        parameters in case you have some extra information. For example,
        if you knew a bound on the performace of the best
        hyperparameters, you might constrain ``a`` (when minimizing) or
        ``b`` (when maximizing):

        .. code:: python

           >>> min_accuracy, max_accuracy = 0., 1.
           >>> QuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ...   constraints=(
           ...     {"a": (min_accuracy, max_accuracy)}
           ...     if minimize else
           ...     {"b": (min_accuracy, max_accuracy)}
           ...   ),
           ... )
           QuadraticDistribution(...)

        Or, you might know that the random search used 3
        hyperparameters, so the effective number of hyperparameters
        (``c``) can be at most that:

        .. code:: python

           >>> n_hyperparameters = 3
           >>> QuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ...   constraints={"c": (1, n_hyperparameters)},
           ... )
           QuadraticDistribution(...)

        You could also fix ``c`` (or ``a``, ``b``, or ``convex``) to a
        particular value:

        .. code:: python

           >>> QuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ...   constraints={"c": 1},
           ... )
           QuadraticDistribution(...)

        Of course, you can mix and match all of these ideas together as
        desired.
        """
        # Validate arguments.
        ys = np.array(ys)
        if len(ys.shape) != 1:
            raise ValueError(f"ys must be a 1D array, not {len(ys.shape)}D.")
        if len(ys) == 0:
            raise ValueError("ys must be non-empty.")
        if limits[0] == -np.inf and np.any(np.isneginf(ys)):
            raise ValueError("ys must not contain -inf unless it is censored.")
        if limits[1] == np.inf and np.any(np.isposinf(ys)):
            raise ValueError("ys must not contain inf unless it is censored.")
        if np.any(np.isnan(ys)):
            raise ValueError("ys must not contain NaN.")
        if np.issubdtype(ys.dtype, np.integer):
            # Only cast ys if it has an integer data type, otherwise
            # preserve its precision which we'll need later in order
            # to decide how much to round the data.
            ys = ys.astype(float)

        limits = np.array(limits)
        if len(limits.shape) != 1:
            raise ValueError("limits must be a 1D sequence.")
        if len(limits) != 2:
            raise ValueError("limits must be a pair.")
        if not np.all(np.isreal(limits)):
            raise TypeError("limits must only contain floats.")
        if np.any(np.isnan(limits)):
            raise ValueError("limits cannot contain NaN values.")
        if limits[0] >= limits[1]:
            raise ValueError(
                "limits must be a proper (left-open) interval. The"
                " lower bound cannot equal or exceed the upper bound.",
            )

        constraints = dict(constraints) if constraints is not None else {}
        for parameter, constraint in constraints.items():
            constraint = np.array(constraint)[()]
            if parameter in {"a", "b", "c"}:
                # Check invariants for a, b, and c.
                if not np.isscalar(constraint) and constraint.shape != (2,):
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a scalar or a pair.",
                    )
                if not np.all(np.isreal(constraint)):
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a float or a pair of floats.",
                    )
                if np.any(np.isnan(constraint)):
                    raise ValueError(
                        f"The constraint for {parameter} cannot"
                        f" contain NaNs.",
                    )
                if constraint.shape == (2,) and constraint[0] > constraint[1]:
                    raise ValueError(
                        f"The constraint for {parameter} cannot have a lower"
                        f" bound greater than the upper bound.",
                    )
                # Check invariants that only apply to c.
                if parameter == "c":
                    if np.any(constraint % 1 != 0):
                        raise ValueError(
                            "The constraint for c must be either an integer"
                            " or a pair of integers.",
                        )
                    if np.isscalar(constraint):
                        if constraint < cls.C_MIN or constraint > cls.C_MAX:
                            raise ValueError(
                                f"The constraint for c fixes its value"
                                f" outside of {cls.C_MIN} to"
                                f" {cls.C_MAX} but only values within"
                                f" that range are supported.",
                            )
                    else:
                        if (
                                constraint[0] > cls.C_MAX
                                or constraint[1] < cls.C_MIN
                        ):
                            raise ValueError(
                                f"The constraint for c excludes all"
                                f" supported values. Only values of c"
                                f" between {cls.C_MIN} and {cls.C_MAX}"
                                f" are supported.",
                            )
                        if (
                                constraint[0] < cls.C_MIN
                                or constraint[1] > cls.C_MAX
                        ):
                            warnings.warn(
                                f"The constraint for c includes values"
                                f" outside of {cls.C_MIN} to"
                                f" {cls.C_MAX}, but only values within"
                                f" that range are supported. Consider"
                                f" revising the constraint to only"
                                f" include values within that range.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
            elif parameter == "convex":
                if not np.isscalar(constraint) and len(constraint.shape) != 1:
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a scalar or 1D list.",
                    )
                if not np.isscalar(constraint) and len(constraint) == 0:
                    raise ValueError(
                        f"The constraint for {parameter} must not be"
                        f" empty.",
                    )
                if constraint.dtype != bool:
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a bool or a list of bools.",
                    )
                if np.any(np.unique(constraint, return_counts=True)[1] > 1):
                    raise ValueError(
                        f"The constraint for {parameter} cannot contain"
                        f" duplicate elements.",
                    )
            else:
                raise ValueError(
                    f"constraints contains an unrecognized key ({parameter}),"
                    f" all keys must be parameters of the class.",
                )

        generator = (
            generator
            if generator is not None else
            opda.random.DEFAULT_GENERATOR
        )

        # Censor the data.
        limit_lower, limit_upper = limits

        n = len(ys)
        n_lower, n_upper = np.sum(ys <= limit_lower), np.sum(ys > limit_upper)
        ys_observed = ys[(limit_lower < ys) & (ys <= limit_upper)]

        if n < 3:
            raise ValueError(
                "ys must contain at least three data points.",
            )

        if len(ys_observed) == 0:
            raise ValueError(
                "All ys are censored because they fall outside of"
                " limits. At least some ys must be observed.",
            )

        # NOTE: When observations are censored in the left or the right
        # tail, the min or max is unavailable. In those cases, use
        # limit_lower as the min and limit_upper as the max as those are
        # the smallest / largest values we know are compatible with the
        # distribution. You could instead use the smallest / largest
        # observed values; however, this approach fails when all values
        # but one are censored because then the min and max are equal.
        y_min = np.min(ys_observed) if n_lower == 0 else limit_lower
        i_min = 1 if n_lower == 0 else n_lower          # the rank of y_min
        y_max = np.max(ys_observed) if n_upper == 0 else limit_upper
        j_max = n if n_upper == 0 else n - n_upper + 1  # the rank of y_max

        # backwards compatibility (scipy < 1.11)
        #
        # This method implements point constraints (fixing a, b, or c to
        # a particular value) differently than interval constraints
        # (restricting a, b, or c to a range). This is necessary when
        # scipy < 1.11 because optimize.differential_evolution
        # encounters zero division errors and uses too big of a
        # population when a parameter has equal upper and lower bounds
        # (see https://github.com/scipy/scipy/issues/17788). After
        # dropping support for scipy < 1.11, try to simplify this method
        # by replacing point constraints ({"a": p}) with interval
        # constraints ({"a": (p, p)}) and sharing the rest of the logic.

        # Handle constraints.
        a_constraint = constraints.get("a", [-np.inf, np.inf])
        a = a_constraint if np.isscalar(a_constraint) else None
        b_constraint = constraints.get("b", [-np.inf, np.inf])
        b = b_constraint if np.isscalar(b_constraint) else None
        c_constraint = constraints.get("c", [cls.C_MIN, cls.C_MAX])
        c = c_constraint if np.isscalar(c_constraint) else None

        cs = range(
            max(cls.C_MIN, c_constraint[0]),
            min(cls.C_MAX, c_constraint[1]) + 1,
        ) if c is None else [c]
        convexs = (
            [False, True] if "convex" not in constraints else
            [constraints["convex"]] if np.isscalar(constraints["convex"]) else
            constraints["convex"]
        )

        # Check for error and warning conditions.
        if y_min == y_max and n_upper == 0 and n_lower == 0\
           and (len(cs) > 1 or len(convexs) > 1):
            warnings.warn(
                "Parameters might be unidentifiable. All ys are equal,"
                " suggesting the distribution is a point mass. The"
                " distribution is a point mass whenever a = b, making c"
                " and convex unidentifiable. If appropriate, use the"
                " constraints parameter to specify c and convex.",
                RuntimeWarning,
                stacklevel=2,
            )

        if a is not None and a > y_min:
            raise ValueError(
                "constraints must not fix a to be greater than the"
                " least observation (or the lower limit if the least"
                " observation is censored).",
            )
        if a is None and a_constraint[0] > y_min:
            raise ValueError(
                "constraints must not constrain a to be greater than the"
                " least observation (or the lower limit if the least"
                " observation is censored).",
            )

        if b is not None and b < y_max:
            raise ValueError(
                "constraints must not fix b to be less than the"
                " greatest observation (or the upper limit if the"
                " greatest observation is censored).",
            )
        if b is None and b_constraint[1] < y_max:
            raise ValueError(
                "constraints must not constrain b to be less than the"
                " greatest observation (or the upper limit if the"
                " greatest observation is censored).",
            )

        # Fit the distribution.
        best_loss = np.inf
        best_parameters = None
        for convex in convexs:
            # Determine the search space.
            bounds = []
            integrality = []

            # NOTE: We must bound a and b for our optimization. Since a
            # is the minimum and b is the maximum of the support, we
            # have: a <= y_min and y_max <= b. For the other sides, we
            # need data-dependent bounds that adapt gracefully across
            # scales.
            #   If we knew (b - a), then we could use:
            # y_min - (b - a) <= a and b <= y_max + (b - a). We'll bound
            # (b - a) as follows. Consider the distribution of
            # (Y_(j) - Y_(i)) / (b - a), it depends only on c and
            # convex. Let w be the 1e-9 quantile of this distribution,
            # then with high probability:
            #
            #   (Y_(j) - Y_(i)) / (b - a) > w
            #
            # or equivalently:
            #
            #   1/w (Y_(j) - Y_(i)) > b - a
            #
            # To approximate w, consider F(Y_(j)) - F(Y_(i)). It has the
            # same distribution as the difference of the i'th and j'th
            # order statistics of the uniform distribution, or
            # Beta(j - i, n - (j - i) + 1). The lower 1e-9 quantile of
            # this distribution bounds the probability mass separating
            # these order statistics. We could then seek the shortest
            # interval containing at least this probability mass in
            # Q(0, 1, c, convex). The length of that interval is a lower
            # bound for w, but it's too conservative. Instead, we
            # approximate w using the length of the equal-tailed
            # interval containing that probability mass.
            p = stats.beta(j_max - i_min, n - (j_max - i_min) + 1).ppf(1e-9)
            w = max(
                1 / np.diff(
                    cls(0, 1, c, convex).ppf([0.5 - p/2, 0.5 + p/2]),
                )[0]
                for c in cs
            )

            a_bounds = (
                # Intersect the constraint and default bounds.
                max(y_min - w * (y_max - y_min), a_constraint[0]),
                min(y_min, a_constraint[1]),
            ) if a is None else (a, a)

            if a_bounds[0] > a_bounds[1]:
                # The intersection of the default bounds and the
                # constraint is empty.
                raise exceptions.OptimizationError(
                    "The constraint on a excludes all promising values"
                    " for it. Consider relaxing the constraint on a or"
                    " re-examining your data.",
                )

            # backwards compatibility (scipy < 1.11)
            if scipy_version < (1, 11) and a_bounds[0] == a_bounds[1]:
                # NOTE: In scipy < 1.11, optimize.differential_evolution
                # encounters a zero division bug when upper and lower
                # bounds are equal. Avoid it by passing a directly.
                a = a_bounds[0]

            if a is None:
                bounds.append(a_bounds)
                integrality.append(False)

            b_bounds = (
                # Intersect the constraint and default bounds.
                max(y_max, b_constraint[0]),
                min(y_max + w * (y_max - y_min), b_constraint[1]),
            ) if b is None else (b, b)

            if b_bounds[0] > b_bounds[1]:
                # The intersection of the default bounds and the
                # constraint is empty.
                raise exceptions.OptimizationError(
                    "The constraint on b excludes all promising values"
                    " for it. Consider relaxing the constraint on b or"
                    " re-examining your data.",
                )

            # backwards compatibility (scipy < 1.11)
            if scipy_version < (1, 11) and b_bounds[0] == b_bounds[1]:
                # NOTE: In scipy < 1.11, optimize.differential_evolution
                # encounters a zero division bug when upper and lower
                # bounds are equal. Avoid it by passing b directly.
                b = b_bounds[0]

            if b is None:
                bounds.append(b_bounds)
                integrality.append(False)

            c_bounds = (
                # Intersect the constraint and default bounds.
                max(cls.C_MIN, c_constraint[0]),
                min(cls.C_MAX, c_constraint[1]),
            ) if c is None else (c, c)

            if c_bounds[0] > c_bounds[1]:
                # The intersection of the default bounds and the
                # constraint is empty.
                raise exceptions.OptimizationError(
                    "The constraint on c excludes all promising values"
                    " for it. Consider relaxing the constraint on c or"
                    " re-examining your data.",
                )

            # backwards compatibility (scipy < 1.11)
            if scipy_version < (1, 11) and c_bounds[0] == c_bounds[1]:
                # NOTE: In scipy < 1.11, optimize.differential_evolution
                # encounters a zero division bug when upper and lower
                # bounds are equal. Avoid it by passing c directly.
                c = c_bounds[0]

            if c is None:
                bounds.append(c_bounds)
                integrality.append(True)

            # Define the loss.
            if method == "maximum_spacing":
                # Maximum spacing estimation is sensitive to closely
                # spaced observations. Two observations that represent
                # the same point but differ due to floating point errors
                # could noticeably impact the estimate's quality. To
                # prevent this, round the data a tiny bit before
                # defining the buckets in order to group such
                # observations together.
                #
                # In practice, this rounding should improve the
                # estimate, but it could cause limit_lower or
                # limit_upper to shift and thus censor a point that's
                # technically within the limits (though extremely close
                # to the boundary).
                finfo = np.finfo(ys_observed.dtype)
                # For rounding, use 3 fewer digits than the coarsest
                # precision of any observed point and clip the number of
                # decimals to stay within those representable by the
                # floating point format.
                decimals = - np.clip(
                    np.log10(np.max(np.abs(np.spacing(ys_observed)))),
                    np.log10(finfo.smallest_normal),
                    np.log10(finfo.max),
                ).astype(int) - 3
                zs, ks = np.unique(np.round(
                    np.concatenate([
                        # lower bound on the support
                        [a_bounds[0]],
                        # lower limit (for the censoring)
                        [limit_lower] if n_lower > 0 else [],
                        # order statistics
                        ys_observed,
                        # upper limit (for the censoring)
                        [limit_upper] if n_upper > 0 else [],
                        # upper bound on the support
                        [b_bounds[1]],
                    ]),
                    decimals=decimals,
                ), return_counts=True)

                ks = ks[1:]  # The leftmost count corresponds to no bucket.
                if n_lower > 0:
                    # Set the count for the left tail bucket.
                    ks[0] = n_lower
                if n_upper > 0:
                    # Remove the extra count created by defining the
                    # right tail bucket.
                    ks[-2] -= 1
                    # Set the count for the right tail bucket with the
                    # extra count at the top of the support.
                    ks[-1] = (n_upper + 1)

                def loss(parameters):
                    dist = cls(
                        a=(
                            a             if a is not None else
                            parameters[0]
                        ),
                        b=(
                            b             if b is not None else
                            parameters[
                                0 + (a is None)
                            ]
                        ),
                        c=(
                            c             if c is not None else
                            parameters[
                                0 + (a is None) + (b is None)
                            ]
                        ),
                        convex=convex,
                    )

                    return - np.sum(
                        # Instead of the raw grouped negative
                        # log-likelihood, divide the sum by (n + 1) and
                        # divide the spacings by ks / (n + 1). This
                        # modification makes the loss an estimator of
                        # the KL-divergence and, more importantly, keeps
                        # the loss's scale constant across sample sizes
                        # which improves the optimization.
                        #
                        # For more discussion, see the M_T2 objective in
                        # "Alternatives to maximum likelihood estimation
                        # based on spacings and the Kullback-Leibler
                        # divergence" (Ekstrom, 2008).
                        ks * np.log(
                            np.diff(dist.cdf(zs)) * (n + 1) / ks,
                        ) / (n + 1),
                        where=ks > 0,
                    )
            else:
                raise ValueError(
                    'method must be "maximum_spacing".',
                )

            # The optimizer can have difficulty finding the optimum, so
            # we provide some initial estimates to make it more robust.
            #
            # The initial estimates use a grid over c. For a and b, we
            # use estimates of the form:
            #
            #   a = y_min - w_a (y_max - y_min)
            #   b = y_max + w_b (y_max - y_min)
            #
            # Ideally, w_a would equal (y_min - a) / (y_max - y_min) and
            # w_b would equal (b - y_max) / (y_max - y_min) because then
            # our estimates would equal a and b exactly. Since y_min and
            # y_max include the location, and (y_max - y_min) includes
            # the scale, the distributions of these quantities depend
            # only on c and convex. Thus, we can treat w_a and w_b as if
            # the underlying distribution was Q(0, 1, c, convex). Let
            # F^-1 be its quantile function. To obtain estimates for w_a
            # and w_b, we replace y_min and y_max with F^-1(i/(n+1)) and
            # F^-1(j/(n+1)) where i and j are the ranks of each. To
            # obtain multiple estimates, we can replace i/(n+1) and
            # j/(n+1) with other quantiles of F(y_min) and
            # F(y_max), which are distributed according to
            # Beta(i, n - i + 1) and Beta(j, n - j + 1). Finally,
            # plugging w_a and w_b into our original equations for a and
            # b yields our final estimates.
            initial_population = []
            for c_candidate in cs:
                initial_estimates = collections.defaultdict(list)
                ds = (
                    # When fitting a and b only adjust y_min lower and
                    # y_max higher than their medians to prevent them
                    # from crossing. Also, only use 3 estimates for each
                    # of a and b since that makes 3 * 3 = 9 estimates
                    # for the pair.
                    [0.0, 0.2, 0.4]
                    if a is None and b is None else
                    # When fitting only a or only b use more initial
                    # estimates for it to ensure the initial population
                    # has a reasonable size.
                    [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
                )
                for d in ds:
                    std_dist = cls(0., 1., c_candidate, convex)
                    std_y_min = std_dist.ppf(
                        stats.beta(i_min, n - i_min + 1).ppf(0.5 - d),
                    )
                    std_y_max = std_dist.ppf(
                        stats.beta(j_max, n - j_max + 1).ppf(0.5 + d),
                    )
                    if a is None:
                        initial_estimates["a"].append(np.clip(
                            # Clip the estimate to obey the constraint.
                            y_min
                            - std_y_min / (std_y_max - std_y_min)
                              * (y_max - y_min),
                            a_bounds[0],
                            a_bounds[1],
                        ))
                    if b is None:
                        initial_estimates["b"].append(np.clip(
                            # Clip the estimate to obey the constraint.
                            y_max
                            + (1 - std_y_max) / (std_y_max - std_y_min)
                              * (y_max - y_min),
                            b_bounds[0],
                            b_bounds[1],
                        ))
                if c is None:
                    initial_estimates["c"].append(c_candidate)

                initial_population.extend(
                    itertools.product(*initial_estimates.values()),
                )

            # Optimize the loss to compute the estimate.
            if len(bounds) > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = optimize.differential_evolution(
                        loss,
                        bounds=bounds,
                        init=initial_population,
                        integrality=integrality,
                        polish=True,
                        updating="immediate",
                        vectorized=False,
                        workers=1,
                        seed=generator,
                    )
                parameters = result.x
                curr_loss = result.fun
            else:  # There are no parameters to fit
                parameters = []
                curr_loss = loss(parameters)

            # Check if the current value of convex is the best so far.
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_parameters = {
                    "a": (
                        a             if a is not None else
                        parameters[0]
                    ),
                    "b": (
                        b             if b is not None else
                        parameters[
                            0 + (a is None)
                        ]
                    ),
                    "c": (
                        c             if c is not None else
                        parameters[
                            0 + (a is None) + (b is None)
                        ]
                    ),
                    "convex": convex,
                }

        if not np.isfinite(best_loss):
            raise exceptions.OptimizationError(
                "fit failed to find parameters with finite loss.",
            )

        if best_parameters.get("c") == 2 and len(convexs) > 1:
            warnings.warn(
                "Parameters might be unidentifiable. The fit found"
                " c = 2. When c = 2, convex is unidentifiable because"
                " either value for convex gives the uniform from a to b."
                " If appropriate, use the constraints parameter to"
                " specify convex.",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(**best_parameters)


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
    convex : bool, optional
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
    C_MIN : int
        A class attribute giving the minimum supported value of ``c``.
    C_MAX : int
        A class attribute giving the maximum supported value of ``c``.

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

    C_MIN, C_MAX = 1, 10

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
        if not np.isfinite(c):
            raise ValueError("c must be finite.")
        if c % 1 != 0:
            raise ValueError("c must be an integer.")
        if c <= 0:
            raise ValueError("c must be positive.")
        if c < self.C_MIN or c > self.C_MAX:
            raise ValueError(
                f"Values of c less than {self.C_MIN} or greater than"
                f" {self.C_MAX} are not supported.",
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
            if convex else  # concave
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

        loc = (
            (ys - a) / (b - a)
            if self.convex else  # concave
            (b - ys) / (b - a)
        )
        scale = o / (b - a)

        ps = c / (2 * (b - a)) * self._partial_normal_moment(
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

        point = (
            (ys - b) / o
            if self.convex else  # concave
            (ys - a) / o
        )
        loc = (
            (ys - a) / (b - a)
            if self.convex else  # concave
            (b - ys) / (b - a)
        )
        scale = o / (b - a)

        qs = (
            utils.normal_cdf(point) + self._partial_normal_moment(
                loc=loc, scale=scale, k=c/2,
            )
            if self.convex else  # concave
            utils.normal_cdf(point) - self._partial_normal_moment(
                loc=loc, scale=scale, k=c/2,
            )
        )

        qs = np.clip(qs, 0., 1.)

        return qs

    def ppf(self, qs):
        r"""Return the quantile at ``qs``.

        We define the quantile function, :math:`Q`, as:

        .. math::

           Q(p) = \inf \{y\in\operatorname{supp} f(y)\mid p\leq F(y)\}

        where :math:`f` is the probability density, :math:`F` is the
        cumulative distribution function, and
        :math:`\operatorname{supp} f(y)` is the support.

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

        a, b, c, o = self.a, self.b, self.c, self.o

        # Use approximations if appropriate.

        if a == b and o == 0.:
            return np.full_like(qs, a)[()]

        if self._approximate_with == "noiseless":
            if self.convex:
                ys = a + (b - a) * qs**(2/c)
            else:  # concave
                ys = b - (b - a) * (1 - qs)**(2/c)

            return ys

        if self._approximate_with == "normal":
            return self.mean + self.variance**0.5 * utils.normal_ppf(qs)

        # Compute the quantiles.

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
        q : float from 0 to 1 inclusive, optional
            The quantile at which to evaluate the tuning curve.
        minimize : bool or None, optional
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
        return self.ppf(
            1 - (1 - q)**(1/ns)
            if minimize else  # maximize
            q**(1/ns),
        )

    def average_tuning_curve(self, ns, minimize=None, *, atol=None):
        """Return the average tuning curve evaluated at ``ns``.

        Parameters
        ----------
        ns : positive float or array of floats, required
            The points at which to evaluate the tuning curve.
        minimize : bool or None, optional
            Whether or not to compute the tuning curve for minimizing a
            metric as opposed to maximizing it. Defaults to
            ``None``, in which case it is taken to be the same as
            ``self.convex``, so convex noisy quadratic distributions
            will minimize and concave ones will maximize.
        atol : non-negative float or None, optional
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

    @classmethod
    def fit(
            cls,
            ys,
            limits = (-np.inf, np.inf),
            constraints = None,
            *,
            generator = None,
            method = "maximum_spacing",
    ):
        r"""Return a noisy quadratic distribution fitted to ``ys``.

        Parameters
        ----------
        ys : 1D array of floats, required
            The sample from the distribution.
        limits : pair of floats, optional
            The left-open interval over which to fit the distribution.
            Observations outside the interval are censored (the fit
            considers only that an observation occured above or below
            the interval, not its exact value).
        constraints : mapping or None, optional
            A mapping from strings (the class's parameter names) to
            constraints. For ``a``, ``b``, ``c``, and ``o`` the constraint
            can be either a float (fixing the parameter's value) or a pair
            of floats (restricting the parameter to that closed
            interval). For ``convex``, the constraint can be either a
            boolean (fixing the parameter's value) or a sequence of
            booleans (restricting the parameter to that set). The
            mapping can be empty and need not include all the
            parameters.
        generator : np.random.Generator or None, optional
            The random number generator to use. If ``None``, then the
            global default random number generator is used. See
            :py:mod:`opda.random` for more information.
        method : str, optional
            One of the strings: "maximum_spacing". The ``method``
            parameter determines how the distribution will be estimated.
            See the notes section for details on different methods.

        Returns
        -------
        NoisyQuadraticDistribution
            The fitted noisy quadratic distribution.

        Notes
        -----
        We fit the distribution via maximum product spacing estimation
        (MPS) [1]_ [2]_. MPS maximizes the product of the spacings:
        :math:`F\left(Y_{(i)}; \theta\right) - F\left(Y_{(i-1)};
        \theta\right)`, where :math:`F` is the CDF at the parameters
        :math:`\theta` and :math:`Y_{(i)}` is the i'th order statistic.

        To compute the maximum, we optimize the spacing function with a
        generic algorithm that might fail to find the maximum; however,
        such failures should be rare.

        **Fitting Part of the Data**

        The noisy quadratic distribution approximates the *tail* of the
        score distribution from random search. Thus, you'll typically
        fit just the tail. You can accomplish this using the ``limits``
        parameter which censors all observations outside a left-open
        interval. Thus, ``limits = (-np.inf, threshold)`` will censor
        everything above ``threshold`` and ``limits = (threshold,
        np.inf)`` will censor everything less than or equal to
        ``threshold``.

        **Why Use Maximum Spacing Estimation?**

        Often, maximum spacing still works where maximum likelihood
        breaks down. In our case, the noisy quadratic distribution has
        unbounded probability density when :math:`\sigma = 0` and
        :math:`\gamma = 1`. Since the parameters :math:`\alpha` and
        :math:`\beta` control the distribution's support, this unbounded
        density leads to the *unbounded likelihood problem* [3]_ which
        makes the maximum likelihood estimator inconsistent. Unlike
        maximum likelihood, maximum spacing remains consistent even with
        an unbounded likelihood.

        **A Variant of Maximum Spacing Estimation**

        Standard MPS can't handle tied data points because then the
        spacing, :math:`F\left(Y_{(i)}; \theta\right) -
        F\left(Y_{(i-1)}; \theta\right)`, would be zero, making the
        whole product of spacings zero.

        To handle ties and censoring, we view maximum spacing as maximum
        likelihood on grouped data [4]_. Roughly, we imagine adding an
        extra data point at :math:`Y_{(n+1)} = \infty`, then grouping
        the data between the order statistics, and finally applying
        maximum likelihood.

        More specifically, we group the data into buckets defined as
        left-open intervals: :math:`\left(Z_{(i-1)}, Z_{(i)}\right]`. To
        construct the buckets, we deduplicate the uncensored order
        statistics along with the lower and upper bounds on the support:

        .. math::

           -\infty = Z_{(0)} < Z_{(1)} = Y_{(i_1)} < \ldots
           < Y_{(i_k)} = Z_{(k)} < Z_{(k+1)} = \infty

        If there are censored observations in the left tail then we
        further divide the first bucket at the lower limit for
        censoring; similarly, if the right tail contains censored
        observations, we divide the last bucket at the upper limit. If
        any of the (uncensored) order statistics equals one of the
        censoring limits, then we leave the bucket as is so as never to
        create a zero-length bucket. In this way, the deduplication
        ensures all buckets have positive length.

        With the buckets defined, we bin the data into each of the
        buckets and put an extra data point in the rightmost one (as in
        typical maximum spacing estimation). Intuitively, this
        corresponds to estimating the CDF at each order statistic by
        i/(n+1), reducing the upward bias at the order statistics. For
        example, if you have one data point, you might guess that it's
        near the median rather than the max of the distribution. If
        :math:`N_i` is the number of data points in bucket
        :math:`\left(Z_{(i-1)}, Z_{(i)}\right]`, then the objective
        becomes:

        .. math::

           \sum_{i=1}^{l} N_i \log\left(
             F\left(Z_{(i)}; \theta\right)
             - F\left(Z_{(i-1)}; \theta\right)
           \right)

        where :math:`l` is the highest index of all the :math:`Z_{(i)}`.

        When fitting with constraints, a minor modification is necessary
        if :math:`\sigma` is constrained to be zero. Instead of using
        :math:`\pm\infty` as the bounds on the support, you must use
        :math:`\alpha_-` (the lower bound on :math:`\alpha`) and
        :math:`\beta_+` (the upper bound on :math:`\beta`); also, the
        leftmost bucket must be closed (i.e., include
        :math:`\alpha_-`). The reason is that if you were to construct
        the buckets using :math:`-\infty` and you had a data point equal
        to :math:`\alpha_-` then you'd get a bucket that contains zero
        probability no matter the parameters. Even worse, that data
        point would get put in this bucket, sending the product of
        spacings to zero. Similarly, if you used :math:`\infty` and you
        had a data point equal to :math:`\beta_+` then you'd get another
        zero probabiliy bucket. The extra data point at the top of the
        support will get put in this bucket and again send the product
        of spacings to zero. These situations can be fairly common in
        practice due to rounding. Thus, in constrained problems where
        :math:`\sigma` is zero we construct the buckets using
        :math:`\alpha_-` and :math:`\beta_+`.

        References
        ----------
        .. [1] Cheng, R. C. H., & Amin, N. A. K. (1983). Estimating
           Parameters in Continuous Univariate Distributions with a
           Shifted Origin. Journal of the Royal Statistical Society.
           Series B (Methodological), 45(3), 394-403.

        .. [2] Ranneby, B. (1984). The Maximum Spacing Method. An
           Estimation Method Related to the Maximum Likelihood Method.
           Scandinavian Journal of Statistics, 11(2), 93-112.

        .. [3] Cheng, R. C. H., & Traylor, L. (1995). Non-Regular
           Maximum Likelihood Problems. Journal of the Royal Statistical
           Society. Series B (Methodological), 57(1), 3-44.

        .. [4] Titterington, D. M. (1985). Comment on “Estimating
           Parameters in Continuous Univariate Distributions.” Journal
           of the Royal Statistical Society. Series B (Methodological),
           47(1), 115-116.

        Examples
        --------
        Use :py:meth:`fit` to estimate the distribution from data:

        .. code:: python

           >>> NoisyQuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ... )
           NoisyQuadraticDistribution(...)

        When fitting a noisy quadratic distribution to the results from
        random search, you'll typically restrict it to be *convex* when
        *minimizing* and *concave* when *maximizing*. You can accomplish
        this with ``constraints``:

        .. code:: python

           >>> minimize = True  # If random search minimized / maximized
           >>> NoisyQuadraticDistribution.fit(
           ...   ys=[5.0, 2.9, 5.1, 6.8, 3.2],
           ...   constraints={"convex": True if minimize else False},
           ... )
           NoisyQuadraticDistribution(...)

        The noisy quadratic distribution approximates the score distribution's
        left tail when minimizing and right tail when maximizing. You
        can fit to only the tail of your data using ``limits``:

        .. code:: python

           >>> threshold = 5.  # The maximum cross-entropy to consider
           >>> NoisyQuadraticDistribution.fit(
           ...   ys=[5.0, 2.9, 5.1, 6.8, 3.2],  # 5.1 & 6.8 are censored
           ...   limits=(
           ...     (-np.inf, threshold)  # minimize: fit the left tail
           ...     if minimize else
           ...     (threshold, np.inf)   # maximize: fit the right tail
           ...   ),
           ...   constraints={"convex": True if minimize else False},
           ... )
           NoisyQuadraticDistribution(...)

        You could also censor both tails if necessary:

        .. code:: python

           >>> NoisyQuadraticDistribution.fit(
           ...   ys=[5.0, 2.9, 5.1, 6.8, 3.2],
           ...   limits=(1., 10.),
           ... )
           NoisyQuadraticDistribution(...)

        Finally, you can use ``constraints`` to bound any of the
        parameters in case you have some extra information. For example,
        if you knew a bound on the average performance of the best
        hyperparameters, you might constrain ``a`` (when minimizing) or
        ``b`` (when maximizing):

        .. code:: python

           >>> min_accuracy, max_accuracy = 0., 1.
           >>> NoisyQuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ...   constraints=(
           ...     {"a": (min_accuracy, max_accuracy)}
           ...     if minimize else
           ...     {"b": (min_accuracy, max_accuracy)}
           ...   ),
           ... )
           NoisyQuadraticDistribution(...)

        Or, you might know that the random search used 3
        hyperparameters, so the effective number of hyperparameters
        (``c``) can be at most that:

        .. code:: python

           >>> n_hyperparameters = 3
           >>> NoisyQuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ...   constraints={"c": (1, n_hyperparameters)},
           ... )
           NoisyQuadraticDistribution(...)

        You could also fix ``c`` (or ``a``, ``b``, ``o``, or ``convex``)
        to a particular value:

        .. code:: python

           >>> NoisyQuadraticDistribution.fit(
           ...   ys=[0.59, 0.86, 0.94, 0.81, 0.68, 0.90, 0.93, 0.75],
           ...   constraints={"c": 1},
           ... )
           NoisyQuadraticDistribution(...)

        Of course, you can mix and match all of these ideas together as
        desired.
        """
        # Validate arguments.
        ys = np.array(ys)
        if len(ys.shape) != 1:
            raise ValueError(f"ys must be a 1D array, not {len(ys.shape)}D.")
        if len(ys) == 0:
            raise ValueError("ys must be non-empty.")
        if limits[0] == -np.inf and np.any(np.isneginf(ys)):
            raise ValueError("ys must not contain -inf unless it is censored.")
        if limits[1] == np.inf and np.any(np.isposinf(ys)):
            raise ValueError("ys must not contain inf unless it is censored.")
        if np.any(np.isnan(ys)):
            raise ValueError("ys must not contain NaN.")
        if np.issubdtype(ys.dtype, np.integer):
            # Only cast ys if it has an integer data type, otherwise
            # preserve its precision which we'll need later in order
            # to decide how much to round the data.
            ys = ys.astype(float)

        limits = np.array(limits)
        if len(limits.shape) != 1:
            raise ValueError("limits must be a 1D sequence.")
        if len(limits) != 2:
            raise ValueError("limits must be a pair.")
        if not np.all(np.isreal(limits)):
            raise TypeError("limits must only contain floats.")
        if np.any(np.isnan(limits)):
            raise ValueError("limits cannot contain NaN values.")
        if limits[0] >= limits[1]:
            raise ValueError(
                "limits must be a proper (left-open) interval. The"
                " lower bound cannot equal or exceed the upper bound.",
            )

        constraints = dict(constraints) if constraints is not None else {}
        for parameter, constraint in constraints.items():
            constraint = np.array(constraint)[()]
            if parameter in {"a", "b", "c", "o"}:
                # Check invaraints for a, b, c, and o.
                if not np.isscalar(constraint) and constraint.shape != (2,):
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a scalar or a pair.",
                    )
                if not np.all(np.isreal(constraint)):
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a float or a pair of floats.",
                    )
                if np.any(np.isnan(constraint)):
                    raise ValueError(
                        f"The constraint for {parameter} cannot"
                        f" contain NaNs.",
                    )
                if constraint.shape == (2,) and constraint[0] > constraint[1]:
                    raise ValueError(
                        f"The constraint for {parameter} cannot have a lower"
                        f" bound greater than the upper bound.",
                    )
                # Check invariants that only apply to c.
                if parameter == "c":
                    if np.any(constraint % 1 != 0):
                        raise ValueError(
                            "The constraint for c must be either an integer"
                            " or a pair of integers.",
                        )
                    if np.isscalar(constraint):
                        if constraint < cls.C_MIN or constraint > cls.C_MAX:
                            raise ValueError(
                                f"The constraint for c fixes its value"
                                f" outside of {cls.C_MIN} to"
                                f" {cls.C_MAX} but only values within"
                                f" that range are supported.",
                            )
                    else:
                        if (
                                constraint[0] > cls.C_MAX
                                or constraint[1] < cls.C_MIN
                        ):
                            raise ValueError(
                                f"The constraint for c excludes all"
                                f" supported values. Only values of c"
                                f" between {cls.C_MIN} and {cls.C_MAX}"
                                f" are supported.",
                            )
                        if (
                                constraint[0] < cls.C_MIN
                                or constraint[1] > cls.C_MAX
                        ):
                            warnings.warn(
                                f"The constraint for c includes values"
                                f" outside of {cls.C_MIN} to"
                                f" {cls.C_MAX}, but only values within"
                                f" that range are supported. Consider"
                                f" revising the constraint to only"
                                f" include values within that range.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                # Check invariants that only apply to o.
                if parameter == "o":
                    if np.any(constraint < 0.):
                        raise ValueError(
                            "The constraint for o must be either a"
                            " non-negative number or a pair of"
                            " non-negative numbers.",
                        )
            elif parameter == "convex":
                if not np.isscalar(constraint) and len(constraint.shape) != 1:
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a scalar or 1D list.",
                    )
                if not np.isscalar(constraint) and len(constraint) == 0:
                    raise ValueError(
                        f"The constraint for {parameter} must not be"
                        f" empty.",
                    )
                if constraint.dtype != bool:
                    raise TypeError(
                        f"The constraint for {parameter} must be"
                        f" either a bool or a list of bools.",
                    )
                if np.any(np.unique(constraint, return_counts=True)[1] > 1):
                    raise ValueError(
                        f"The constraint for {parameter} cannot contain"
                        f" duplicate elements.",
                    )
            else:
                raise ValueError(
                    f"constraints contains an unrecognized key ({parameter}),"
                    f" all keys must be parameters of the class.",
                )

        generator = (
            generator
            if generator is not None else
            opda.random.DEFAULT_GENERATOR
        )

        # Censor the data.
        limit_lower, limit_upper = limits

        n = len(ys)
        n_lower, n_upper = np.sum(ys <= limit_lower), np.sum(ys > limit_upper)
        ys_observed = ys[(limit_lower < ys) & (ys <= limit_upper)]

        if n < 3:
            raise ValueError(
                "ys must contain at least three data points.",
            )

        if len(ys_observed) == 0:
            raise ValueError(
                "All ys are censored because they fall outside of"
                " limits. At least some ys must be observed.",
            )

        # NOTE: When observations are censored in the left or the right
        # tail, the min or max is unavailable. In those cases, use
        # limit_lower as the min and limit_upper as the max as those are
        # the smallest / largest values we know are compatible with the
        # distribution. You could instead use the smallest / largest
        # observed values; however, this approach fails when all values
        # but one are censored because then the min and max are equal.
        y_min = np.min(ys_observed) if n_lower == 0 else limit_lower
        i_min = 1 if n_lower == 0 else n_lower          # the rank of y_min
        y_max = np.max(ys_observed) if n_upper == 0 else limit_upper
        j_max = n if n_upper == 0 else n - n_upper + 1  # the rank of y_max

        # backwards compatibility (scipy < 1.11)
        #
        # This method implements point constraints (fixing a, b, c, or o
        # to a particular value) differently than interval constraints
        # (restricting a, b, c, or o to a range). This is necessary when
        # scipy < 1.11 because optimize.differential_evolution
        # encounters zero division errors and uses too big of a
        # population when a parameter has equal upper and lower bounds
        # (see https://github.com/scipy/scipy/issues/17788). After
        # dropping support for scipy < 1.11, try to simplify this method
        # by replacing point constraints ({"a": p}) with interval
        # constraints ({"a": (p, p)}) and sharing the rest of the logic.

        # Handle constraints.
        a_constraint = constraints.get("a", [-np.inf, np.inf])
        a = a_constraint if np.isscalar(a_constraint) else None
        b_constraint = constraints.get("b", [-np.inf, np.inf])
        b = b_constraint if np.isscalar(b_constraint) else None
        c_constraint = constraints.get("c", [cls.C_MIN, cls.C_MAX])
        c = c_constraint if np.isscalar(c_constraint) else None
        o_constraint = constraints.get("o", [0., np.inf])
        o = o_constraint if np.isscalar(o_constraint) else None

        cs = range(
            max(cls.C_MIN, c_constraint[0]),
            min(cls.C_MAX, c_constraint[1]) + 1,
        ) if c is None else [c]
        convexs = (
            [False, True] if "convex" not in constraints else
            [constraints["convex"]] if np.isscalar(constraints["convex"]) else
            constraints["convex"]
        )

        # Check for error and warning conditions.
        if y_min == y_max and n_upper == 0 and n_lower == 0\
           and (len(cs) > 1 or len(convexs) > 1):
            warnings.warn(
                "Parameters might be unidentifiable. All ys are equal,"
                " suggesting the distribution is a point mass. The"
                " distribution is a point mass whenever a = b and o = 0,"
                " making c and convex unidentifiable. If appropriate,"
                " use the constraints parameter to specify c and convex.",
                RuntimeWarning,
                stacklevel=2,
            )

        if a is not None and np.max(o_constraint) <= 0. and a > y_min:
            raise ValueError(
                "constraints must not fix o to be zero and a to be"
                " greater than the least observation (or the lower"
                " limit if the least observation is censored).",
            )
        if a is None and np.max(o_constraint) <= 0. and a_constraint[0] > y_min:
            raise ValueError(
                "constraints must not constrain o to be zero and a to"
                " be greater than the least observation (or the lower"
                " limit if the least observation is censored).",
            )

        if b is not None and np.max(o_constraint) <= 0. and b < y_max:
            raise ValueError(
                "constraints must not fix o to be zero and b to be"
                " less than the greatest observation (or the upper"
                " limit if the greatest observation is censored).",
            )
        if b is None and np.max(o_constraint) <= 0. and b_constraint[1] < y_max:
            raise ValueError(
                "constraints must not constrain o to be zero and b to"
                " be less than the greatest observation (or the upper"
                " limit if the greatest observation is censored).",
            )

        # Fit the distribution.
        best_loss = np.inf
        best_parameters = None
        for convex in convexs:
            # Determine the search space.
            bounds = []
            integrality = []

            # NOTE: We must bound a and b for our optimization. We need
            # data-dependent bounds that adapt gracefully across scales.
            #   Imagine o = 0, then if we knew (b - a) we could use:
            # y_min - (b - a) <= a <= b <= y_max + (b - a). We'll
            # bound (b - a) as follows. Consider the distribution of
            # (Y_(j) - Y_(i)) / (b - a), it depends only on c, o = 0,
            # and convex. Let w be the 1e-9 quantile of this
            # distribution, then with high probability:
            #
            #   (Y_(j) - Y_(i)) / (b - a) > w
            #
            # or equivalently:
            #
            #   1/w (Y_(j) - Y_(i)) > b - a
            #
            # To approximate w, consider F(Y_(j)) - F(Y_(i)). It has the
            # same distribution as the difference of the i'th and j'th
            # order statistics of the uniform distribution, or
            # Beta(j - i, n - (j - i) + 1). The lower 1e-9 quantile of
            # this distribution bounds the probability mass separating
            # these order statistics. We could then seek the shortest
            # interval containing at least this probability mass in
            # Q(0, 1, c, 0, convex). The length of that interval is a
            # lower bound for w, but it's too conservative. Instead, we
            # approximate w using the length of the equal-tailed
            # interval containing that probability mass.
            #   Intuitively, as o increases (Y_(j) -  Y_(i)) / (b - a)
            # will tend to be larger. Specifically, y_min will be
            # farther left and y_max farther right due to the sample's
            # increased spread. Thus, the numerator gets larger while
            # the denominator remains the same. Simulations seem to
            # confirm this intuition. So, we obtain bounds by assuming
            # o = 0 and then applying them in the general case.
            p = stats.beta(j_max - i_min, n - (j_max - i_min) + 1).ppf(1e-9)
            w = max(
                1 / np.diff(
                    cls(0, 1, c, 0, convex).ppf([0.5 - p/2, 0.5 + p/2]),
                )[0]
                for c in cs
            )

            a_bounds = (
                # Intersect the constraint and default bounds.
                max(y_min - w * (y_max - y_min), a_constraint[0]),
                min(y_max + w * (y_max - y_min), a_constraint[1]),
            ) if a is None else (a, a)

            if a_bounds[0] > a_bounds[1]:
                # The intersection of the default bounds and the
                # constraint is empty.
                raise exceptions.OptimizationError(
                    "The constraint on a excludes all promising values"
                    " for it. Consider relaxing the constraint on a or"
                    " re-examining your data.",
                )

            # backwards compatibility (scipy < 1.11)
            if scipy_version < (1, 11) and a_bounds[0] == a_bounds[1]:
                # NOTE: In scipy < 1.11, optimize.differential_evolution
                # encounters a zero division bug when upper and lower
                # bounds are equal. Avoid it by passing a directly.
                a = a_bounds[0]

            if a is None:
                bounds.append(a_bounds)
                integrality.append(False)

            b_bounds = (
                # Intersect the constraint and default bounds.
                max(y_min - w * (y_max - y_min), b_constraint[0]),
                min(y_max + w * (y_max - y_min), b_constraint[1]),
            ) if b is None else (b, b)

            if b_bounds[0] > b_bounds[1]:
                # The intersection of the default bounds and the
                # constraint is empty.
                raise exceptions.OptimizationError(
                    "The constraint on b excludes all promising values"
                    " for it. Consider relaxing the constraint on b or"
                    " re-examining your data.",
                )

            # backwards compatibility (scipy < 1.11)
            if scipy_version < (1, 11) and b_bounds[0] == b_bounds[1]:
                # NOTE: In scipy < 1.11, optimize.differential_evolution
                # encounters a zero division bug when upper and lower
                # bounds are equal. Avoid it by passing b directly.
                b = b_bounds[0]

            if b is None:
                bounds.append(b_bounds)
                integrality.append(False)

            c_bounds = (
                # Intersect the constraint and default bounds.
                max(cls.C_MIN, c_constraint[0]),
                min(cls.C_MAX, c_constraint[1]),
            ) if c is None else (c, c)

            if c_bounds[0] > c_bounds[1]:
                # The intersection of the default bounds and the
                # constraint is empty.
                raise exceptions.OptimizationError(
                    "The constraint on c excludes all promising values"
                    " for it. Consider relaxing the constraint on c or"
                    " re-examining your data.",
                )

            # backwards compatibility (scipy < 1.11)
            if scipy_version < (1, 11) and c_bounds[0] == c_bounds[1]:
                # NOTE: In scipy < 1.11, optimize.differential_evolution
                # encounters a zero division bug when upper and lower
                # bounds are equal. Avoid it by passing c directly.
                c = c_bounds[0]

            if c is None:
                bounds.append(c_bounds)
                integrality.append(True)

            # NOTE: Now we need bounds for o. For a lower bound we can
            # use 0. For an upper bound, consider o / (Y_(j) - Y_(i)).
            # Let v be the 1-1e-9 quantile for this quantity, then with
            # high probability:
            #
            #   o / (Y_(j) - Y_(i)) < v
            #
            # or equivalently:
            #
            #   o < v (Y_(j) - Y_(i))
            #
            # Typically, the range of Y = Q + E will be greater than
            # that of E alone. Thus, we obtain a lower bound on v
            # by ignoring Q and assuming the data are normally
            # distributed. For the normal, the standard deviation
            # cancels the scale so we need only consider the standard
            # normal distribution. To approximate v, consider
            # F(Y_(j)) - F(Y_(i)). As before, it has the distribution of
            # the difference of the uniform's i'th and j'th order
            # statistics. The lower 1e-9 quantile of this distribution
            # provides a lower bound on how much probability mass can
            # separate these order statistics. Since the standard normal
            # is symmetric with a mode at 0, the equal-tailed interval
            # is also the highest probability density interval, so it
            # gives a lower bound on Y_(j) - Y_(i) and thus an upper
            # bound on o / (Y_(j) - Y_(i)) = v. Simulations with the
            # standard normal show that this bound is not too
            # conservative.
            v = 1 / np.diff(stats.norm(0., 1.).ppf([0.5 - p/2, 0.5 + p/2]))[0]

            o_bounds = (
                # Intersect the constraint and default bounds.
                max(0., o_constraint[0]),
                min(v * (y_max - y_min), o_constraint[1]),
            ) if o is None else (o, o)

            if o_bounds[0] > o_bounds[1]:
                # The intersection of the default bounds and the
                # constraint is empty.
                raise exceptions.OptimizationError(
                    "The constraint on o excludes all promising values"
                    " for it. Consider relaxing the constraint on o or"
                    " re-examining your data.",
                )

            # backwards compatibility (scipy < 1.11)
            if scipy_version < (1, 11) and o_bounds[0] == o_bounds[1]:
                # NOTE: In scipy < 1.11, optimize.differential_evolution
                # encounters a zero division bug when upper and lower
                # bounds are equal. Avoid it by passing o directly.
                o = o_bounds[0]

            if o is None:
                bounds.append(o_bounds)
                integrality.append(False)

            # Define the loss.
            if method == "maximum_spacing":
                # Maximum spacing estimation is sensitive to closely
                # spaced observations. Two observations that represent
                # the same point but differ due to floating point errors
                # could noticeably impact the estimate's quality. To
                # prevent this, round the data a tiny bit before
                # defining the buckets in order to group such
                # observations together.
                #
                # In practice, this rounding should improve the
                # estimate, but it could cause limit_lower or
                # limit_upper to shift and thus censor a point that's
                # technically within the limits (though extremely close
                # to the boundary).
                finfo = np.finfo(ys_observed.dtype)
                # For rounding, use 3 fewer digits than the coarsest
                # precision of any observed point and clip the number of
                # decimals to stay within those representable by the
                # floating point format.
                decimals = - np.clip(
                    np.log10(np.max(np.abs(np.spacing(ys_observed)))),
                    np.log10(finfo.smallest_normal),
                    np.log10(finfo.max),
                ).astype(int) - 3
                zs, ks = np.unique(np.round(
                    np.concatenate([
                        # lower bound on the support
                        [-np.inf] if np.max(o_bounds) > 0. else [a_bounds[0]],
                        # lower limit (for the censoring)
                        [limit_lower] if n_lower > 0 else [],
                        # order statistics
                        ys_observed,
                        # upper limit (for the censoring)
                        [limit_upper] if n_upper > 0 else [],
                        # upper bound on the support
                        [np.inf] if np.max(o_bounds) > 0. else [b_bounds[1]],
                    ]),
                    decimals=decimals,
                ), return_counts=True)

                ks = ks[1:]  # The leftmost count corresponds to no bucket.
                if n_lower > 0:
                    # Set the count for the left tail bucket.
                    ks[0] = n_lower
                if n_upper > 0:
                    # Remove the extra count created by defining the
                    # right tail bucket.
                    ks[-2] -= 1
                    # Set the count for the right tail bucket with the
                    # extra count at the top of the support.
                    ks[-1] = (n_upper + 1)

                def loss(parameters):
                    # Attempt to initialize the distribution. If the
                    # parameters are invalid, return infinite loss.
                    try:
                        dist = cls(
                            a=(
                                a             if a is not None else
                                parameters[0]
                            ),
                            b=(
                                b             if b is not None else
                                parameters[
                                    0 + (a is None)
                                ]
                            ),
                            c=(
                                c             if c is not None else
                                parameters[
                                    0 + (a is None) + (b is None)
                                ]
                            ),
                            o=(
                                o             if o is not None else
                                parameters[
                                    0 + (a is None) + (b is None) + (c is None)
                                ]
                            ),
                            convex=convex,
                        )
                    except ValueError:
                        # The parameters are invalid (e.g., a > b).
                        return np.inf

                    # In practice, the cdf method is not monotonic due
                    # to small approximation errors. Non-monotonicity
                    # can create negative spacings, which become NaNs
                    # after taking the log. Sorting the cdf values both
                    # fixes this non-monotonicity and reduces the
                    # average approximation error, improving the fit.
                    #
                    # Since the cdf values will be perfectly or almost
                    # perfectly sorted, use an in-place timsort (i.e.,
                    # kind="stable") which can take advantage of this.
                    ps = dist.cdf(zs)
                    ps.sort(kind="stable")

                    return - np.sum(
                        # Instead of the raw grouped negative
                        # log-likelihood, divide the sum by (n + 1) and
                        # divide the spacings by ks / (n + 1). This
                        # modification makes the loss an estimator of
                        # the KL-divergence and, more importantly, keeps
                        # the loss's scale constant across sample sizes
                        # which improves the optimization.
                        #
                        # For more discussion, see the M_T2 objective in
                        # "Alternatives to maximum likelihood estimation
                        # based on spacings and the Kullback-Leibler
                        # divergence" (Ekstrom, 2008).
                        ks * np.log(
                            np.diff(ps) * (n + 1) / ks,
                        ) / (n + 1),
                        where=ks > 0,
                    )
            else:
                raise ValueError(
                    'method must be "maximum_spacing".',
                )

            # The optimizer can have difficulty finding the optimum, so
            # we provide some initial estimates to make it more robust.
            #
            # For the initial estimates, we reparametrize the
            # distribution in terms of a, b, c, s, and convex where
            # s = o / (b - a) is a shape parameter. The initial
            # estimates then use a grid over c and s. For a and b, we
            # use estimates of the form:
            #
            #   a = y_min - w_a (y_max - y_min)
            #   b = y_max + w_b (y_max - y_min)
            #
            # Ideally, w_a would equal (y_min - a) / (y_max - y_min) and
            # w_b would equal (b - y_max) / (y_max - y_min) because then
            # our estimates would equal a and b exactly. Since y_min and
            # y_max include the location, and (y_max - y_min) includes
            # the scale, the distributions of these quantities depend
            # only on c, s = o / (b - a), and convex. Thus, we can treat
            # w_a and w_b as if the underlying distribution was
            # Q(0, 1, c, s, convex). Let F^-1 be its quantile function.
            # To obtain estimates for w_a and w_b, we replace y_min and
            # y_max with F^-1(i/(n+1)) and F^-1(j/(n+1)) where i and j
            # are the ranks of each. To obtain multiple estimates, we
            # can replace i/(n+1) and j/(n+1) with other quantiles of
            # F(y_min) and F(y_max), which are distributed according to
            # Beta(i, n - i + 1) and Beta(j, n - j + 1). Finally,
            # plugging w_a and w_b into our original equations for a and
            # b yields our final estimates.
            initial_population = []
            ss = (
                [1e-6, 1e-4, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2]
                if a is None or b is None or o is None else
                [1.]  # Use at least one s so that we include the c estimate.
            )
            for c_candidate in cs:
                for s in ss:
                    initial_estimates = collections.defaultdict(list)
                    ds = (
                        # When fitting a and b, use 2 estimates for
                        # each since there are 7 s estimates for a
                        # total of 2 * 2 * 7 = 28 estimates.
                        [-0.2, 0.2]
                        if a is None and b is None else
                        # When fitting only a or only b use more initial
                        # estimates for it to ensure the initial population
                        # has a reasonable size.
                        [-0.2, -0.1, 0.1, 0.2]
                    )
                    for d in ds:
                        std_dist = cls(0., 1., c_candidate, s, convex)
                        std_y_min = std_dist.ppf(
                            stats.beta(i_min, n - i_min + 1).ppf(0.5 + d),
                        )
                        std_y_max = std_dist.ppf(
                            stats.beta(j_max, n - j_max + 1).ppf(0.5 + d),
                        )
                        if a is None:
                            initial_estimates["a"].append(np.clip(
                                # Clip the estimate to obey the constraint.
                                y_min
                                - std_y_min / (std_y_max - std_y_min)
                                  * (y_max - y_min),
                                a_bounds[0],
                                a_bounds[1],
                            ))
                        if b is None:
                            initial_estimates["b"].append(np.clip(
                                # Clip the estimate to obey the constraint.
                                y_max
                                + (1 - std_y_max) / (std_y_max - std_y_min)
                                  * (y_max - y_min),
                                b_bounds[0],
                                b_bounds[1],
                            ))
                    if c is None:
                        initial_estimates["c"].append(c_candidate)
                    if o is None:
                        # Estimate o as s times an estimate of b - a.
                        std_dist = cls(0., 1., c_candidate, s, convex)
                        std_y_min = std_dist.ppf(i_min / (n + 1))
                        std_y_max = std_dist.ppf(j_max / (n + 1))

                        initial_estimates["o"].append(np.clip(
                            # Clip the estimate to obey the constraint.
                            s * ((
                                y_max
                                + (1 - std_y_max) / (std_y_max - std_y_min)
                                  * (y_max - y_min)
                                if b is None else
                                b
                            ) - (
                                y_min
                                - std_y_min / (std_y_max - std_y_min)
                                  * (y_max - y_min)
                                if a is None else
                                a
                            ))
                            if a is None or b is None or a < b else
                            # Since a == b, the distribution is just a
                            # normal and b - a = 0 so we can't multiply
                            # s by (b - a). Instead, multiply s by an
                            # estimate of the standard deviation.
                            #
                            # To estimate the standard deviation,
                            # consider o / (Y_(j) - Y_(i)). It's
                            # distribution is independent of the mean
                            # and variance. For the standard normal, we
                            # could approximate Y_(i) and Y_(j) by the
                            # i/(n+1)'th and j/(n+1)'th quantiles, z_i
                            # and z_j. Then o / (Y_(j) - Y_(i)) is
                            # approximately c = 1/(z_j - z_i) so we
                            # estimate o by c * (Y_(j) - Y_(i)).
                            s * (y_max - y_min) / (np.diff(
                                stats.norm(0., 1.).ppf([
                                    i_min / (n + 1),
                                    j_max / (n + 1),
                                ]),
                            )[0]),
                            o_bounds[0],
                            o_bounds[1],
                        ))

                    initial_population.extend(
                        itertools.product(*initial_estimates.values()),
                    )

            # Select the best estimates for the initial population.
            with np.errstate(divide="ignore", invalid="ignore"):
                _, initial_population = zip(*sorted([
                    (loss(parameters), parameters)
                    for parameters in initial_population
                ])[:90])

            # Optimize the loss to compute the estimate.
            if len(bounds) > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    result = optimize.differential_evolution(
                        loss,
                        bounds=bounds,
                        init=initial_population,
                        integrality=integrality,
                        strategy="rand2exp",
                        mutation=(0.5, 1.5),
                        recombination=0.9,
                        updating="immediate",
                        tol=0.001,
                        polish=True,
                        vectorized=False,
                        workers=1,
                        seed=generator,
                    )
                parameters = result.x
                curr_loss = result.fun
            else:  # There are no parameters to fit
                parameters = []
                curr_loss = loss(parameters)

            # Check if the current value of convex is the best so far.
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_parameters = {
                    "a": (
                        a             if a is not None else
                        parameters[0]
                    ),
                    "b": (
                        b             if b is not None else
                        parameters[
                            0 + (a is None)
                        ]
                    ),
                    "c": (
                        c             if c is not None else
                        parameters[
                            0 + (a is None) + (b is None)
                        ]
                    ),
                    "o": (
                        o             if o is not None else
                        parameters[
                            0 + (a is None) + (b is None) + (c is None)
                        ]
                    ),
                    "convex": convex,
                }

        if not np.isfinite(best_loss):
            raise exceptions.OptimizationError(
                "fit failed to find parameters with finite loss.",
            )

        if best_parameters.get("c") == 2 and len(convexs) > 1:
            warnings.warn(
                "Parameters might be unidentifiable. The fit found"
                " c = 2. When c = 2, convex is unidentifiable because"
                " either value for convex gives the same distribution."
                " If appropriate, use the constraints parameter to"
                " specify convex.",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(**best_parameters)

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
        # ``loc`` (u) and standard deviation ``scale`` (o). While [1]
        # provides a derivation only for integer moments, the formula is
        # also valid for fractional moments. The recursive formula can
        # be used to step up or down to the desired moment from a pair
        # of base moments.
        #
        # [1]: Robert L. Winkler, Gary M. Roodman, Robert R. Britney,
        # (1972) The Determination of Partial Moments. Management
        # Science 19(3):290-296.

        # Validate arguments.
        if scale < 0:
            raise ValueError(f"scale (scale={scale}) must be greater than 0.")
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
            if k == -0.5 and scale < 5e-2:
                # For small scales, compute the -0.5 partial moment
                # directly (using the Chebyshev approximation).
                return (
                    -0.5,
                    None,
                    self._partial_fractional_normal_moment(loc, scale, -0.5),
                )
            if k == -0.5 and scale >= 5e-2:
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
        # standard deviation ``scale``. In other words, it computes:
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
        # ``loc`` (u) and standard deviation ``scale`` (o).
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

        # Check for a piecewise minimax polynomial approximation.
        for approximation in _APPROXIMATIONS.get(k, []):
            if scale < approximation["min_scale"]:
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
        lo = np.clip(loc - 6 * scale, 0., 1. - scale)
        hi = np.clip(loc + 6 * scale, scale, 1.)
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
            if scale >= min_scale
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
  The minimum scale (i.e., the standard deviation) for which to use the
  approximation.
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
