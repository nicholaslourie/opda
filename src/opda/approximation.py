"""Approximation of univariate functions."""

import warnings

import numpy as np
from scipy import special

from opda import exceptions


def lagrange_interpolate(xs, ys):
    r"""Interpolate ``xs`` and ``ys`` with a polynomial.

    Interpolate the points with a polynomial, :math:`p(x)`, such that:

    .. math::

       \forall i, p(x_i) = y_i

    If there are n+1 points, then the polynomial will have degree n.

    Parameters
    ----------
    xs : 1D array of finite floats, required
        The x values (asbscissas) to interpolate.
    ys : 1D array of finite floats, required
        The y values (ordinates) to interpolate.

    Returns
    -------
    function
        A function that evaluates the interpolating polynomial on arrays
        of floats, entrywise.
    """
    # Validate the arguments.
    xs = np.array(xs)
    if len(xs) == 0:
        raise ValueError("xs must be non-empty.")
    if np.any(~np.isfinite(xs)):
        raise ValueError("xs must contain only finite floats.")

    ys = np.array(ys)
    if len(ys) == 0:
        raise ValueError("ys must be non-empty.")
    if np.any(~np.isfinite(ys)):
        raise ValueError("ys must contain only finite floats.")

    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")

    # Compute the interpolating polynomial.
    ws = xs[:, None] - xs[None, :]
    np.fill_diagonal(ws, 1.)
    ws = 1. / np.prod(ws, axis=1)

    xs_orig = xs
    ys_orig = ys

    def lagrange_polynomial(xs):
        # Handle both scalars and arrays.
        xs = np.array(xs)
        shape = xs.shape

        xs = np.atleast_1d(xs)

        # Compute y with the first form of the barycentric
        # interpolation formula.
        xs_minus_xs_orig = xs[..., :, None] - xs_orig[..., None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ys = (
                np.prod(xs_minus_xs_orig, axis=-1)
                * np.sum(ys_orig * ws / xs_minus_xs_orig, axis=-1)
            )
        # Fix any ys corresponding to original points.
        *i, j = np.nonzero(xs[..., :, None] == xs_orig[..., None, :])
        ys[tuple(i)] = ys_orig[j]

        return ys.reshape(shape)

    return lagrange_polynomial


def remez(f, a, b, n, *, atol=None):
    """Return the reference for the minimax polynomial.

    Use the Remez exchange algorithm to compute the reference
    corresponding to the minimax polynomial approximation to ``f``. The
    reference is the set of x values at which the minimax polynomial
    achieves its equioscillating maximal error.

    The reference for the minimax polynomial is primarily useful as an
    input for other approximation theoretic operations. Most users will
    prefer instead to use other functions that implement these
    operations such as :py:func:`minimax_polynomial_approximation` to
    evaluate the minimax polynomial or
    :py:func:`minimax_polynomial_coefficients` to compute its
    coefficients.

    For background on the Remez algorithm and approximation theory, see
    "Approximation theory and methods" [1]_.

    Parameters
    ----------
    f : function, required
        The function to approximate. The function should map floats to
        floats.
    a : finite float, required
        The lower end point of the interval over which to approximate
        ``f``.
    b : finite float, required
        The upper end point of the interval over which to approximate
        ``f``.
    n : non-negative int, required
        The degree of the polynomial approximation.
    atol : non-negative float or None, optional
        The absolute tolerance to use for stopping the computation. The
        algorithm ends when the current approximation has a maximum
        error within ``atol`` of the best possible approximation. If
        ``None``, then it will be set to a multiple of machine epsilon.

        Only set this value if you encounter numerical or optimization
        issues. In that case, raise the value until the numerical issues
        disappear and as long as the new absolute tolerance represents
        an acceptable level of error above the best approximation.

    Returns
    -------
    1D array of floats from a to b inclusive
        The reference, or x values where the minimax polynomial
        achieves equioscillating error.
    1D array of floats
        The y values of ``f`` evaluated on the reference.
    non-negative float
        The worst case absolute error of the minimax polynomial
        approximation to ``f`` from ``a`` to ``b``.

    See Also
    --------
    minimax_polynomial_approximation :
        Return a function representing the minimax polynomial.
    minimax_polynomial_coefficients :
        Return the minimax polynomial's coefficients.

    References
    ----------
    .. [1] Powell, M.J.D., "Approximation theory and methods"
           (1981). Cambridge University Press.
    """
    # Validate the arguments.
    if not callable(f):
        raise TypeError("f must be callable.")

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

    n = np.array(n)[()]
    if not np.isscalar(n):
        raise ValueError("n must be a scalar.")
    if n % 1 != 0:
        raise ValueError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")

    if atol is not None and atol < 0:
        raise ValueError("atol must be non-negative.")

    if a > b:
        raise ValueError("a must be less than or equal to b.")

    # Define constants.
    machine_epsilon = np.spacing(1.)
    phi = (1 + 5**0.5)/2

    # Set arguments to appropriate defaults.
    if atol is None:
        # NOTE: The factor, 256, was chosen empirically to avoid
        # numerical issues when f can be fit exactly.
        atol = 256. * machine_epsilon

    ns = np.arange(n + 2)

    # NOTE: Initialize the reference to the Chebyshev nodes. The
    # Chebyshev nodes have good approximation theoretic properties
    # that guarantee faster convergence. See Section 8.4 of
    # "Approximation theory and methods" by M.J.D. Powell.
    rs = a + (b - a) * 0.5 * (1 - np.cos(np.pi * ns/(n+1)))

    # Iteratively improve the reference via the Remez exchange algorithm.
    h_curr = -np.inf
    for _ in range(25):
        # Step 1: Interpolate the reference.

        # NOTE: The first step of the Remez exchange algorithm solves
        # the linear system:
        #
        #   a_0 + ... + a_n x_{0}^n   + e = f(x_0)
        #    .                 .                 .
        #    .                 .                 .
        #    .                 .                 .
        #   a_0 + ... + a_n x_{n+1}^n + e = f(x_{n+1})
        #
        # for a_0, ..., a_n, and e. The solution to this linear system
        # yields the best polynomial approximation of degree n to f on
        # the n+2 points x_0, ..., x_{n+1}.
        #   We can solve this linear system implicitly using only O(n^2)
        # operations via Lagrange interpolation [1]. Namely, let p0(x)
        # interpolate f and p1(x) interpolate (-1)^i on the first n+1
        # points. Then let p = p0 - h * p1. We have:
        #
        #   f(x_i) - p(x_i)
        #     = f(x_i) - (p0(x_i) - h * p1(x_i))
        #     = (f(x_i) - p0(x_i)) + h * p1(x_i)
        #     = h * (-1)^i
        #
        # Thus, p is the unique degree n polynomial with oscillating
        # error on the first n+1 points at a value equal to h. h can
        # then be chosen to guarantee the error on the last point is
        # also equal to h:
        #
        #   f(x_{n+1}) - p(x_{n+1}) = (-1)^{n+1} h
        #   =>
        #   h = (p0(x_{n+1}) - f(x_{n+1})) / (p1(x_{n+1}) + (-1)^n)
        #
        # Using barycentric Lagrange interpolation also improves the
        # numerical stability over solving the equations directly. The
        # first form of the barycentric interpolation formula is
        # preferred over the second because it is numerically stable for
        # all references, unlike the second (see Section 5, p. 38-39 of
        # [2]).
        #
        # [1]: https://en.wikipedia.org/wiki/Remez_algorithm#Detailed_discussion.
        # [2]: Trefethen, Lloyd N. Approximation Theory and
        #      Approximation Practice. Oxford, United Kingdom, Oxford
        #      University Press, 2013.

        p0 = lagrange_interpolate(rs[:-1], f(rs[:-1]))
        p1 = lagrange_interpolate(rs[:-1], (-1)**ns[:-1])
        h = (p0(rs[-1]) - f(rs[-1])) / (p1(rs[-1]) + (-1)**n)
        h_sign = (-1)**(h < 0)
        # NOTE: Don't use np.sign to compute h_sign because h_sign must
        # be either -1 or 1. Even when h is 0, either -1 or 1 must be
        # picked.

        p = lagrange_interpolate(rs[:-1], f(rs[:-1]) - h * (-1)**ns[:-1])
        # Evaluating a freshly interpolated polynomial on the points
        # and errors is about twice as fast as directly evaluating
        # p0 - h * p1. Since we evaluate p many times in the
        # optimization step, this speed up is worthwhile.


        # Step 2: Maximize the reference error.

        # NOTE: Two conditions are necessary to increase the leveled
        # reference error (abs(h)):
        #
        #   1. When the reference points are in increasing order,
        #      x_0 < x1 < ... < x_{n+1}, the errors must oscillate in sign:
        #      sign(f(x_i) - p(x_i)) = -sign(f(x_{i+1}) - p(x_{i+1})).
        #   2. The minimum absolute error of the new reference must be
        #      greater than the leveled error of the old reference:
        #      min_i |f(x_i) - p(x_i)| > |h|.
        #
        # In addition, we'd like to globally optimize the error function
        # so that we can get an estimate for the worst case error of the
        # approximation.
        #   The error function can have many forms, for example it can
        # have an arbitrarily large number of zeros and local optima. We
        # can't make assumptions like there's only one optimum between
        # each reference point. To guarantee the signs oscillate and the
        # error increases, we optimize each reference point's error in
        # the same direction and on non-overlapping intervals. That way,
        # the minimum error increases as long as no reference is a local
        # optimum, the sign of the error remains the same at each point,
        # and the reference points can't cross each other since they're
        # optimized in non-overlapping intervals.
        #   For maximally robust optimization, we use golden section
        # search. To guarantee the error increases, we include the old
        # reference point on the boundary of the search bracket. Since
        # we don't know which side of the reference has the new optimum,
        # we search brackets on either side of each reference point.
        #   To summarize: for each reference point we construct two
        # intervals, one on either side extending from the reference
        # point half-way to its neighbor. For the reference points on
        # either end, the approximation bounds (a and b) are considered
        # half-way to their (non-existent) neighbors. We then run golden
        # section search in each of these intervals and move each
        # reference point to the point from whichever interval has
        # higher error.

        # shape: (side, reference point, search point)
        search_points = np.zeros((2, len(rs), 4))
        # Set the left-most search point.
        search_points[0,   0, 0] = a
        search_points[0,  1:, 0] = (rs[:-1] + rs[1:]) / 2
        search_points[1,   :, 0] = rs
        # Set the right-most search point.
        search_points[0,   :, 3] = rs
        search_points[1, :-1, 3] = (rs[:-1] + rs[1:]) / 2
        search_points[1,  -1, 3] = b
        # Set the left and right inner search points.
        length = (search_points[:, :, 3] - search_points[:, :, 0]) / phi
        search_points[:, :, 1] = search_points[:, :, 3] - length
        search_points[:, :, 2] = search_points[:, :, 0] + length

        # Choose the number of search iterations so that the error is
        # optimized to machine precision.
        n_iters = int(np.ceil(np.log(machine_epsilon) / np.log(phi - 1)))
        for _ in range(n_iters):
            # Multiply each evaluation point by the sign of the error
            # for the corresponding reference point, that way we
            # minimize negative errors and maximize positive errors.
            #
            # NOTE: You can't just take the absolute value because you
            # can't assume all errors at the search points have the
            # same sign as the reference.
            errs = (h_sign * (-1)**ns)[None, :, None] * (
                f(search_points) - p(search_points)
            )

            left_higher = (
                np.max(errs[..., :2], axis=-1)
                > np.max(errs[..., 2:], axis=-1)
            )
            right_higher = ~left_higher

            search_points[left_higher, 2:4] = search_points[left_higher, 1:3]
            search_points[left_higher, 1] = search_points[left_higher, 3] - (
                search_points[left_higher, 3] - search_points[left_higher, 0]
            ) / phi

            search_points[right_higher, 0:2] = search_points[right_higher, 1:3]
            search_points[right_higher, 2] = search_points[right_higher, 0] + (
                search_points[right_higher, 3] - search_points[right_higher, 0]
            ) / phi

        # Collapse the search points.
        rs = (search_points[..., 0] + search_points[..., 3]) / 2
        errs = (h_sign * (-1)**ns)[None, :] * (f(rs) - p(rs))
        # Select the better optimum from the intervals on either side.
        rs = np.where(errs[0, :] > errs[1, :], rs[0, :], rs[1, :])
        errs = np.max(errs, axis=0)


        # Step 3: Check exit conditions.

        err_lo = np.min(errs)
        err_hi = np.max(errs)
        if err_hi - err_lo < atol:
            # NOTE: Check this exit condition *before* checking the
            # leveled error increased. Otherwise, the leveled error
            # might decrease due to rounding errors when err_hi is near
            # machine precision.
            break

        h_prev, h_curr = h_curr, abs(h)
        if h_curr <= h_prev:
            # When the algorithm is working properly, the leveled error
            # will monotonically increase. See "Approximation theory and
            # methods" (Powell, S8.1, p. 87).
            raise exceptions.OptimizationError(
                "Leveled error increased when it should not. Rounding"
                " error is likely causing numerical issues. Try"
                " increasing atol.",
            )
    else:
        raise exceptions.OptimizationError(
            "Convergence failed in the allocated number of iterations.",
        )

    return rs, f(rs), err_hi


def minimax_polynomial_approximation(f, a, b, n, *, atol=None):
    """Return a function for evaluating the minimax polynomial.

    Parameters
    ----------
    f : function, required
        The function to approximate. The function should map floats to
        floats.
    a : finite float, required
        The lower end point of the interval over which to approximate
        ``f``.
    b : finite float, required
        The upper end point of the interval over which to approximate
        ``f``.
    n : non-negative int, required
        The degree of the polynomial approximation.
    atol : non-negative float or None, optional
        The absolute tolerance to use for stopping the computation. The
        algorithm ends when the current approximation has a maximum
        error within ``atol`` of the best possible approximation. If
        ``None``, then it will be set to a multiple of machine epsilon.

        Only set this value if you encounter numerical or optimization
        issues. In that case, raise the value until the numerical issues
        disappear and as long as the new absolute tolerance represents
        an acceptable level of error above the best approximation.

        See the docstring of the :py:func:`remez` function for details.

    Returns
    -------
    function
        A function that evaluates the minimax polynomial on arrays of
        floats, entrywise.
    non-negative float
        The worst case absolute error of the minimax polynomial
        approximation to ``f`` from ``a`` to ``b``.

    See Also
    --------
    minimax_polynomial_coefficients :
        Return the minimax polynomial's coefficients.
    remez :
        Return the reference for the minimax polynomial.
    """
    # Validate the arguments.
    if not callable(f):
        raise TypeError("f must be callable.")

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

    n = np.array(n)[()]
    if not np.isscalar(n):
        raise ValueError("n must be a scalar.")
    if n % 1 != 0:
        raise ValueError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")

    if atol is not None and atol < 0:
        raise ValueError("atol must be non-negative.")

    if a > b:
        raise ValueError("a must be less than or equal to b.")

    # Set arguments to appropriate defaults.
    if atol is None:
        machine_epsilon = np.spacing(1.)
        # NOTE: The factor, 256, was chosen empirically to avoid
        # numerical issues when f can be fit exactly.
        atol = 256. * machine_epsilon

    # Compute the minimax polynomial approximation.
    rs, ys, err = remez(f, a, b, n, atol=atol)

    ns = np.arange(n + 2)

    # NOTE: See the ``remez`` function for background on the Remez
    # exchange algorithm and why the following computation yields the
    # minimax polynomial approximation to ``f``.
    p0 = lagrange_interpolate(rs[:-1], ys[:-1])
    p1 = lagrange_interpolate(rs[:-1], (-1)**ns[:-1])
    h = (p0(rs[-1]) - ys[-1]) / (p1(rs[-1]) + (-1)**n)

    p = lagrange_interpolate(rs[:-1], ys[:-1] - h * (-1)**ns[:-1])

    return p, err


def minimax_polynomial_coefficients(
        f,
        a,
        b,
        n,
        *,
        transform=(-1., 1.),
        atol=None,
):
    """Return the minimax polynomial's coefficients.

    Typically, computations for the coefficients of the minimax
    polynomial suffer from numerical issues when the degree of the
    polynomial is high (e.g., greater than 20 or so). These numerical
    issues stem from the fact that the Vandermonde matrix is usually
    ill-conditioned for high degrees.

    For evaluating the minimax polynomial, there exist numerically
    stable algorithms that should be used instead. See the function:
    :py:func:`minimax_polynomial_approximation`.

    Parameters
    ----------
    f : function, required
        The function to approximate. The function should map floats to
        floats.
    a : finite float, required
        The lower end point of the interval over which to approximate
        ``f``.
    b : finite float, required
        The upper end point of the interval over which to approximate
        ``f``.
    n : non-negative int, required
        The degree of the polynomial approximation.
    transform : pair of finite floats or None, optional
        For numerical stability, it can be helpful to map the inputs
        to some other range, e.g. -1 to 1, compute the minimax
        polynomial's coefficients in this space, and then transform
        the coefficients back to the original space. The intervals
        from -1 to 1 or 0 to 1 are good choices. If ``None``, then no
        transformation is applied.
    atol : non-negative float or None, optional
        The absolute tolerance to use for stopping the computation. The
        algorithm ends when the current approximation has a maximum
        error within ``atol`` of the best possible approximation. If
        ``None``, then it will be set to a multiple of machine epsilon.

        Only set this value if you encounter numerical or optimization
        issues. In that case, raise the value until the numerical issues
        disappear and as long as the new absolute tolerance represents
        an acceptable level of error above the best approximation.

        See the docstring of the :py:func:`remez` function for details.

    Returns
    -------
    1D array of finite floats
        The coefficients of the minimax polynomial approximation to
        ``f``, starting with the constant term followed by
        coefficients of increasing order.
    non-negative float
        The worst case absolute error of the minimax polynomial
        approximation to ``f`` from ``a`` to ``b``.

    See Also
    --------
    minimax_polynomial_approximation :
        Return a function representing the minimax polynomial.
    remez :
        Return the reference for the minimax polynomial.
    """
    # Validate the arguments.
    if not callable(f):
        raise TypeError("f must be callable.")

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

    n = np.array(n)[()]
    if not np.isscalar(n):
        raise ValueError("n must be a scalar.")
    if n % 1 != 0:
        raise ValueError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")

    if len(transform) != 2:
        raise ValueError(
            "transform must have exactly two elements: a lower bound"
            " and an upper bound.",
        )
    if np.any(~np.isfinite(transform)):
        raise ValueError("transform must contain only finite floats.")
    if transform[0] >= transform[1]:
        raise ValueError(
            "transform's first element (lower bound) must be strictly"
            " less than the second element (upper bound).",
        )

    if atol is not None and atol < 0:
        raise ValueError("atol must be non-negative.")

    if a > b:
        raise ValueError("a must be less than or equal to b.")

    # Set arguments to appropriate defaults.
    if atol is None:
        machine_epsilon = np.spacing(1.)
        # NOTE: The factor, 256, was chosen empirically to avoid
        # numerical issues when f can be fit exactly.
        atol = 256. * machine_epsilon

    # Compute the minimax polynomial coefficients.

    # Apply the input transformation.

    if transform is not None:
        a_orig, b_orig = a, b
        a, b = transform

        m = (b_orig - a_orig) / (b - a)
        f_orig = f
        def f(xs): return f_orig(a_orig + m * (xs - a))

    # Compute the minimax polynomial.

    p, err = minimax_polynomial_approximation(f, a, b, n, atol=atol)

    # Solve for the polynomial's coefficients using least squares.

    # NOTE: The Chebyshev nodes provide more numerical stability than
    # evenly spaced nodes.
    ns = np.arange(n + 1)
    xs = a + (b - a) * 0.5 * (1 - np.cos(np.pi * (ns + 0.5)/(n+1)))
    coeffs = np.linalg.lstsq(xs[:, None]**ns[None, :], p(xs), rcond=None)[0]

    # Unapply the input transformation.

    if transform is not None:
        # Untransform the minimax polynomial and interpolation nodes.
        m_inv = 1. / m
        p_orig = p
        def p(xs): return p_orig(a + m_inv * (xs - a_orig))

        xs = a_orig + m * (xs - a)

        # Untransform the polynomial's coefficients.
        #
        # We can compute the coefficients for the polynomial after
        # transforming x as follows:
        #
        #   a_n (c(x + b))^n + ... + a_0
        #   = a_n c^n (x + b)^n + ... + a_0
        #   = a_n c^n \sum_{i=0}^n \binom{n, i} b^{n - i} x^i
        #     + ...
        #     + a_0
        #   = \sum_{i=0}^n \binom{n, i} b^{n - i} c^n a_n x^i
        #     + ...
        #     + a_0
        #
        # Collecting terms for x^i leads to the following formula:
        #
        #   a_i' = \sum_{j=i}^n \binom{j, i} b^{j - i} c^j a_j
        #
        # We then just need to apply this formula to the inverse of the
        # transform we initially applied to the inputs.
        ##
        coeffs = np.array([
            np.sum(
                special.binom(np.arange(i, n+1), i)
                * (m * a - a_orig)**(np.arange(i, n+1) - i)
                * m_inv**np.arange(i, n+1)
                * coeffs[i:],
            )
            for i in range(len(coeffs))
        ])

    # Check that the minimax polynomial was effectively recovered.

    if err < atol:
        warnings.warn(
            "f can be approximated with error less than atol. The"
            " minimax polynomial coefficients may be unstable. Consider"
            " reducing n, the degree of the approximation.",
            RuntimeWarning,
            stacklevel=2,
        )

    interpolation_err = np.max(np.abs(
        p(xs) - xs[:, None]**ns[None, :] @ coeffs,
    ))
    if interpolation_err > 0.01 * err + atol:
        raise exceptions.NumericalError(
            "Numerical instability detected when solving for the"
            " minimax polynomial coefficients. Try using the"
            " transform parameter, reducing the polynomial degree,"
            " or evaluating the minimax polynomial directly using"
            " minimax_polynomial_approximation.",
        )

    return coeffs, err


def piecewise_polynomial_knots(f, a, b, ns, *, atol=None):
    """Return knots for the minimax piecewise polynomial approximation.

    The knots of a piecewise polynomial (i.e., spline) are the points
    that divide the domain into pieces. This function computes the
    minimax polynomial approximation to ``f`` independently on each
    piece, so in general the piecewise polynomial will be
    discontinuous. The returned points are chosen such that the worst
    case approximation error is minimized.

    Parameters
    ----------
    f : function, required
        The function to approximate. The function should map floats to
        floats.
    a : finite float, required
        The lower end point of the interval over which to approximate
        ``f``.
    b : finite float, required
        The upper end point of the interval over which to approximate
        ``f``.
    ns : 1D array of non-negative ints, required
        The degree of the polynomial approximation on each piece. The
        length of ``ns`` determines the number of pieces.
    atol : non-negative float or None, optional
        The absolute tolerance to use for stopping the computation. The
        algorithm ends when the current approximation has a maximum
        error within ``atol`` of the best possible approximation. If
        ``None``, then it will be set to a multiple of machine epsilon.

        Only set this value if you encounter numerical or optimization
        issues. In that case, raise the value until the numerical issues
        disappear and as long as the new absolute tolerance represents
        an acceptable level of error above the best approximation.

        See the docstring of the :py:func:`remez` function for details.

    Returns
    -------
    1D array of floats from a to b inclusive
        The ``len(ns) + 1`` knots defining the optimal pieces.
    non-negative float
        The worst case absolute error of the piecewise minimax
        polynomial approximation to ``f`` from ``a`` to ``b`` using the
        returned knots.
    """
    # Validate the arguments.
    if not callable(f):
        raise TypeError("f must be callable.")

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

    ns = np.array(ns)
    if len(ns.shape) != 1:
        raise ValueError("ns must be a 1D array.")
    if np.any(ns % 1 != 0):
        raise ValueError("ns must all be integers.")
    if np.any(ns < 0):
        raise ValueError("ns must all be non-negative.")

    if atol is not None and atol < 0:
        raise ValueError("atol must be non-negative.")

    if a > b:
        raise ValueError("a must be less than or equal to b.")

    # Define constants.
    machine_epsilon = np.spacing(1.)

    # Set arguments to appropriate defaults.
    if atol is None:
        # NOTE: The factor, 256, was chosen empirically to avoid
        # numerical issues when f can be fit exactly.
        atol = 256. * machine_epsilon

    # Compute the piecewise polynomial knots.
    knots = a + (b - a) * np.arange(len(ns)+1) / len(ns)
    errs = np.array([
        remez(f, a, b, n, atol=atol)[2]
        for a, b, n in zip(knots[:-1], knots[1:], ns)
    ])
    err_min, err_max = np.min(errs), np.max(errs)

    # Run binary search *on the worst case error* to find the level at
    # which each of the polynomial pieces has the same error.
    #
    # NOTE: err_min and err_max measure the quality of the *current*
    # approximation, while err_lo and err_hi bound the error level
    # we're seeking via binary search.
    err_lo, err_hi = err_min, err_max
    for _ in range(100):
        # Check if the solution has been found.

        if (err_max - err_min) / err_min < 0.01:
            break

        # Set the target error.

        # NOTE: Use the geometric mean to choose a point halfway
        # between the bounds in log space.
        err_target = (err_lo * err_hi)**0.5

        # Find the knots.

        errs = []
        knots = [a]
        for n in ns[:-1]:
            # Using binary search, find where the next knot must be to
            # achieve the target error.
            knot_prev = knots[-1]
            knot_lo, knot_hi = knot_prev, b
            for _ in range(int(np.ceil(np.log(machine_epsilon) / np.log(0.5)))):
                knot_curr = (knot_lo + knot_hi) / 2
                _, _, err = remez(f, knot_prev, knot_curr, n)
                if err >= err_target:
                    knot_hi = knot_curr
                elif err <= err_target:
                    knot_lo = knot_curr
            errs.append(err)
            knots.append(knot_curr)

        knot_prev = knots[-1]
        knot_curr = b
        _, _, err = remez(f, knot_prev, knot_curr, ns[-1])

        errs.append(err)
        knots.append(knot_curr)

        err_min, err_max = np.min(errs), np.max(errs)
        err_lo = max(err_min, err_lo)
        err_hi = min(err_max, err_hi)
    else:
        raise exceptions.OptimizationError(
            "Convergence failed in the allocated number of iterations.",
        )

    return np.array(knots), err_hi
