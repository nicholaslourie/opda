"""Approximation of univariate functions."""

import numpy as np


def lagrange_interpolate(xs, ys):
    r"""Interpolate ``xs`` and ``ys`` with a polynomial.

    Interpolate the points with a polynomial, :math:`p(x)`, such that:

    .. math::

       \forall i, p(x_i) = y_i

    If there are n+1 points, then the polynomial will have degree n.

    Parameters
    ----------
    xs : 1D array of floats, required
        The x values (asbscissas) to interpolate.
    ys : 1D array of floats, required
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

    ys = np.array(ys)
    if len(ys) == 0:
        raise ValueError("ys must be non-empty.")

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
