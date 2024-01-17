"""Analytic approximations to random search."""

from autograd import hessian
import numpy as np
from scipy import optimize, special


def ellipse_volume(cs):
    """Return the volume of an ellipse with ``cs`` as the axes' lengths.

    Parameters
    ----------
    cs : array of floats, required
        A 1D array of floats providing the axes' lengths for the
        ellipse.

    Return
    ------
    float
        The volume of the ellipse.
    """
    cs = np.array(cs)
    if not len(cs.shape) == 1:
        raise ValueError(f"cs must be a 1D array, not have shape {cs.shape}.")

    n_dims, = cs.shape

    return (
        np.pi ** (n_dims / 2)
        / special.gamma(n_dims/2 + 1)
        * np.prod(cs)
    )


def get_approximation_parameters(func, bounds):
    """Return parameters approximating random search on ``func``.

    Return parameters for the asymptotic approximation of the tail of
    values obtained by applying random search to ``func``.

    Parameters
    ----------
    func : function, required
        The function used in the random search.
    bounds : d x 2 array of floats, required
        An array of d pairs of floats, where d is the dimension of
        ``func``'s domain. Each pair, ``(lo, hi)`` defines a lower and
        upper bound for the random search on that respective coordinate.

    Returns
    -------
    float, float, float
        The parameters, ``a``, ``b``, and ``c`` for the quadratic
        distribution asymptotically approximating the random search
        distribution.
    """
    bounds = np.array(bounds)
    if not len(bounds.shape) == 2:
        raise ValueError(
            f"bounds must be a d x 2 array of pairs, not have shape"
            f" {bounds.shape}, where d is the dimension of func's"
            f" domain.",
        )
    if not bounds.shape[1] == 2:
        raise ValueError(
            f"bounds must be an array of pairs, not have shape"
            f" {bounds.shape}. Each pair, (lo, hi), must bound the"
            f" corresponding coordinate of the input to func.",
        )

    n_dims, _ = bounds.shape

    y_argmax = optimize.differential_evolution(
        lambda x: -func(x), bounds=bounds,
    ).x
    y_max = func(y_argmax)

    vol = ellipse_volume(
        cs=(-2 / np.linalg.eigvals(hessian(func)(y_argmax)))**0.5,
    )

    omega = vol / np.prod(bounds[:, 1] - bounds[:, 0])

    a = y_max - (1 / omega)**(2 / n_dims)
    b = y_max
    c = n_dims

    return a, b, c
