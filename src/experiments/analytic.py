"""Analytic approximations to random search."""

import numpy as np
from scipy import optimize, special

# backwards compatibility (autograd < ???)
import warnings  # ruff: isort: skip

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"numpy.core.einsumfunc is deprecated",
        category=DeprecationWarning,
    )
    # Importing autograd can raise deprecation warnings depending on
    # the version of autograd and numpy. For example, autograd 1.7.0
    # and numpy == 2.0.2 give a deprecation warning on import.
    from autograd import hessian


def ellipse_volume(cs):
    """Return the volume of an ellipse with ``cs`` as the axes' lengths.

    Parameters
    ----------
    cs : 1D array of floats, required
        Floats providing the axes' lengths for the ellipse.

    Returns
    -------
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


def get_approximation_parameters(
        func,
        bounds,
        *,
        generator = None,
):
    """Return parameters approximating random search on ``func``.

    Return :py:class:`opda.parametric.QuadraticDistribution` parameters
    that give the asymptotic approximation of the tail of the
    distribution obtained by applying random search to ``func``.

    Parameters
    ----------
    func : function, required
        The function used in the random search.
    bounds : d x 2 array of finite floats, required
        An array of d pairs of floats, where d is the dimension of
        ``func``'s domain. Each pair, ``(lo, hi)`` defines a lower and
        upper bound for the random search on that respective coordinate.
    generator : np.random.Generator or None, optional
        The random number generator to use. If ``None``, then a new
        random number generator is created, seeded with entropy from the
        operating system.

    Returns
    -------
    finite float
       The ``a`` parameter for
       :py:class:`opda.parametric.QuadraticDistribution` that
       asymptotically approximates the tail of the score distribution
       from random search.
    finite float
       The ``b`` parameter for
       :py:class:`opda.parametric.QuadraticDistribution` that
       asymptotically approximates the tail of the score distribution
       from random search.
    positive int
       The ``c`` parameter for
       :py:class:`opda.parametric.QuadraticDistribution` that
       asymptotically approximates the tail of the score distribution
       from random search.
    """
    # Validate the arguments.
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
    if np.any(~np.isfinite(bounds)):
        raise ValueError("bounds must contain only finite floats.")

    generator = (
        generator
        if generator is not None else
        np.random.default_rng()
    )

    # Determine the approximation parameters.
    n_dims, _ = bounds.shape

    y_argmax = optimize.differential_evolution(
        lambda x: -func(x), bounds=bounds, seed=generator,
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
