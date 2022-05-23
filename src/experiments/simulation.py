"""Simulations of random search"""

from autograd import numpy as npx


# Test Functions

def make_damped_linear_sin(
        weights,
        bias = 0.,
        scale = 1.,
):
    """Return a damped sinusoidal function with a linear trend.

    Parameters
    ----------
    weights : 1D array of floats, required
        Weights controlling the sensitivity of the output to each
        dimension of the input.
    bias : float, optional (default=0.)
        The bias term.
    scale : float, optional (default=1.)
        A scalar multiplier for the function's outputs.

    Returns
    -------
    function
        A function mapping vectors of dimension equal to the length of
       ``weights`` to scalars. The function broadcasts over the last
        dimension.
    """
    weights = npx.array(weights)
    bias = npx.array(bias)
    scale = npx.array(scale)

    if len(weights.shape) != 1:
        raise ValueError(
            f'weights must be 1D, not {len(weights.shape)}D.'
        )

    n_dim, = weights.shape

    def func(xs):
        xs = npx.array(xs)
        if xs.shape[-1] != n_dim:
            raise ValueError(
                f'The last dimension of xs should be length {n_dim}, not'
                f' {xs.shape[-1]}.'
            )

        zs = weights * xs + bias
        ys = scale * (
            npx.exp(- npx.sum(zs**2, axis=-1))
            * npx.sum(zs + npx.sin(2 * npx.pi * zs), axis=-1)
        )

        return ys

    return func
