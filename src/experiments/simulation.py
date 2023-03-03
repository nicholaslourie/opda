"""Simulations of random search"""

import dataclasses
import typing

from autograd import numpy as npx
import numpy as np
from scipy import optimize


# test functions

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


# simulations

@dataclasses.dataclass
class Simulation:
    """A simulation of random search.

    Parameters
    ----------
    n_trials : int, required
        The number of trials to simulate.
    n_samples : int, required
        The number of samples to take in the random search for each
        trial.
    n_dims : int, required
        The number of dimensions that ``func`` expects in each data
        point.
    func : function, required
        A broadcastable function mapping vectors of length ``n_dims`` to
        scalars.
    bounds : n_dims x 2 array of floats, required
        Bounds on each dimension for the random search's uniform
        sampling.
    y_argmin : n_dims array of floats, required
        An input at which ``func`` takes its minimum in ``bounds``.
    y_argmax : n_dims array of floats, required
        An input at which ``func`` takes its maximum in ``bounds``.
    y_min : float, required
        The minimum value that ``func`` takes in ``bounds``.
    y_max : float, required
        The maximum value that ``func`` takes in ``bounds``.
    ns : 1D array of ints, required
        The range from 1 to ``n_samples``, inclusive.
    xss : n_trials x n_samples x n_dims array of floats, required
        An array where the ij'th element is the random vector for sample
        j in trial i.
    yss : n_trials x n_samples array of floats, required
        An array whose ij'th element is the value of ``func`` at
        ``xss[i, j]``.
    xs : n_samples x n_dims array of floats, required
        The inputs for the first trial simulating random search:
        ``xss[0, :]``.
    ys : n_samples array of floats, required
        The outputs for the first trial simulating random search:
        ``yss[0, :]``.
    yss_cummax : n_trials x n_samples array of floats, required
        The highest function value seen at each point in the random
        search simulations, i.e. the cumulative maximum of ``yss`` along
        the second axis.

    Attributes
    ----------
    See `Parameters`_ for the attributes.
    """
    n_trials: int
    n_samples: int
    n_dims: int

    func: typing.Callable
    bounds: np.ndarray

    y_argmin: np.ndarray
    y_argmax: np.ndarray
    y_min: float
    y_max: float

    ns: np.ndarray
    xss: np.ndarray
    yss: np.ndarray

    xs: np.ndarray
    ys: np.ndarray
    yss_cummax: np.ndarray

    @classmethod
    def run(
        cls,
        n_trials,
        n_samples,
        n_dims,
        func,
        bounds,
    ):
        """Run and return the simulation.

        Parameters
        ----------
        n_trials : int, required
            The number of trials to simulate.
        n_samples : int, required
            The number of samples to take in the random search for each
            trial.
        n_dims : int, required
            The number of dimensions that ``func`` expects in each data
            point.
        func : function, required
            A broadcastable function mapping vectors of length
            ``n_dims`` to scalars.
        bounds : n_dims x 2 array of floats, required
            Bounds on each dimension for the random search's uniform
            sampling.

        Returns
        -------
        Simulation
            An object representing the simulation's results.
        """
        bounds = np.array(bounds)
        if bounds.shape != (n_dims, 2):
            raise ValueError(
                f'bounds should have shape {(n_dims, 2)}, not {bounds.shape}.'
                f' Each dimension must have a lower and upper bound.'
            )

        y_argmin = optimize.differential_evolution(
            func=func, bounds=bounds,
        ).x
        y_argmax = optimize.differential_evolution(
            func=lambda x: -func(x), bounds=bounds,
        ).x
        y_min = func(y_argmin)
        y_max = func(y_argmax)

        ns = np.arange(1, n_samples + 1)
        xss = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.uniform(
            0, 1, size=(n_trials, n_samples, n_dims),
        )
        yss = func(xss)

        xs = xss[0, :]
        ys = yss[0, :]
        yss_cummax = np.maximum.accumulate(yss, axis=1)

        return cls(
            n_trials=n_trials,
            n_samples=n_samples,
            n_dims=n_dims,
            func=func,
            bounds=bounds,
            y_argmin=y_argmin,
            y_argmax=y_argmax,
            y_min=y_min,
            y_max=y_max,
            ns=ns,
            xss=xss,
            yss=yss,
            yss_cummax=yss_cummax,
            xs=xs,
            ys=ys,
        )
