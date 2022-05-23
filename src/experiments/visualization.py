"""Visualizations"""

from matplotlib import pyplot as plt
import numpy as np


def plot_random_search(
    func,
    *,
    bounds = (-1., 1.),
    n_samples = 10,
    n_grid = 1_000,
    ax = None,
):
    """Return a plot visualizing the random search process.

    Parameters
    ----------
    func : function, required
        A function mapping 1 dimensional vectors to scalars.
    bounds : pair of floats, optional (default=(-1., 1.))
        Bounds for the random search's uniform sampling.
    n_samples : int, optional (default=10)
        The number of samples to take in the random search.
    n_grid : int, optional (default=1_000)
        The number of grid points to use when plotting the function.
    ax : plt.Axes or None, optional (default=None)
        An axes on which to make the plot, or ``None``. If ``None``,
        then a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure, plt.Axes
        The figure and axes on which the plot was made. If ``ax`` was
        not ``None``, then the returned figure will be ``None``.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    else:
        fig = None

    grid = np.linspace(*bounds, num=n_grid).reshape(n_grid, 1)

    y_argmax = grid[np.argmax(func(grid))]

    xs = np.random.uniform(*bounds, size=(n_samples, 1))
    ys = func(xs)

    ax.plot(
        grid,
        func(grid),
        linestyle='-',
        c='grey',
        label='$y = f(x)$',
    )
    ax.axvline(
        y_argmax,
        linestyle='--',
        c='grey',
        label='$\\operatorname{arg\\,max} f(x)$',
    )
    ax.scatter(xs, ys, marker='x', c='k', s=50)
    for x in xs:
        ax.axvline(x, linestyle=':', c='grey')

    ax.legend()
    ax.set_title('Random Search')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    return fig, ax


def plot_cdf(
        xs,
        name,
        *,
        ax = None,
):
    """Return a plot visualizing the empirical CDF of ``xs``.

    Parameters
    ----------
    xs : 1D array of floats, required
        The sample whose empirical CDF should be visualized.
    name : str, required
        The name of the random variable represented by ``xs``.
    ax : plt.Axes or None, optional (default=None)
        An axes on which to make the plot, or ``None``. If ``None``,
        then a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure, plt.Axes
        The figure and axes on which the plot was made. If ``ax`` was
        not ``None``, then the returned figure will be ``None``.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    else:
        fig = None

    n, = xs.shape

    ax.plot(np.sort(xs), (np.arange(n) + 1) / n)

    ax.set_xlabel(f'${name.lower()}$')
    ax.set_ylabel(f'$\mathbb{{P}}({name.upper()} \leq {name.lower()})$')
    ax.set_title(f'CDF (${name}$)')

    return fig, ax


def plot_pdf(
        xs,
        name,
        *,
        ax = None,
):
    """Return a plot visualizing a histogram of ``xs``.

    Parameters
    ----------
    xs : 1D array of floats, required
        The sample for which to create a histogram.
    name : str, required
        The name of the random variable represented by ``xs``.
    ax : plt.Axes or None, optional (default=None)
        An axes on which to make the plot, or ``None``. If ``None``,
        then a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure, plt.Axes
        The figure and axes on which the plot was made. If ``ax`` was
        not ``None``, then the returned figure will be ``None``.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    else:
        fig = None

    n, = xs.shape

    ax.hist(xs, density=True)

    ax.set_xlabel(f'${name.lower()}$')
    ax.set_ylabel(f'$d\mathbb{{P}}({name.upper()} = {name.lower()})$')
    ax.set_title(f'PDF (${name}$)')

    return fig, ax
