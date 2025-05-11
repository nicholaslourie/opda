"""Visualizations."""

import colorsys

from matplotlib import (
    colors,
    pyplot as plt,
)
import numpy as np


def plot_random_search(
        func,
        bounds = (-1., 1.),
        n_samples = 10,
        n_grid = 1_000,
        ax = None,
        *,
        generator = None,
):
    """Return a plot visualizing the random search process.

    Parameters
    ----------
    func : function, required
        A function mapping 1 dimensional vectors to scalars.
    bounds : pair of floats, optional
        Bounds for the random search's uniform sampling.
    n_samples : int, optional
        The number of samples to take in the random search.
    n_grid : int, optional
        The number of grid points to use when plotting the function.
    ax : plt.Axes or None, optional
        An axes on which to make the plot, or ``None``. If ``None``,
        then a figure and axes for the plot will be automatically
        generated.
    generator : np.random.Generator or None, optional
        The random number generator to use. If ``None``, then a new
        random number generator is created, seeded with entropy from
        the operating system.

    Returns
    -------
    plt.Figure or None
        The figure on which the plot was made. If ``ax`` was not
        ``None``, then the returned figure will be ``None``.
    plt.Axes
        The axis on which the plot was made. If ``ax`` was not ``None``,
        then the returned axis will be ``ax``.
    """
    # Validate arguments.
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
    else:
        fig = None

    generator = (
        generator
        if generator is not None else
        np.random.default_rng()
    )

    # Construct the plot.
    grid = np.linspace(*bounds, num=n_grid).reshape(n_grid, 1)

    y_argmax = grid[np.argmax(func(grid))]

    xs = generator.uniform(*bounds, size=(n_samples, 1))
    ys = func(xs)

    ax.plot(
        grid,
        func(grid),
        linestyle="-",
        c="grey",
        label=r"$y = g(x)$",
    )
    ax.axvline(
        y_argmax,
        linestyle="--",
        c="grey",
        label=r"$\operatorname{arg\,max} g(x)$",
    )
    ax.scatter(xs, ys, marker="x", c="k", s=50)
    for x in xs:
        ax.axvline(x, linestyle=":", c="grey")

    ax.legend()
    ax.set_title("Random Search")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    return fig, ax


def plot_cdf(
        xs,
        name,
        ax = None,
):
    """Return a plot visualizing the empirical CDF of ``xs``.

    Parameters
    ----------
    xs : 1D array of floats, required
        The sample whose empirical CDF should be visualized.
    name : str, required
        The name of the random variable represented by ``xs``.
    ax : plt.Axes or None, optional
        An axes on which to make the plot, or ``None``. If ``None``,
        then a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure or None
        The figure on which the plot was made. If ``ax`` was not
        ``None``, then the returned figure will be ``None``.
    plt.Axes
        The axis on which the plot was made. If ``ax`` was not ``None``,
        then the returned axis will be ``ax``.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    else:
        fig = None

    n, = xs.shape

    ax.plot(np.sort(xs), (np.arange(n) + 1) / n)

    ax.set_xlabel(rf"${name.lower()}$")
    ax.set_ylabel(rf"$\mathbb{{P}}({name.upper()} \leq {name.lower()})$")
    ax.set_title(rf"CDF (${name}$)")

    return fig, ax


def plot_pdf(
        xs,
        name,
        ax = None,
):
    """Return a plot visualizing a histogram of ``xs``.

    Parameters
    ----------
    xs : 1D array of floats, required
        The sample for which to create a histogram.
    name : str, required
        The name of the random variable represented by ``xs``.
    ax : plt.Axes or None, optional
        An axes on which to make the plot, or ``None``. If ``None``,
        then a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure or None
        The figure on which the plot was made. If ``ax`` was not
        ``None``, then the returned figure will be ``None``.
    plt.Axes
        The axis on which the plot was made. If ``ax`` was not ``None``,
        then the returned axis will be ``ax``.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    else:
        fig = None

    n, = xs.shape

    ax.hist(xs, bins="auto", density=True)

    ax.set_xlabel(rf"${name.lower()}$")
    ax.set_ylabel(rf"$d\mathbb{{P}}({name.upper()} = {name.lower()})$")
    ax.set_title(rf"PDF (${name}$)")

    return fig, ax


def plot_distribution(
        xs,
        name,
        axes = None,
):
    """Return a plot visualizing the distribution of ``xs``.

    Parameters
    ----------
    xs : 1D array of floats, required
        The sample whose distribution should be visualized.
    name : str, required
        The name of the random variable represented by ``xs``.
    axes : plt.Axes or None, optional
        Axes on which to make the plot, or ``None``. If ``None``, then
        a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure or None
        The figure on which the plot was made. If ``axes`` was not
        ``None``, then the returned figure will be ``None``.
    plt.Axes
        The axes on which the plot was made. If ``axes`` was not
        ``None``, then the returned axes will be ``axes``.
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
    else:
        fig = None

    plot_cdf(xs, name=name, ax=axes[0])
    plot_pdf(xs, name=name, ax=axes[1])

    return fig, axes


def plot_distribution_approximation(
        simulation,
        approximating_distribution,
        n,
        axes = None,
):
    """Return a plot of the approximation to the max's distribution.

    Return a plot that visualizes the Hessian-based asymptotic
    approximation to the distribution of the maximum for the
    simulation ``simulation``.

    Parameters
    ----------
    simulation : Simulation, required
        The simulation for which to visualize the approximation.
    approximating_distribution : QuadraticDistribution, required
        The distribution approximating the distribution of the maximum
        in the simulation.
    n : int, required
        The number of samples from which to take the maximum.
    axes : plt.Axes or None, optional
        Axes on which to make the plot, or ``None``. If ``None``, then
        a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure or None
        The figure on which the plot was made. If ``axes`` was not
        ``None``, then the returned figure will be ``None``.
    plt.Axes
        The axes on which the plot was made. If ``axes`` was not
        ``None``, then the returned axes will be ``axes``.
    """
    fig, axes = plot_distribution(
        simulation.yss_cummax[:, n-1],
        name=f"T_{{{n}}}",
        axes=axes,
    )

    xlims = [ax.get_xlim() for ax in axes]
    ylims = [ax.get_ylim() for ax in axes]

    grid = np.linspace(simulation.y_min, simulation.y_max, num=10_000)
    axes[0].plot(
        grid,
        approximating_distribution.cdf(grid) ** n,
        label="Approximation",
    )
    axes[1].plot(
        grid,
        # d/dy (F(y)^n) = n F(y)^(n - 1) dF(y)
        n * approximating_distribution.cdf(grid)**(n - 1)
          * approximating_distribution.pdf(grid),
        label="Approximation",
    )

    for ax, xlim, ylim in zip(axes, xlims, ylims):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend()

    return fig, axes


def plot_tuning_curve_approximation(
        simulation,
        approximating_distribution,
        axes = None,
):
    """Return a plot of the approximation to the tuning curve.

    Return a plot that visualizes the Hessian-based asymptotic
    approximation to the tuning curve for the simulation ``simulation``.

    Parameters
    ----------
    simulation : Simulation, required
        The simulation for which to visualize the approximation.
    approximating_distribution : QuadraticDistribution, required
        The distribution approximating the distribution of the maximum
        in the simulation.
    axes : plt.Axes or None, optional
        Axes on which to make the plot, or ``None``. If ``None``,
        then a figure and axes for the plot will be automatically
        generated.

    Returns
    -------
    plt.Figure or None
        The figure on which the plot was made. If ``axes`` was not
        ``None``, then the returned figure will be ``None``.
    plt.Axes
        The axes on which the plot was made. If ``axes`` was not
        ``None``, then the returned axes will be ``axes``.
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
    else:
        fig = None

    axes[0].plot(
        simulation.ns,
        np.median(simulation.yss_cummax, axis=0),
        label="Truth",
    )
    axes[1].plot(
        simulation.ns,
        np.mean(simulation.yss_cummax, axis=0),
        label="Truth",
    )

    xlims = [ax.get_xlim() for ax in axes]
    ylims = [ax.get_ylim() for ax in axes]

    axes[0].plot(
        simulation.ns,
        approximating_distribution.quantile_tuning_curve(simulation.ns),
        label="Approximation",
    )
    axes[1].plot(
        simulation.ns,
        approximating_distribution.average_tuning_curve(simulation.ns),
        label="Approximation",
    )

    for ax, xlim, ylim in zip(axes, xlims, ylims):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.legend()

    axes[0].set_xlabel(r"$k$")
    axes[0].set_ylabel(r"$\tau_m(k)$")
    axes[0].set_title("Median Tuning Curve")

    axes[1].set_xlabel(r"$k$")
    axes[1].set_ylabel(r"$\tau_e(k)$")
    axes[1].set_title("Expected Tuning Curve")

    return fig, axes


def color_with_lightness(c, lightness):
    """Return a new color by changing the lightness.

    Parameters
    ----------
    c : str or tuple of float, required
        A value that can be interpreted as a matplotlib color, like a
        color name, hex string, or tuple of floats. See the matplotlib
        documentation on `specifying colors
        <https://matplotlib.org/stable/tutorials/colors/colors.html>`_.
    lightness : float from 0 to 1 inclusive, required
        A float between 0 and 1 specifying the lightness for the new
        color in the Hue-Saturation-Lightness color space.

    Returns
    -------
    tuple of floats from 0 to 1 inclusive
        A tuple of the RGB channel values on a scale from 0 to 1.
    """
    # Validate the arguments.
    if lightness < 0. or lightness > 1.:
        raise ValueError(
            "lightness must be between 0 and 1, inclusive.",
        )

    # Compute the color with the new lightness.
    hue, _, saturation = colorsys.rgb_to_hls(
        *colors.to_rgb(c),
    )

    return colorsys.hls_to_rgb(hue, lightness, saturation)
