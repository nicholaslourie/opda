"""Random number generation."""

import numpy as np


def set_seed(seed):
    """Set the global default random state.

    Parameters
    ----------
    seed : None, int, or np.random.Generator, required
        The random seed for setting the global default random
        state. If ``None``, then the random seed is set using entropy
        from the operating system. If an integer, then that integer is
        used as a random seed. If an instance of
        :py:func:`np.random.Generator`, then that generator is used as
        the global default random number generator. See the ``seed``
        parameter of :py:func:`np.random.default_rng` for details.

    Returns
    -------
    None
    """
    global DEFAULT_GENERATOR

    DEFAULT_GENERATOR = (
        seed
        if isinstance(seed, np.random.RandomState) else
        np.random.default_rng(seed)
    )


DEFAULT_GENERATOR = None
"""The global default random number generator."""

# NOTE: Call set_seed to set up DEFAULT_GENERATOR.
set_seed(seed=None)
