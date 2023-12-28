"""Random number generation.

Functions and methods in ``opda`` generate random values using the
:py:const:`DEFAULT_GENERATOR` by default.

To set the global random state, use the :py:func:`set_seed` function:

.. code-block:: python

   >>> import opda.random
   >>> opda.random.set_seed(0)
   >>> opda.random.DEFAULT_GENERATOR.uniform()
   0.6369616873214543

Functions and methods using randomness also provide a ``generator``
keyword argument. If ``generator`` is ``None`` then the global random
state, :py:const:`DEFAULT_GENERATOR`, is used. Otherwise, the
``generator`` argument should be an instance of
:py:class:`numpy.random.Generator` and the function or method will use
that for generating random values.

If you require *local* control of random behavior (e.g., making a
single function deterministic), pass the generator explicitly to the
functions or methods. If you require *global* control of random
behavior (e.g., making a script reproducible), use
:py:func:`set_seed`.

.. warning::

   :py:mod:`numpy` does *not* guarantee that fixing the random
   seed will make random computations reproducible across versions or
   even operating systems. While fixing the random seed aids in
   reproducibility, you must also provide enough information to
   reproduce the rest of your computing environment, such as pinning
   dependencies, or even potentially specifying the Python version,
   operating system, and hardware. See `numpy's discussion on
   reproducibility
   <https://numpy.org/doc/stable/reference/random/compatibility.html>`_.

For scripts that are run once and then distributed to other parties,
you may want to set the random seed at the start of the script for
reproducibility:

.. code-block:: python

   #! /usr/bin/env python

   import opda.random

   opda.random.set_seed(0)

   # ... script logic ...

For scripts that are run many times (and not necessarily distributed
to other parties), you may want to randomly set the seed and log it,
so the script's run can be reproduced in the event of a failure:

.. code-block:: python

   #! /usr/bin/env python

   import logging

   import numpy as np

   import opda.random

   logging.basicConfig()
   logger = logging.getLogger(__name__)

   seed = np.random.SeedSequence().entropy
   opda.random.set_seed(seed)
   logger.info(f"Random seed for opda set to {seed}.")

   # ... script logic ...

Avoid setting the random seed multiple times without careful
consideration. Setting the random seed twice can lead to poor
statistical properties in the pseudo-random number generation (e.g.,
if the seed is reset to the same value). Consequently, the random seed
should typically only be set by application code, and not by a library.
"""

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
        :py:class:`numpy.random.Generator`, then that generator is used
        as the global default random number generator. See the ``seed``
        parameter of :py:func:`numpy.random.default_rng` for details.

    Returns
    -------
    None
    """
    global DEFAULT_GENERATOR  # noqa: PLW0603

    DEFAULT_GENERATOR = (
        seed
        if isinstance(seed, np.random.RandomState) else
        np.random.default_rng(seed)
    )


DEFAULT_GENERATOR = None
"""The global default random number generator."""

# NOTE: Call set_seed to set up DEFAULT_GENERATOR.
set_seed(seed=None)
