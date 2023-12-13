=============================
opda: Optimal Design Analysis
=============================
A framework for the design and analysis of deep learning experiments.

This repository compiles code and resources related to *optimal design
analysis* (OPDA). Optimal design analysis combines an empirical theory
of deep learning with statistical analyses to answer questions such as:

1. Does a change actually improve performance when you account for
   hyperparameter tuning?
2. What aspects of the data or existing hyperparameters does a new
   hyperparameter interact with?
3. What is the best possible score a model can achieve with perfectly
   tuned hyperparameters?

This toolkit provides everything you need to get started with optimal
design analysis. Jump to the section most relevant to you:

- `Setup <#setup>`_
- `Usage <#usage>`_
- `Development <#development>`_
- `Examples <#examples>`_
- `Resources <#resources>`_
- `Citation <#citation>`_
- `Contact <#contact>`_


Setup
=====
Check you have a virtual environment with Python 3.8 or above
installed.

1. Clone the repo:

   .. code-block:: bash

      $ git clone git@github.com:nalourie/opda.git

2. Install the package with ``pip``:

   .. code-block:: bash

      $ pip install .

   Use the ``--editable`` option for development. Editable installs
   require `pip v21.3 or higher
   <https://pip.pypa.io/en/stable/news/#v21-3>`_.

If you also wish to develop or test the package, then:

1. Install the extra development dependencies:

   .. code-block:: bash

      $ pip install .[tests]

2. Run the tests:

   .. code-block:: bash

      $ pytest

If you want to run the notebooks in this repository:

1. Install the extra notebook dependencies:

   .. code-block:: bash

      $ pip install .[nbs]

2. Run the notebook server:

   .. code-block:: bash

      $ cd nbs/ && jupyter notebook


Usage
=====
The code is self-documenting, use the ``help`` function to read the
documentation for a function or class:

.. code-block:: python

   >>> from opda.nonparametric import EmpiricalDistribution
   >>> help(EmpiricalDistribution)
   Help on class EmpiricalDistribution in module opda.nonparametric:
   ...

See ``opda.nonparametric`` for the primary functionality.


Development
===========
For development, we use `pyenv <https://github.com/pyenv/pyenv>`_ to
manage python versions and
`pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_
with a
`.python-version <https://github.com/pyenv/pyenv-virtualenv#activate-virtualenv>`_
file for managing the virtual environment.

Run tests with `pytest <https://docs.pytest.org/>`_:

.. code-block:: bash

   $ pytest

Some tests use randomness. For reproducibility, the random seed prints
when a test fails if the log level is at least INFO (the default).

Tests are organized into levels. Lower levels run faster and are
suitable for quick feedback during development. To run the tests at and
below a specific level, use the ``--level`` option:

.. code-block:: bash

   $ pytest --level 2

Tests up to level 0 are run by default. Tests without a specified level
are always run.

Check the documentation's correctness by executing code examples as
`doctests <https://docs.python.org/3/library/doctest.html>`_. Run
these doctests with pytest:

.. code-block:: bash

   $ pytest --doctest-modules --doctest-glob *.rst -- README.rst src

``--doctest-modules`` runs doctests from the docstrings in any
python modules, while ``--doctest-globs *.rst`` searches
reStructuredText files for doctests. The arguments (``README.rst src``)
ensure pytest looks at the right paths for these tests.


Examples
========
Let's evaluate a model while accounting for hyperparameter tuning
effort. The ``opda.nonparametric.EmpiricalDistribution`` class allows
us to generate tuning curves that capture the cost-benefit
trade-off. First, make an array of floats, ``ys``, representing the
scores obtained from random hyperparameter search. Then, use it to
instantiate ``EmpiricalDistribution`` with confidence bands:

.. code-block:: python

   >>> import numpy as np
   >>> from opda.nonparametric import EmpiricalDistribution
   >>>
   >>> ys = np.random.default_rng(0).uniform(0.5, 0.8, size=64)
   >>> lower_cdf, point_cdf, upper_cdf =\
   ...   EmpiricalDistribution.confidence_bands(
   ...     ys=ys,            # accuracy results from random search
   ...     confidence=0.80,  # confidence level
   ...     a=0.,             # lower bound on accuracy
   ...     b=1.,             # upper bound on accuracy
   ...   )

This code yields lower and upper 80% confidence bands for the CDF, as
well as a point estimate. You can compute tuning curves from these
distributions via the ``.quantile_tuning_curve`` method:

.. code-block:: python

   >>> ns = np.arange(1, 11)
   >>> point_cdf.quantile_tuning_curve(ns)
   array([0.6576063 , 0.70653402, 0.73889728, 0.74979324, 0.75895368,
          0.76684635, 0.76684635, 0.76708231, 0.77382667, 0.77382667])

The *lower* CDF band gives the *upper* tuning curve band, and the
*upper* CDF band gives the *lower* tuning curve band:

.. code-block:: python

   >>> lower_tuning_curve = upper_cdf.quantile_tuning_curve(ns)
   >>> point_tuning_curve = point_cdf.quantile_tuning_curve(ns)
   >>> upper_tuning_curve = lower_cdf.quantile_tuning_curve(ns)
   >>> (
   ...   lower_tuning_curve < point_tuning_curve
   ... ) & (
   ...   point_tuning_curve < upper_tuning_curve
   ... )
   array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
           True])

Using these functions, you could then plot the tuning curve with
confidence bands:

.. code-block:: python

   >>> import matplotlib as mpl; mpl.use('AGG');
   >>> from matplotlib import pyplot as plt
   >>>
   >>> ns = np.linspace(1, 12, num=1_000)
   >>> plt.plot(
   ...   ns,
   ...   point_cdf.quantile_tuning_curve(ns),
   ...   label='tuning curve',
   ... )
   [<matplotlib.lines.Line2D object at ...>]
   >>> plt.fill_between(
   ...   ns,
   ...   upper_cdf.quantile_tuning_curve(ns),
   ...   lower_cdf.quantile_tuning_curve(ns),
   ...   alpha=0.275,
   ...   label=f'80% confidence',
   ... )
   <matplotlib.collections.PolyCollection object at ...>
   >>> plt.xlabel('search iterations')
   Text(0.5, 0, 'search iterations')
   >>> plt.ylabel('accuracy')
   Text(0, 0.5, 'accuracy')
   >>> plt.legend(loc='lower right')
   <matplotlib.legend.Legend object at ...>
   >>> # plt.savefig('figure.png')

Run ``help(EmpiricalDistribution)`` to see its documentation and learn
about other helpful methods.


Resources
=========
For more information on OPDA, checkout our paper: `Show Your Work with
Confidence: Confidence Bands for Tuning Curves
<https://arxiv.org/abs/2311.09480>`_.


Citation
========
If you use the code, data, or other work presented in this repository,
please cite:

.. code-block:: none

    @misc{lourie2023work,
        title={Show Your Work with Confidence: Confidence Bands for Tuning Curves},
        author={Nicholas Lourie and Kyunghyun Cho and He He},
        year={2023},
        eprint={2311.09480},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }


Contact
=======
For more information, see the code
repository, `opda <https://github.com/nalourie/opda>`_. Questions and
comments may be addressed to Nicholas Lourie.
