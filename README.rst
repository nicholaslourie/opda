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
- `Examples <#examples>`_
- `Development <#development>`_
- `Contact <#contact>`_


Setup
=====
Check you have a virtual environment with Python 3.9 or above
installed.

1. Clone the repo:

   .. code-block:: bash

      $ git clone git@github.com:nalourie/opda.git

2. Install the package with ``pip``:

   .. code-block:: bash

      $ pip install .

   Use the ``--editable`` option for development.

If you also wish to develop or test the package, then:

1. Install the development requirements:

   .. code-block:: bash

      $ pip install --requirement dev-requirements.txt

2. Run the tests:

   .. code-block:: bash

      $ pytest

If you want to run the notebooks in this repository:

1. Install jupyter notebooks:

   .. code-block:: bash

      $ pip install notebook

2. Run the notebook server:

   .. code-block:: bash

      $ cd nbs/ && jupyter notebook


Usage
=====
The code is self-documenting, use the ``help`` function to read the
documentation for a function or class:

.. code-block:: python

   >>> from opda.nonparametric import EmpiricalDistribution
   >>> help(QuadraticDistribution)
   Help on class EmpiricalDistribution in module opda.nonparametric:
   ...

See ``opda.parametric`` for parametric and ``opda.nonparametric`` for
nonparametric analyses.


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
   >>> ys = np.random.uniform(size=64)
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
   array([0.47992688, 0.67358247, 0.75169446, 0.78485399, 0.81752114,
          0.85299978, 0.85299978, 0.86373213, 0.89545778, 0.89545778])

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

   >>> from matplotlib import pyplot as plt
   >>>
   >>> ns = np.linspace(1, 10, num=1_000)
   >>> plt.plot(
   ...   ns,
   ...   point_cdf.quantile_tuning_curve(ns),
   ...   label='tuning curve',
   ... )
   >>> plt.fill_between(
   ...   ns,
   ...   upper_cdf.quantile_tuning_curve(ns),
   ...   lower_cdf.quantile_tuning_curve(ns),
   ...   alpha=0.275,
   ...   label=f'80% confidence',
   ... )
   >>> plt.xlabel('search iterations')
   >>> plt.ylabel('accuracy')
   >>> plt.legend()
   >>> plt.show()

Run ``help(EmpiricalDistribution)`` to see its documentation and learn
about other helpful methods.


Development
===========
Run tests with ``pytest``:

.. code-block:: bash

   $ pytest

Tests are organized into levels. Lower levels run faster and are
suitable for quick feedback during development. To run the tests at and
below a specific level, use the ``--level`` option:

.. code-block:: bash

   $ pytest --level 2

Tests up to level 0 are run by default. Tests without a specified level
are always run.


Contact
=======
For more information, see the code
repository, `opda <https://github.com/nalourie/opda>`_. Questions and
comments may be addressed to Nicholas Lourie.
