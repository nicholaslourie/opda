=========================================
ERSA: Extrapolated Random Search Analysis
=========================================
A method for analyzing and extrapolating random search.

This repository compiles code and resources related to ERSA
(Extrapolated Random Search Analaysis). ERSA analyzes and extrapolates
random search, answering questions such as:

1. How difficult is this function to optimize?
2. What score can you expect after a given number of iterations of
   random search?
3. What's the best possible score you could achieve?

A major application of ERSA is understanding how performance improves as
you tune a machine learning model's hyper-parameters.


Setup
=====
Check you have a virtual environment with Python 3.9 installed.

1. Clone the repo:

   .. code:: bash

      $ git clone git@github.com:nalourie/ersa.git

2. Install the package with ``pip``:

   .. code:: bash

      $ pip install .

   Use the ``--editable`` option for development.


Usage
=====
The code is self-documenting, use the ``help`` function to read the
documentation for a function or class:

.. code:: bash

   >>> from ersa.parametric import QuadraticDistribution
   >>> help(QuadraticDistribution)
   Help on class QuadraticDistribution in module ersa.parametric:
   ...

See ``ersa.parametric`` for parametric and ``ersa.nonparametric`` for
nonparametric analyses.


Development
===========
Run tests with ``pytest``:

.. code:: bash

   $ pytest

Tests are organized into levels. Lower levels run faster and are
suitable for quick feedback during development. To run the tests at and
below a specific level, use the ``--level`` option:

.. code:: bash

   $ pytest --level 2

Tests up to level 0 are run by default. Tests without a specified level
are always run.
