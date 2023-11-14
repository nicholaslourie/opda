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


Setup
=====
Check you have a virtual environment with Python 3.9 or above
installed.

1. Clone the repo:

   .. code:: bash

      $ git clone git@github.com:nalourie/opda.git

2. Install the package with ``pip``:

   .. code:: bash

      $ pip install .

   Use the ``--editable`` option for development.

If you also wish to develop or test the package, then:

1. Install the development requirements:

   .. code:: bash

      $ pip install --requirement dev-requirements.txt

2. Run the tests:

   .. code:: bash

      $ pytest


Usage
=====
The code is self-documenting, use the ``help`` function to read the
documentation for a function or class:

.. code:: bash

   >>> from opda.nonparametric import EmpiricalDistribution
   >>> help(QuadraticDistribution)
   Help on class EmpiricalDistribution in module opda.nonparametric:
   ...

See ``opda.parametric`` for parametric and ``opda.nonparametric`` for
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
