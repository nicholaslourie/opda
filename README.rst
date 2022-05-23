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
You can use the following commands to accomplish common tasks:

Run tests:

.. code:: bash

          $ pytest
