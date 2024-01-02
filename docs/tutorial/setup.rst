=====
Setup
=====
Check that you have a virtual environment with Python
|minimum_python_version| or above installed.

.. admonition:: Python versions and virtual environments
   :class: tip

   Virtual environments are an essential tool because they isolate a
   project's dependencies from the rest of the environment. You can
   build different Python versions with `pyenv
   <https://github.com/pyenv/pyenv>`_, while `pyenv-virtualenv
   <https://github.com/pyenv/pyenv-virtualenv>`_ helps manage virtual
   environments (for example, automatically activating the right
   environment if you place a `.python-version
   <https://github.com/pyenv/pyenv-virtualenv#activate-virtualenv>`_
   file at the repository's root).


Basic Setup
===========
Clone the repo and cd into it:

.. code-block:: console

   $ git clone https://github.com/nalourie/opda.git
   $ cd opda

Then install opda with ``pip``:

.. code-block:: console

   $ pip install .

Use ``pip install --editable`` if setting up the repository for
development. The ``--editable`` option automatically updates the package
when you change the source code. Editable installs require `pip v21.3 or
higher <https://pip.pypa.io/en/stable/news/#v21-3>`_.


Optional Dependencies
=====================
Opda has several optional dependencies. Install any of the following if
you want to:

``opda[experiments]``
  Use the ``experiments`` package (available only in checkouts of the
  repo):

  .. code-block:: console

     $ pip install .[experiments]

``opda[nbs]``
  Run notebooks available in the :source-dir:`nbs/` directory:

  .. code-block:: console

     $ pip install .[nbs]

``opda[tests]``
  Run the tests.

  .. code-block:: console

     $ pip install .[tests]

``opda[lint]``
  Run the linter (when developing or extending opda):

  .. code-block:: console

     $ pip install .[lint]

``opda[docs]``
  Build the documentation.

  .. code-block:: console

     $ pip install .[docs]

You can also install any combination or all of the above:

.. code-block:: console

   $ pip install .[docs,experiments,lint,nbs,tests]

See :doc:`Usage </tutorial/usage>` and :doc:`Development
</tutorial/development>` for more information on how to use these
dependencies.
