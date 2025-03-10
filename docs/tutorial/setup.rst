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
Install opda via ``pip``:

.. code-block:: console

   $ pip install opda

Or, if you're interested in *development*, clone the repo and do an
``--editable`` install:

.. code-block:: console

   $ git clone https://github.com/nicholaslourie/opda.git
   $ cd opda
   $ pip install --editable .

The ``--editable`` option automatically updates the package when you
change the source code. Editable installs require `pip v21.3 or higher
<https://pip.pypa.io/en/stable/news/#v21-3>`_.


Optional Dependencies
=====================
Opda has several optional dependencies. Most optional dependencies
need to be installed with `pip v21.2 or higher
<https://pip.pypa.io/en/stable/news/#v21-2>`_.

Install any of the following if you want to:

``opda[experiments]``
  Use the ``experiments`` package (**available only in checkouts of the
  repo**):

  .. code-block:: console

     $ pip install .[experiments]

``opda[nbs]``
  Run notebooks available in the :source-dir:`nbs/` directory:

  .. code-block:: console

     $ pip install opda[nbs]

``opda[test]``
  Run the tests.

  .. code-block:: console

     $ pip install opda[test]

``opda[lint]``
  Run the linter (when developing or extending opda):

  .. code-block:: console

     $ pip install opda[lint]

``opda[docs]``
  Build the documentation.

  .. code-block:: console

     $ pip install opda[docs]

``opda[package]``
  Build the distribution package.

  .. code-block:: console

     $ pip install opda[package]

``opda[ci]``
  Run continuous integration commands using `nox
  <https://nox.thea.codes/en/stable/>`_:

  .. code-block:: console

     $ pip install opda[ci]

You can also install any combination or all of the above:

.. code-block:: console

   $ pip install opda[ci,docs,experiments,lint,nbs,package,test]

For local development setups, use a ``.`` in place of ``opda`` in all
of the above.

See :doc:`Usage </tutorial/usage>` and :doc:`Development
</contributing/development>` for more information on how to use these
dependencies.


Python Versions
===============
Opda uses tools like `nox <https://nox.thea.codes/en/stable/>`_ to test
itself against the Python versions it supports. To :doc:`develop
</contributing/development>` opda, you must install these Python
versions. They can be found in the package's metadata:

.. code-block:: python

   >>> from importlib.metadata import metadata
   >>> for classifier in metadata("opda").get_all("Classifier"):
   ...   *prefix, version = classifier.split(" :: ")
   ...   if prefix != ["Programming Language", "Python"] or "." not in version:
   ...     continue
   ...   print(version)
   3.9
   3.10
   3.11
   3.12
   3.13

To install them, we recommend `pyenv <https://github.com/pyenv/pyenv>`_:

.. code-block:: console

   $ pyenv install 3.9 3.10 3.11 3.12 3.13

After the required versions are installed, make sure they're available
on your PATH. You can do this either `globally
<https://github.com/pyenv/pyenv/blob/master/COMMANDS.md#pyenv-global>`_:

.. code-block:: console

   $ pyenv global system 3.9 3.10 3.11 3.12 3.13

Or `locally
<https://github.com/pyenv/pyenv/blob/master/COMMANDS.md#pyenv-local>`_
(just within the opda repository):

.. code-block:: console

   $ pyenv local opda 3.9 3.10 3.11 3.12 3.13

The above example assumes you have a virtual environment named ``opda``
that you wish to `activate using pyenv-virtualenv
<https://github.com/pyenv/pyenv-virtualenv#activate-virtualenv>`_
whenever inside the repository. If you have no such virtual environment,
then omit ``opda`` from the command.
