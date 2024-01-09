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
Opda has several optional dependencies. Most optional dependencies
need to be installed with `pip v21.2 or higher
<https://pip.pypa.io/en/stable/news/#v21-2>`_.

Install any of the following if you want to:

``opda[experiments]``
  Use the ``experiments`` package (available only in checkouts of the
  repo):

  .. code-block:: console

     $ pip install .[experiments]

``opda[nbs]``
  Run notebooks available in the :source-dir:`nbs/` directory:

  .. code-block:: console

     $ pip install .[nbs]

``opda[test]``
  Run the tests.

  .. code-block:: console

     $ pip install .[test]

``opda[lint]``
  Run the linter (when developing or extending opda):

  .. code-block:: console

     $ pip install .[lint]

``opda[docs]``
  Build the documentation.

  .. code-block:: console

     $ pip install .[docs]

``opda[ci]``
  Run continuous integration commands using `nox
  <https://nox.thea.codes/en/stable/>`_:

  .. code-block:: console

     $ pip install .[ci]

You can also install any combination or all of the above:

.. code-block:: console

   $ pip install .[ci,docs,experiments,lint,nbs,test]

See :doc:`Usage </tutorial/usage>` and :doc:`Development
</tutorial/development>` for more information on how to use these
dependencies.


Python Versions
===============
Opda uses tools like `nox <https://nox.thea.codes/en/stable/>`_ to test
itself against the Python versions it supports. To :doc:`develop
</tutorial/development>` opda, you must install these Python versions.
They can be found in the package's metadata:

.. code-block:: python

   >>> from importlib.metadata import metadata
   >>> for classifier in metadata("opda").get_all("Classifier"):
   ...   *prefix, version = classifier.split(" :: ")
   ...   if prefix != ["Programming Language", "Python"] or "." not in version:
   ...     continue
   ...   print(version)
   3.8
   3.9
   3.10
   3.11
   3.12

To install them, we recommend `pyenv <https://github.com/pyenv/pyenv>`_:

.. code-block:: console

   $ pyenv install 3.8 3.9 3.10 3.11 3.12

After the required versions are installed, make sure they're available
on your PATH. You can do this either `globally
<https://github.com/pyenv/pyenv/blob/master/COMMANDS.md#pyenv-global>`_:

.. code-block:: console

   $ pyenv global system 3.8 3.9 3.10 3.11 3.12

Or `locally
<https://github.com/pyenv/pyenv/blob/master/COMMANDS.md#pyenv-local>`_
(just within the opda repository):

.. code-block:: console

   $ pyenv local opda 3.8 3.9 3.10 3.11 3.12

The above example assumes you have a virtual environment named ``opda``
that you wish to `activate using pyenv-virtualenv
<https://github.com/pyenv/pyenv-virtualenv#activate-virtualenv>`_
whenever inside the repository. If you have no such virtual environment,
then omit ``opda`` from the command.
