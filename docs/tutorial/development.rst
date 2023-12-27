===========
Development
===========
We use the following tools and guidelines when developing opda.

.. admonition:: Before contributing to opda
   :class: caution

   Before contributing to opda, it's a good idea to discuss in an issue
   what you'd like to do. A pull request might not get accepted. You can
   save effort by discussing things beforehand and reducing the chance
   of this outcome.

Make sure you've installed all :ref:`optional dependencies
<tutorial/setup:Optional Dependencies>` necessary for development.


Tests
=====
Run tests with `pytest <https://docs.pytest.org/>`_:

.. code-block:: console

   $ pytest

Some tests use randomness. For reproducibility, the random seed prints
when a test fails if the log level is at least :py:data:`~logging.INFO`
(the default).

Tests are organized into levels. Lower levels run faster and are
suitable for quick feedback during development. To run the tests at and
below a specific level, use the ``--level`` option:

.. code-block:: console

   $ pytest --level 2

Tests up to level 0 are run by default. Tests without a specified level
are always run. To run all levels, use the ``--all-levels`` option:

.. code-block:: console

   $ pytest --all-levels


Lint
====
Lint the repository using `ruff <https://docs.astral.sh/ruff/>`_:

.. code-block:: console

   $ ruff check .

Use ``ruff check --watch .`` to continually lint the repository with
updates on file changes.

This project does *not* use a formatter. Basic stylistic conventions are
enforced by the linter; otherwise, style should be used to maximize the
readability and communicate the intent of the code.

The linter can automatically fix many errors it identifies, which can be
helpful for formatting the more rote stylistic issues:

.. code-block:: console

   $ ruff check --fix .


Docs
====
Build the docs with `Sphinx
<https://www.sphinx-doc.org/en/master/index.html>`_.

First, generate the API reference documentation:

.. code-block:: console

   $ rm -rf docs/reference/  # delete existing files if necessary
   $ SPHINX_APIDOC_OPTIONS='members' \
     sphinx-apidoc \
       --maxdepth 1 \
       --module-first \
       --no-toc \
       --separate \
       --output docs/reference/ \
       src/opda/

Then, build the documentation:

.. code-block:: console

    $ sphinx-build -M html docs/ docs/_build/ --jobs auto -W --keep-going

Finally, serve the documentation locally using Python's
:py:mod:`http.server`:

.. code-block:: console

   $ python -m http.server --directory docs/_build/html

Now, you can navigate in your browser to the printed URL in order to
view the docs.

To validate the documentation, check for broken links using
:py:mod:`~sphinx.builders.linkcheck`:

.. code-block:: console

    $ sphinx-build -M linkcheck docs/ docs/_build/ --jobs auto -W --keep-going

And test the documentation's correctness by executing examples as
:py:mod:`doctests <doctest>`:

.. code-block:: console

   $ pytest --doctest-modules --doctest-glob "**/*.rst" -- README.rst docs/ src/

``--doctest-modules`` runs doctests from the docstrings in any python
modules, while ``--doctest-globs "**/*.rst"`` searches reStructuredText
files for doctests. The arguments (``README.rst docs/ src/``) ensure
pytest looks at the right paths for these tests.
