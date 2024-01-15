=========
Changelog
=========
..
  This changelog is included into the docs.

All notable changes to this project will be documented here.

The format is based on `Keep a Changelog
<https://keepachangelog.com/en/1.1.0/>`_, and this project adheres to
`Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

..
  To finalize the "Unreleased" section for a new release:

    1. Change the title to "`${VERSION}`_ - YYYY-MM-DD".
    2. Update the ".. _unreleased:" link definition at the bottom of
       this document, changing "_unreleased" and "HEAD" to the next
       version.
    3. Remove any empty rubric subsections.

  To create a new "Unreleased" section:

    1. Copy the following template and paste it above the latest
       release:

           `Unreleased`_
           =============
           .. rubric:: Additions
           .. rubric:: Changes
           .. rubric:: Deprecations
           .. rubric:: Removals
           .. rubric:: Fixes
           .. rubric:: Documentation
           .. rubric:: Security

    2. Add the following link defintion above the others at the bottom
       of this document, and replace ${VERSION} in it with the most
       recent version:

           .. _unreleased: https://github.com/nalourie/opda/compare/${VERSION}...HEAD


`Unreleased`_
=============
.. rubric:: Additions

* Add a continuous integration job to ensure every pull request
  updates the changelog.
* In the continuous integration job for building the packages, add a
  step to list the packages' contents.

.. rubric:: Changes
.. rubric:: Deprecations
.. rubric:: Removals
.. rubric:: Fixes

* Fix flakiness in the test:
  ``EmpiricalDistributionTestCase.test_average_tuning_curve``.

.. rubric:: Documentation

* Pin links to the source on GitHub to the commit that built the
  documentation.
* Move development documentation into the "Contributing" section of
  the sidebar and URL tree.
* Omit project URLs that link to the documentation from the
  documentation's sidebar.
* Add an announcement banner to the documentation when it's built for
  an unreleased version.
* Add a changelog (``CHANGELOG.rst``).
* Document conventions used in the project in the development docs.

.. rubric:: Security


`v0.4.0`_ - 2024-01-10
======================
.. rubric:: Additions

* Add the ``package`` optional dependencies.
* Add a build for "distribution" as opposed to "local" use. The
  distribution package contains only the ``opda`` library and not
  ``experiments``.

  * Add a ``nox`` session for building the distribution package.
  * Add a continuous integration job to build the package and store it
    as an artifact on each pull request.
  * Add a continuous integration job to test the distribution package
    against all combinations of supported versions of major
    dependencies.

.. rubric:: Changes

* Increase retention for documentation build artifacts from 60 to 90
  days in continuous integration.
* Prune each set of optional dependencies.
* Rename the ``tests`` optional dependencies to ``test``.
* Split the ``test`` session in ``nox`` into ``test``, for testing the
  local project, and ``testpackage``, for testing distribution packages.
* In continuous integration, only test the local build against *default*
  versions of major dependencies, since we now build and test the
  distribution package against *all* combinations of supported versions.

.. rubric:: Documentation

* Document how to build and test the distribution package.


`v0.3.0`_ - 2024-01-07
======================
.. rubric:: Additions

* Extend ``nonparametric.QuadraticDistribution.sample`` and
  ``nonparametric.EmpiricalDistribution.sample`` to return a scalar when
  ``size=None``, and make it the default argument.
* Add documentation builds via Sphinx:

  * Create a Sphinx setup for building the documentation.
  * Add tutorial-style documentation for users.
  * Add development documentation.
  * Automatically generate API reference documentation.

* Add a GitHub Actions workflow for building and publishing the
  documentation to GitHub Pages.
* Make tests backwards compatible with ``numpy >= 1.21``.
* Adjust package dependency requirements to allow ``numpy >= 1.21`` and
  ``scipy >= 1.8``.
* Add ``ci`` optional dependencies for continuous integration.
* Add ``nox`` for automating development tasks, testing against all
  supported major dependencies, and continuous integration.
* Add a GitHub Actions workflow for continuous integration. Run it on
  each pull request as well as every calendar quarter. Use the
  workflow to:

  * Check ``opda``'s major dependency versions are up-to-date.
  * Lint the project.
  * Build and test the documentation.
  * Test the project against all combinations of supported versions of
    major dependencies.

.. rubric:: Changes

* Always return scalars rather than 0 dimensional arrays from methods
  (``nonparametric.EmpiricalDistribution.pmf`` and
  ``parametric.QuadraticDistribution.pdf``).
* Explicitly test that all methods of
  ``nonparametric.EmpiricalDistribution`` and
  ``parametric.QuadraticDistribution`` return scalars rather than 0D
  arrays.
* Configure ``pytest`` to always use a non-interactive backend for
  ``matplotlib``.
* Update the project URLs in packaging.
* Split out the ``experiments`` package's dependencies as optional
  dependencies.

.. rubric:: Fixes

* Include ``src/experiments/default.mplstyle`` in the package data for
  the experiments package so the style can be used from non-editable
  installs.
* Make tests more robust to changes in rounding errors across
  environments by replacing some equality checks with near equality.

.. rubric:: Documentation

* Remove broken references to the sections of numpy-style
  docstrings. Standard tooling doesn't make these sections linkable.
* Fix errors in the docstrings' markup.
* Use cross-references in the docs wherever possible and appropriate.
* Use proper markup for citations.
* Change the language from ``bash`` to ``console`` in code blocks.
* Improve the modules' docstrings.
* Rewrite ``README.rst``, adding a "Quickstart" section and moving much
  of the old content into new tutorial-style documentation built with
  Sphinx.
* Document how to build and test the documentation.
* Document how to setup and use ``nox`` for common development tasks.


`v0.2.0`_ - 2023-12-16
======================
.. rubric:: Additions

* Add backwards compatibility for Python 3.8.
* Add ``pyproject.toml`` for building the project, replacing the
  ``setup.py`` based build.
* Add and increase argument validation in functions and methods.
* Add the ``--all-levels`` pytest flag for running all tests.
* Add new tests for ``nonparametric.EmpiricalDistribution`` and
  ``parametric.QuadraticDistribution``.
* Give all tuning curve methods a new parameter, ``minimize``, for
  computing *minimizing* hyperparameter tuning curves.

  * ``nonparametric.EmpiricalDistribution`` methods:
    ``quantile_tuning_curve``, ``average_tuning_curve``,
    ``naive_tuning_curve``, ``v_tuning_curve``, and
    ``u_tuning_curve``.
  * ``parametric.QuadraticDistribution`` methods:
    ``quantile_tuning_curve``, and ``average_tuning_curve``.

* Add ``__repr__``, ``__str__``, and ``__eq__`` methods to
  ``nonparamatric.EmpiricalDistribution`` and
  ``parametric.QuadraticDistribution``.
* Add a ``generator`` parameter to set the random seed in functions
  and methods using randomness
  (``experiments.simulation.Simulation.run``,
  ``experiments.visualization.plot_random_search``,
  ``nonparametric.EmpiricalDistribution.confidence_bands``,
  ``nonparametric.EmpiricalDistribution.sample``, and
  ``parametric.QuadraticDistribution.sample``).
* Add the ``opda.random`` module to migrate off of numpy's legacy API
  for random numbers while still enabling control of ``opda``'s
  global random state via ``opda.random.set_seed``.
* Add the ``RandomTestCase`` class for making tests using randomness
  reproducible.
* Configure ``ruff`` for linting the project.

.. rubric:: Changes

* Require ``pytest >= 6`` for running tests.
* Configure ``pytest`` to use the ``tests/`` test path.
* Use ``Private :: Do Not Upload`` classifier to prevent the package
  from being uploaded to PyPI.
* Speed up coverage tests for
  ``nonparametric.EmpiricalDistribution.confidence_bands``.
* Rename optional dependencies from ``dev`` to ``tests``.
* Standardize the error messages for violating argument type
  constraints.
* Expand existing tests to cover more cases for
  ``EmpiricalDistribution`` and ``QuadraticDistribution``.
* Rename ``exceptions.OptimizationException`` to
  ``exceptions.OptimizationError``.
* Use ``TypeError`` in place of ``ValueError`` for type errors.
* Across all functions and methods, standardize which parameters are
  keyword-only. Reserve keyword-only status for rarely used arguments,
  such as implementation details like optimization tolerances.
* Disallow ``None`` as an argument for the ``a`` and ``b`` parameters
  of ``nonparametric.EmpiricalDistribution``.

.. rubric:: Fixes

* Fix flakiness in various tests.
* Ensure ``utils.beta_highest_density_interval`` always returns an
  interval containing the mode, even for very small intervals.
* Fix bug in ``nonparametric.EmpiricalDistribution.confidence_bands``
  that caused coverage to be too high, especially given small samples.
* Improve coverage tests for
  ``nonparametric.EmpiricalDistribution.confidence_bands`` so that
  they're more sensitive and explicitly test small sample sizes.
* Prevent warnings during expected use of various methods of
  ``QuadraticDistribution``.
* Suppress expected warnings in tests.
* Fix ``parametric.QuadraticDistribution.quantile_tuning_curve`` which
  would throw an exception when the instance had ``convex=True``.
* Fix tests for ``parametric.QuadraticDistribution`` so that they
  actually check all intended cases.

.. rubric:: Removals

* Remove the ``setup.py`` based build and associated files
  (``setup.py``, ``setup.cfg``, ``MANIFEST.in``, and
  ``requirements.txt``), replacing it with ``pyproject.toml``.

.. rubric:: Documentation

* Add sections and improve markup in ``README.rst``.
* Add links to and citations for `Show Your Work with Confidence
  <https://arxiv.org/abs/2311.09480>`_.
* Add sections, update content, and improve markup in existing
  docstrings.
* Document development tools for the project.
* Begin running doctests on all documentation.

  * Document how to run doctests in ``README.rst``.
  * Set the random seed in documentation examples to make them testable.
  * Fix errors in examples discovered via doctests.

* Document ``pip`` version requirements for editable installs in
  ``README.rst``.
* Document type constraints (e.g., non-negative integers as opposed to
  integers) in functions and methods' docstrings.
* Document the ``atol`` parameter of
  ``utils.beta_highest_density_interval`` and
  ``utils.highest_density_coverage``.


`v0.1.0`_ - 2023-11-14
======================
.. rubric:: Additions

* Initial release.


..
  Link Definitions

.. _unreleased: https://github.com/nalourie/opda/compare/v0.4.0...HEAD
.. _v0.4.0: https://github.com/nalourie/opda/compare/v0.3.0...v0.4.0
.. _v0.3.0: https://github.com/nalourie/opda/compare/v0.2.0...v0.3.0
.. _v0.2.0: https://github.com/nalourie/opda/compare/v0.1.0...v0.2.0
.. _v0.1.0: https://github.com/nalourie/opda/releases/tag/v0.1.0
