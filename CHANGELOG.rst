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

           .. _unreleased: https://github.com/nicholaslourie/opda/compare/${VERSION}...HEAD


`Unreleased`_
=============
.. rubric:: Additions

* Add support for ``numpy == 2.2`` and ``scipy == 1.15``.

.. rubric:: Changes
.. rubric:: Deprecations
.. rubric:: Removals

* Drop support for ``numpy == 1.23`` and ``scipy == 1.10``.
* Drop support for Python 3.8.

.. rubric:: Fixes

* Fix the support policy check in continuous integration:

  * Determine currently supported python versions more correctly.
  * Mention updating the setup documentation when versions change.

* Suppress expected deprecation warnings that ``autograd`` fires when
  installed with certain versions of ``numpy``.
* Make ``opda.parametric.NoisyQuadraticDistribution.fit`` optimize
  more robustly.
* Fix flakiness in various tests.

.. rubric:: Documentation
.. rubric:: Security


`v0.7.0`_ - 2024-11-11
======================
.. rubric:: Additions

* Add the hyperparameter tuning scaling results from tuning ResNet18
  for ImageNet via random search.

  * Add the data itself: ``data/resnet/resnet18_scaling.results.jsonl``.
  * Add a license for the data: ``data/resnet/LICENSE``.
  * Add a README for the data: ``data/resnet/README.md``.

* Add support for ``scipy == 1.14`` and ``numpy == 2.1``.
* Add the ``C_MIN`` and ``C_MAX`` class attributes to
  ``opda.parametric.QuadraticDistribution`` and
  ``opda.parametric.NoisyQuadraticDistribution``.
* Add ``opda.utils.normal_ppf``, a fast function for computing the
  standard normal's PPF (quantile function).
* Check that ``c`` is finite when validating the arguments to
  ``opda.parametric.QuadraticDistribution`` and
  ``opda.parametric.NoisyQuadraticDistribution``.
* Add ``fit`` methods to the parametric distributions:
  ``opda.parametric.QuadraticDistribution.fit`` and
  ``opda.parametric.NoisyQuadraticDistribution.fit``.
* Add support for Python 3.13.

.. rubric:: Changes

* Simplify the implementation of
  ``opda.parametric.NoisyQuadraticDistribution`` by making ``scale``
  the standard deviation of the corresponding normal random variable.
* Upgrade packaging tools to ``build == 1.2.1`` and ``twine == 5.1.1``.
* Skip some tests for ``opda.utils.beta_equal_tailed_interval``,
  ``opda.utils.beta_highest_density_interval``, and
  ``opda.utils.beta_equal_tailed_coverage`` when they fail due to
  `an issue <https://github.com/scipy/scipy/issues/21303>`_  in
  ``scipy == 1.14.0`` on Linux that causes spurious NaN values.
* Use the point mass, noiseless, and normal approximations to speed up
  the ``ppf`` method of ``opda.parametric.NoisyQuadraticDistribution``.
* Disable lint rules:

  * ``B023``: "Function definition does not bind loop variable
    {name}".
  * ``PLW2901``: "Outer {outer_kind} variable {name} overwritten by
    inner {inner_kind} target".
  * ``SIM102``: "Use a single ``if`` statement instead of nested
    ``if`` statements".

* Restrict the range of values for ``c`` supported by
  ``opda.parametric.QuadraticDistribution`` to ``1`` to ``10``.

.. rubric:: Removals

* Drop support for ``numpy == 1.21``, ``numpy == 1.22``,
  ``scipy == 1.8``, and  ``scipy == 1.9``.
* Remove the ``estimate_initial_parameters_and_bounds`` method from
  ``opda.parametric.QuadraticDistribution``.

.. rubric:: Fixes

* Update tests for the ``__repr__`` methods of
  ``opda.nonparametric.EmpiricalDistribution``,
  ``opda.parametric.QuadraticDistribution``, and
  ``opda.parametric.NoisyQuadraticDistribution`` so that they're
  compatible with ``numpy == 2.0``.
* Fix flakiness in the test:
  ``NoisyQuadraticDistributionTestCase.test_ppf_is_inverse_of_cdf``.
* Fix the ``quantile_tuning_curve`` methods of
  ``opda.parametric.QuadraticDistribution`` and
  ``opda.parametric.NoisyQuadratricDistribution`` which gave incorrect
  output when ``q`` wasn't equal to ``0.5`` and either ``minimize`` was
  ``True`` or ``minimize`` was ``None`` and ``convex`` was ``True``.

.. rubric:: Documentation

* Correct the quantile function's definition in the docstrings for the
  ``ppf`` methods of ``opda.parametric.QuadraticDistribution`` and
  ``opda.parametric.NoisyQuadraticDistribution``.
* Make docstring formatting more consistent:

  * Always use "Returns" and never "Return" for the docstring section
    header.
  * Always document optional types with "or None" coming at the *end* of
    the type description.
  * Always document each returned value with a separate subheading when
    a function or method returns multiple values.

* Remove default values from parameters' type descriptions.


`v0.6.1`_ - 2024-04-03
======================
.. rubric:: Additions

* Add support for ``numpy == 2.0`` and ``scipy == 1.13``.

.. rubric:: Changes

* Disable lint rule ``PLR1714``: "Consider merging multiple
  comparisons".

.. rubric:: Documentation

* Split the Jupyter notebook *Evaluating DeBERTaV3 with the
  Nonparametric Analysis* into several smaller more focused notebooks:

  * *Evaluating DeBERTaV3 with the Nonparametric Analysis*
  * *Choosing a Sample Size for the Nonparametric Analysis*
  * *Demonstrating the Exact Coverage of the Nonparametric Analysis*
  * *Studying Ablations of the Nonparametric Analysis*

* Add a section about how to analyze a hyperparameter (the number of
  epochs) to the Jupyter notebook *Evaluating DeBERTaV3 with the
  Nonparametric Analysis*.
* In the *Examples* doc (``docs/tutorial/examples.rst``), improve the
  code, wording, and title of the model comparison example (previously
  titled *Compare Models' Tuning Curves*, now titled *Compare Models*).
* In the *Examples* doc (``docs/tutorial/examples.rst``), expand the
  *Compare Models* example with discussion on how to compare models with
  different training costs.
* Fix incorrect markup in the *Examples* doc
  (``docs/tutorial/examples.rst``).
* Add a new example showing how to analyze a hyperparameter in the
  *Examples* doc (``docs/tutorial/examples.rst``).
* Add plots of DeBERTa and DeBERTaV3's tuning curves with confidence
  bands using large sample sizes (1,024) to the Jupyter notebook
  *Evaluating DeBERTaV3 with the Nonparametric Analysis*.


`v0.6.0`_ - 2024-03-04
======================
.. rubric:: Additions

* Add ``opda.approximation``, a module for approximation-theoretic
  operations, with the following functions:

  * ``opda.approximation.lagrange_interpolate``: Interpolate points with
    a polynomial.
  * ``opda.approximation.remez``: Identify the reference corresponding
    to the minimax polynomial approximation of a function.
  * ``opda.approximation.minimax_polynomial_approximation``: Evaluate
    the minimax polynomial approximation of a function.
  * ``opda.approximation.minimax_polynomial_coefficients``: Compute the
    coefficients of the minimax polynomial approximation of a function.
  * ``opda.approximation.piecewise_polynomial_knots``: Find the knots
    for the minimax piecewise polynomial approximation of a function.

* Add ``opda.exceptions.NumericalError``, an exception for numerical
  issues.
* Add more tests for ``opda.parametric.QuadraticDistribution``.
* Add ``mean`` and ``variance`` attributes to
  ``opda.nonparametric.EmpiricalDistribution``.
* Add ``mean`` and ``variance`` attributes to
  ``opda.parametric.QuadraticDistribution``.
* Add ``opda.utils.normal_pdf``, a fast function for computing the
  standard normal's PDF.
* Add ``opda.utils.normal_cdf``, a fast function for computing the
  standard normal's CDF.
* Add ``opda.exceptions.IntegrationError``, an exception for
  integration issues.
* Add ``opda.parametric.NoisyQuadraticDistribution``, a probability
  distribution representing a quadratic random variable plus normal
  noise.
* Increase argument validation in ``opda.utils.dkw_epsilon``.
* Add more test cases for ``opda.utils.dkw_epsilon``.
* Validate that ``lightness`` is between 0 and 1 (inclusive) in
  ``experiments.visualization.color_with_lightness``.
* Validate that arguments are *finite* floats where appropriate.

.. rubric:: Changes

* Reparametrize ``parametric.QuadraticDistribution`` so ``c`` is the
  effective number of hyperparameters instead of half the number.
* Completely disable the eradicate (``ERA``) lint rules.
* Enable ``"py"`` as the primary domain in the documentation.
* Always use numpy's numeric types for scalar class attributes,
  instead of Python's native numeric types.
* Improve tests for ``parametric.QuadraticDistribution``, making them
  more thorough, robust, and avoiding re-running redundant test cases.
* Update the tests for ``parametric.QuadraticDistribution`` to cover
  the case when ``a == b``.
* Move the source repository from ``github.com/nalourie/opda`` to
  ``github.com/nicholaslourie/opda``, and move the docs from
  ``nalourie.github.io/opda`` to
  ``nicholaslourie.github.io/opda``. Update the project URLs in
  ``pyproject.toml`` and all the links throughout the repository to
  reflect these changes.
* Require ``fraction`` is greater than 0 in
  ``opda.parametric.QuadraticDistribution.estimate_initial_parameters_and_bounds``.
* Throw an error if ``fraction`` is too small and thus causes
  ``opda.parametric.QuadraticDistribution.estimate_initial_parameters_and_bounds`` to
  try and form an estimate from an empty list.

.. rubric:: Fixes

* Fix ``parametric.QuadraticDistribution`` (the ``.pdf``, ``.cdf``,
  and ``.estimate_initial_parameters_and_bounds`` methods) for the
  case when ``a == b``, in which case the distribution is an atom
  (point mass).
* Fix
  ``opda.parametric.QuadraticDistribution.estimate_initial_parameters_and_bounds``
  when ``convex`` is ``False`` and ``fraction`` is small enough so
  that the estimate should be based on an empty list. In this case,
  the method incorrectly uses all of ``ys``. Instead, throw an error
  saying that fraction is too small (as it produces an empty list).
* Avoid throwing an unnecessary warning in ``opda.utils.dkw_epsilon``
  when ``confidence`` is 1.

.. rubric:: Documentation

* Improve the docstring for
  ``experiments.analytic.get_approximation_parameters``.
* Use inline math markup in docstrings.
* Fix the equation in the docstring for
  ``opda.parametric.QuadraticDistribution.ppf``. The infimum that
  defines the quantile function has as its domain the interval from
  ``a`` to ``b``, not the entire real line.
* Add "See Also" and "Notes" sections to the docstring for
  ``opda.parametric.QuadraticDistribution``, matching the newly added
  docstring for ``opda.parametric.NoisyQuadraticDistribution``.
* Update all links to use ``github.com/nicholaslourie`` and
  ``nicholaslourie.github.io/opda`` in place of
  ``github.com/nalourie`` and ``nalourie.github.io/opda``.
* Document stricter dependent type constraints (e.g., non-negativity,
  finiteness) for function and method inputs and outputs.
* Document range constraints for inputs and outputs more precisely and
  consistently (e.g., ``q`` is a float from 0 to 1 inclusive).
* Improve the docstring for ``experiments.analytic.ellipse_volume``.
* Fix docstrings across the code base in order to consistently
  document when a value can take on either scalar (e.g., float) or
  array (e.g., array of floats) values.


`v0.5.0`_ - 2024-01-15
======================
This version is the first uploaded to PyPI and available via ``pip``!

.. rubric:: Additions

* Add a continuous integration job to ensure every pull request
  updates the changelog.
* In the continuous integration job for building the packages, add a
  step to list the packages' contents.
* Add the "release" nox session for making new releases to PyPI.

.. rubric:: Changes

* Upgrade the development dependencies.
* Upgrade the ``Development Status`` PyPI classifier for opda from
  ``3 - Alpha`` to ``4 - Beta``.

.. rubric:: Fixes

* Fix flakiness in the test:
  ``EmpiricalDistributionTestCase.test_average_tuning_curve``.

.. rubric:: Documentation

* Pin links to the source on GitHub to the commit that builds the
  documentation.
* Move development documentation into the "Contributing" section of
  the sidebar and URL tree.
* Omit from the documentation's sidebar any project URLs that link to
  the documentation.
* Add an announcement banner to the documentation when it's built for
  an unreleased version.
* Add a changelog (``CHANGELOG.rst``).
* Document the project's various conventions in the development docs.
* Add the "Release" doc describing the release process.
* Update the docs to suggest installing opda from PyPI rather than the
  source for regular usage.


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

.. _unreleased: https://github.com/nicholaslourie/opda/compare/v0.7.0...HEAD
.. _v0.7.0: https://github.com/nicholaslourie/opda/compare/v0.6.1...v0.7.0
.. _v0.6.1: https://github.com/nicholaslourie/opda/compare/v0.6.0...v0.6.1
.. _v0.6.0: https://github.com/nicholaslourie/opda/compare/v0.5.0...v0.6.0
.. _v0.5.0: https://github.com/nicholaslourie/opda/compare/v0.4.0...v0.5.0
.. _v0.4.0: https://github.com/nicholaslourie/opda/compare/v0.3.0...v0.4.0
.. _v0.3.0: https://github.com/nicholaslourie/opda/compare/v0.2.0...v0.3.0
.. _v0.2.0: https://github.com/nicholaslourie/opda/compare/v0.1.0...v0.2.0
.. _v0.1.0: https://github.com/nicholaslourie/opda/releases/tag/v0.1.0
