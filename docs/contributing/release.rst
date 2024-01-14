=======
Release
=======
To make a new release, follow the process below.


Before a Release
================
Before making a release, review the following maintenance tasks:

1. Consider updating the development dependencies in
   :source-file:`pyproject.toml`, especially the packaging tools:
   `build <https://build.pypa.io/en/stable/>`_ and `twine
   <https://twine.readthedocs.io/en/stable/>`_.
2. Check if any backwards compatability code can be dropped. Run:

   .. code-block:: console

      $ { \
           echo "file:line number:comment"; \
           echo "----:-----------:-------"; \
           git grep \
             --line-number \
             --only-matching \
             --ignore-case \
             --recursive \
             -- \
             "# backwards compatibility.*$" \
             . \
           | grep --invert-match "\.rst:"; \
        } | column -t -s':'

   The command produces a table summarizing all backwards
   compatibility code in the codebase.


Making a Release
================
Decide the next version based off the :doc:`Changelog </changelog>`
using `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. Let
``${VERSION}`` denote the version to release. For example:

.. code-block:: console

   $ VERSION="v1.0.0"

Then:

1.  Create a new branch:

    .. code-block:: console

       $ git checkout -b "bump-version-to-${VERSION}"

2.  Update and commit any changes to :source-file:`CHANGELOG.rst`:

    a. Clean up and combine existing entries if appropriate.
    b. Follow the instructions from the source code comment in
       :source-file:`CHANGELOG.rst` to finalize the "Unreleased"
       section for a new release.

3.  Update the ``project.version`` key to ``${VERSION}`` in
    :source-file:`pyproject.toml`, and commit the change with the
    message: ``"Bump version to ${VERSION}"``.
4.  Open and merge a pull request for the branch.
5.  Update the ``main`` branch:

    .. code-block:: console

       $ git checkout main && git pull
       $ git branch --delete "bump-version-to-${VERSION}" && git fetch --prune

    Tag the **merge** commit with ``git``:

    .. code-block:: console

       $ git tag --annotate "${VERSION}" --message "Release ${VERSION}"

    Then push the tag back to the remote:

    .. code-block:: console

       $ git push origin "${VERSION}"

6.  Check that the :repo-workflow:`release workflow <release.yml>`
    completes successfully.
7.  View the documentation that the :repo-workflow:`release workflow
    <release.yml>` just deployed.
8.  Download the package artifact, ``dist``, that the
    :repo-workflow:`release workflow <release.yml>` built for the tag.
    You may also inspect the packages' contents by viewing the logs from
    ``Build and test artifacts. > Package the source. > List the
    packages' contents.``.

    Once you have the artifact, unzip it in your repository to the
    ``dist/`` directory:

    .. code-block:: console

       $ rm -rf dist/
       $ unzip -d "dist/" dist.zip && rm dist.zip

9.  Ensure the ``TWINE_USERNAME``, ``TWINE_TESTPYPI_PASSWORD``, and
    ``TWINE_PYPI_PASSWORD`` environment variables are available. See
    the `PyPA documentation
    <https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives>`_
    for details. Then, run the ``release`` session with ``nox``:

    .. code-block:: console

       $ nox --session release -- dist/

    This session will upload the package to `TestPyPI
    <https://test.pypi.org/>`_, validate it from TestPyPI, then upload
    to `PyPI <https://pypi.org/>`_, and finally validate it from PyPI.
10. Make a :repo-release:`GitHub Release <new>` out of the new tag.

    a. Select the tag for the release.
    b. Use the tag's name, ``${VERSION}``, for the release title.
    c. Copy and paste the changelog into the release description. Edit
       the text so that it's properly formatted.
    d. Acknowledge anyone who contributed to the release at the end of
       the description.
    e. Submit the release.

Congratulations! The release is complete!
