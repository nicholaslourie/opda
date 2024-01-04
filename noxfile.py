"""Test across versions using nox."""

import datetime
import itertools
import json
import pathlib
import re
import shutil
import urllib.request

from lxml import etree
import nox

# backwards compatibility

# ruff: isort: off
import sys

if sys.version_info < (3, 11, 0):
    import tomli as tomllib
else:
    import tomllib
# ruff: isort: on


# constants

ROOT = pathlib.Path(__file__).resolve().parent
"""The repository root."""

SUPPORTED_PYTHON_VERSIONS = {"3.8", "3.9", "3.10", "3.11", "3.12"}
"""All Python versions currently supported by opda."""

SUPPORTED_PACKAGE_VERSIONS = {
    "numpy": {"1.21", "1.22", "1.23", "1.24", "1.25", "1.26"},
    "scipy": {"1.8", "1.9", "1.10", "1.11", "1.12"},
}
"""All core package versions currently supported by opda."""

DEFAULT_PYTHON_VERSION = min(
    SUPPORTED_PYTHON_VERSIONS,
    key=lambda version: tuple(map(int, version.split("."))),
)
"""The default python version to use in tests."""

requires_python_regex = re.compile(r"^>=\s*(\d+\.\d+)$")
"""A regex for the expected requires-python format in pyproject.toml."""

version_regex = re.compile(r"(\d+\.\d+)(\.\d+(a|b|rc)\d+)?")
"""A regex for the expected version format."""

dependency_regex = re.compile(r"([A-Za-z][A-Za-z0-9_]*) >= (\d+.\d+)")
"""A regex for the expected dependency specifier format."""


# helper functions

def sorted_versions(vs):
    """Return a sorted list of the version strings from ``vs``."""
    return sorted(vs, key=lambda v: tuple(map(int, v.split("."))))


def fetch_supported_python_versions():
    """Fetch from python.org all python versions opda should support.

    opda's support policy is to maintain compatibility with all python
    versions that still receive security updates.
    """
    # Get currently supported python versions from the downloads page.
    request = urllib.request.urlopen("https://www.python.org/downloads/")
    root = etree.parse(request, parser=etree.HTMLParser())

    supported_versions = set()
    for element in root.xpath(
            "//div[contains(@class, 'active-release-list-widget')]"
            "/ol/li[span[@class='release-status']/text() != 'prerelease']"
            "/span[@class='release-version']",
    ):
        python_version = element.text
        feature_version, *_ = version_regex.match(python_version).groups()

        supported_versions.add(feature_version)

    return supported_versions


def fetch_supported_package_versions(package):
    """Fetch from PyPI the package versions opda should support.

    opda's support policy is to maintain compatibility with core
    packages' feature releases for 2 years.
    """
    request = urllib.request.urlopen(  # noqa: S310
        f"https://pypi.org/pypi/{package}/json",
    )
    version_to_release = json.load(request)["releases"]

    supported_versions = set()
    for version, release in version_to_release.items():
        support_date = (
            datetime.datetime.now().astimezone()
            - datetime.timedelta(days=2 * 365)
        )
        upload_date = max(
            (
                datetime.datetime.fromisoformat(
                    upload["upload_time_iso_8601"].replace("Z", "+00:00"),
                )
                for upload in release
                if not upload["yanked"]
            ),
            default=None,
        )

        if upload_date is None or upload_date < support_date:
            continue

        feature_version, *_ = version_regex.match(version).groups()

        supported_versions.add(feature_version)

    return supported_versions


# nox configuration

# Use Python's built-in virtual environment backend.
nox.options.default_venv_backend = "venv"

# Reuse existing virtual environments to speed up local runs.
nox.options.reuse_existing_virtualenvs = True

# Throw an error rather than skip the session if the interpreter for the
# Python version is missing.
nox.options.error_on_missing_interpreters = True

# Throw an error rather than log a warning if a command from outside the
# virtual environment is run without passing ``external=True``.
nox.options.error_on_external_run = True


# sessions

@nox.session(python=DEFAULT_PYTHON_VERSION)
def support(session):
    """Check opda's compliance with its support policy."""
    # Verify supported versions match those defined in noxfile.py.
    support_errors = []
    #   .default-python-version
    default_python_version = ROOT.joinpath(
        ".default-python-version",
    ).read_text().strip()
    if default_python_version != DEFAULT_PYTHON_VERSION:
        support_errors.append(
            f"MISMATCH Python {default_python_version} is the default"
            f" version in .default-python-version but is not"
            f" DEFAULT_PYTHON_VERSION in noxfile.py.",
        )
    #   pyproject.toml
    pyproject = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text())
    #     requires-python
    (pyproject_requires_python,) = requires_python_regex.match(
        pyproject["project"]["requires-python"],
    ).groups()
    if pyproject_requires_python != DEFAULT_PYTHON_VERSION:
        support_errors.append(
            f"MISMATCH Python {pyproject_requires_python} is the"
            f" minimum version in pyproject.toml but is not"
            f" DEFAULT_PYTHON_VERSION in noxfile.py.",
        )
    #     python version classifiers
    pyproject_python_versions = set()
    for classifier in pyproject["project"]["classifiers"]:
        *prefix, python_version = classifier.split(" :: ")
        if prefix != ["Programming Language", "Python"]:
            # Skip classifiers unrelated to Python versions.
            continue
        if "." not in python_version:
            # Skip solely major versions (e.g., Python 3 vs. Python 3.2).
            continue
        pyproject_python_versions.add(python_version)
    if pyproject_python_versions != SUPPORTED_PYTHON_VERSIONS:
        for python_version in (
                pyproject_python_versions - SUPPORTED_PYTHON_VERSIONS
        ):
            support_errors.append(
                f"MISMATCH Python {python_version} is in the"
                f" classifiers from pyproject.toml but not in"
                f" SUPPORTED_PYTHON_VERSIONS from noxfile.py.",
            )
        for python_version in (
                SUPPORTED_PYTHON_VERSIONS - pyproject_python_versions
        ):
            support_errors.append(
                f"MISMATCH Python {python_version} is in"
                f" SUPPORTED_PYTHON_VERSIONS from noxfile.py but not in"
                f" the classifiers from pyproject.toml.",
            )
    #     package versions
    for dependency in pyproject["project"]["dependencies"]:
        dependency_match = dependency_regex.match(dependency)
        if dependency_match is None:
            support_errors.append(
                f"FIX {dependency}'s dependency specification in"
                f" pyproject.toml. Only use the >= operator against"
                f" minor version numbers.",
            )
            continue
        package, version = dependency_match.groups()
        min_supported_version = min(
            SUPPORTED_PACKAGE_VERSIONS[package],
            key=lambda version: tuple(map(int, version.split("."))),
        )
        if version != min_supported_version:
            support_errors.append(
                f"MISMATCH minimum supported version for {package}"
                f" differs between noxfile SUPPORTED_PACKAGE_VERSIONS"
                f" and pyproject.toml dependencies.",
            )

    # Report any version mismatches.
    if support_errors:
        session.error(
            "Fix the following errors:\n"
            + "\n".join(f"  * {error}" for error in support_errors),
        )

    # Determine if maintenance is necessary to comply with the support policy.
    maintenance_actions = []
    #   python versions
    policy_python_versions = fetch_supported_python_versions()
    for python_version in (
            policy_python_versions - SUPPORTED_PYTHON_VERSIONS
    ):
        maintenance_actions.append(
            f"ADD Python {python_version} to classifiers in"
            f" pyproject.toml.",
        )
        maintenance_actions.append(
            f"ADD Python {python_version} to SUPPORTED_PYTHON_VERSIONS"
            f" in noxfile.py.",
        )
    for python_version in (
            SUPPORTED_PYTHON_VERSIONS - policy_python_versions
    ):
        maintenance_actions.append(
            f"DROP Python {python_version} from classifiers in"
            f" pyproject.toml.",
        )
        maintenance_actions.append(
            f"DROP Python {python_version} from SUPPORTED_PYTHON_VERSIONS"
            f" in noxfile.py.",
        )
        maintenance_actions.append(
            "UPDATE .default-python-version file.",
        )
        maintenance_actions.append(
            "UPDATE requires-python in pyproject.toml.",
        )
    #   core package versions
    for package in SUPPORTED_PACKAGE_VERSIONS.keys():
        policy_package_versions = fetch_supported_package_versions(package)
        for package_version in (
                policy_package_versions - SUPPORTED_PACKAGE_VERSIONS[package]
        ):
            maintenance_actions.append(
                f"ADD {package} {package_version} to"
                f" SUPPORTED_PACKAGE_VERSIONS in noxfile.py.",
            )
        for package_version in (
                SUPPORTED_PACKAGE_VERSIONS[package] - policy_package_versions
        ):
            maintenance_actions.append(
                f"DROP {package} {package_version} from dependencies in"
                f" pyproject.toml.",
            )
            maintenance_actions.append(
                f"DROP {package} {package_version} from"
                f" SUPPORTED_PACKAGE_VERSIONS in noxfile.py.",
            )

    # Report any required maintenance.
    if maintenance_actions:
        session.error(
            "Update the package to comply with the support policy:\n"
            + "\n".join(f"  * {action}" for action in maintenance_actions),
        )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def docs(session):
    """Build and test the documentation."""
    session.install("pip >= 21.2")  # backwards compatibility

    # Build the documentaiton.

    session.install(".[docs]")
    # Ensure no files are left over from previous builds.
    build_dpath = ROOT / "docs" / "_build"
    if build_dpath.exists():
        shutil.rmtree(build_dpath)
    reference_dpath = ROOT / "docs" / "reference"
    if reference_dpath.exists():
        shutil.rmtree(reference_dpath)
    # Generate the API reference documentation source.
    session.run(
        "python", "-Im", "sphinx.ext.apidoc",
        "--force",
        "--separate",
        "--no-toc",
        "--maxdepth", "1",
        "--module-first",
        "--output-dir", "docs/reference/",
        "--",
        "src/opda/",
        env={"SPHINX_APIDOC_OPTIONS": "members"},
    )
    # Build the documentation.
    session.run(
        "python", "-Im", "sphinx",
        "--jobs", "auto",
        "-T",  # print full tracebacks on errors
        "-W",  # turn warnings into errors
        "--keep-going",
        "-a",  # write all files
        "-E",  # re-read all environment files
        "-d", "docs/_build/doctrees/",
        "-b", "html",
        "--",
        "docs/",
        "docs/_build/html/",
    )

    # Test the documentation.

    session.install(".[tests]")
    # Check for broken links.
    session.run(
        "python", "-Im", "sphinx",
        "--jobs", "auto",
        "-T",  # print full tracebacks on errors
        "-W",  # turn warnings into errors
        "--keep-going",
        "-a",  # write all files
        "-E",  # re-read all environment files
        "-d", "docs/_build/doctrees/",
        "-b", "linkcheck",
        "--",
        "docs/",
        "docs/_build/linkcheck/",
    )
    # Run doctests.
    # NOTE: doctests are only for checking that examples in the
    # documentation are correct. We assume that the package itself
    # works if it passes the regular tests. Thus, we only need to run
    # doctests with one version of python and the dependencies.
    session.run(
        "pytest",
        "--doctest-modules",
        "--doctest-glob", '"**/*.rst"',
        "--",
        "README.rst",
        "docs/",
        "src/",
    )


@nox.session(python=DEFAULT_PYTHON_VERSION)
def lint(session):
    """Run lint."""
    session.install("pip >= 21.2")  # backwards compatibility

    session.install(".[lint]")
    session.run("ruff", "check", ".")


@nox.session(python=sorted_versions(SUPPORTED_PYTHON_VERSIONS))
@nox.parametrize(
    list(SUPPORTED_PACKAGE_VERSIONS.keys()),
    list(itertools.product(
        *map(sorted_versions, SUPPORTED_PACKAGE_VERSIONS.values()),
    )),
)
def test(session, **kwargs):
    """Run tests."""
    session.install("pip >= 21.2")  # backwards compatibility

    # Check that dependencies are compatible.
    output = session.run(
        "pip",
        "install",
        "--ignore-install",
        "--dry-run",
        *(f"{package}=={version}" for package, version in kwargs.items()),
        success_codes=[0, 1],
        silent=True,
    )
    if "ERROR:" in output and (
            "No matching distribution found" in output
            or "package versions have conflicting dependencies" in output
    ):
        session.skip(
            "Skip this session because the dependencies are incompatible.",
        )

    # Run the tests.
    session.install(
        ".[tests]",
        *(f"{package}=={version}" for package, version in kwargs.items()),
    )
    session.run("pytest", "--all-levels")


# Define which sessions to run by default.
nox.options.sessions = [
    "lint",
    "docs",
    f"test-{DEFAULT_PYTHON_VERSION}({test.parametrize[0]})",
]
