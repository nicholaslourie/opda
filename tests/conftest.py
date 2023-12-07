"""Test configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--level', type=int, default=0,
        help='Run test suite levels up to and including this level.',
    )
    parser.addoption(
        '--all-levels', action='store_true', default=False,
        help='Run all test suite levels.',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'level(level): Mark the test to only be run at this level.',
    )


def pytest_runtest_setup(item):
    level = item.config.getoption('--level')
    all_levels = item.config.getoption('--all-levels')

    item_levels = [mark.args[0] for mark in item.iter_markers(name='level')]
    if len(item_levels) == 0:
        item_level = None
    elif len(item_levels) == 1:
        item_level = item_levels[0]
    else:
        raise RuntimeError('The test was marked with multiple levels.')

    if (
            not all_levels
            and item_level is not None
            and item_level > level
    ):
        pytest.skip(
            'The test\'s level is higher than the current level being run.',
        )
