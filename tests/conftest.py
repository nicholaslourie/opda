"""Test configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--level', type=int, default=0,
        help='Run test suite levels up to and including this level.',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'level(level): Mark the test to only be run at this level.',
    )


def pytest_runtest_setup(item):
    level = item.config.getoption('--level')
    item_levels = [mark.args[0] for mark in item.iter_markers(name='level')]
    if item_levels and all(item_level > level for item_level in item_levels):
        pytest.skip(
            'The test\'s level is higher than the current level being run.'
        )
