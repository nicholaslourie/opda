"""Tests for ersa.utils"""

import unittest

import numpy as np

from ersa import utils


class SortByFirstTestCase(unittest.TestCase):
    """Test ersa.utils.sort_by_first."""

    def test_sort_by_first(self):
        tolist = lambda a: a.tolist()

        # Test on no arguments.
        self.assertEqual(utils.sort_by_first(), ())
        # Test on empty lists.
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([]))),
            ([],)
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([], []))),
            ([], []),
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([], [], []))),
            ([], [], []),
        )
        # Test list of length 1.
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([1.]))),
            ([1.],),
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([1.], [2.]))),
            ([1.], [2.]),
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([1.], [2.], ['a']))),
            ([1.], [2.], ['a']),
        )
        # Test lists of length greater than 1.
        #   when first is sorted
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([1., 2., 3.]))),
            ([1., 2., 3.],),
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first(
                [1., 2., 3.],
                [3., 2., 1.],
            ))),
            (
                [1., 2., 3.],
                [3., 2., 1.],
            ),
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first(
                [1., 2., 3.],
                [3., 2., 1.],
                ['a', 'c', 'b'],
            ))),
            (
                [1., 2., 3.],
                [3., 2., 1.],
                ['a', 'c', 'b'],
            ),
        )
        #   when first is unsorted
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([2., 1., 3.]))),
            ([1., 2., 3.],),
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first(
                [2., 1., 3.],
                [3., 2., 1.],
            ))),
            (
                [1., 2., 3.],
                [2., 3., 1.],
            ),
        )
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first(
                [2., 1., 3.],
                [3., 2., 1.],
                ['a', 'c', 'b'],
            ))),
            (
                [1., 2., 3.],
                [2., 3., 1.],
                ['c', 'a', 'b'],
            ),
        )


class DkwEpsilonTestCase(unittest.TestCase):
    """Test ersa.utils.dkw_epsilon."""

    def test_dkw_epsilon(self):
        self.assertEqual(utils.dkw_epsilon(2, 1. - 2./np.e), 0.5)
        self.assertEqual(utils.dkw_epsilon(8, 1. - 2./np.e), 0.25)
        self.assertEqual(utils.dkw_epsilon(1, 1. - 2./np.e**2), 1.)
        self.assertEqual(utils.dkw_epsilon(4, 1. - 2./np.e**2), 0.5)
