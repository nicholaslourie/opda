"""Tests for ersa.utils"""

import unittest

import numpy as np
import pytest
from scipy import stats

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


class BetaEqualTailedIntervalTestCase(unittest.TestCase):
    """Test ersa.utils.beta_equal_tailed_interval."""

    def test_beta_equal_tailed_interval(self):
        # Test when a and b are scalars.
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                beta = stats.beta(a, b)
                # when coverage is a scalar.
                for coverage in [0.25, 0.50, 0.75]:
                    lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                    self.assertEqual(lo.shape, ())
                    self.assertEqual(hi.shape, ())
                    self.assertAlmostEqual(
                        beta.cdf(hi) - beta.cdf(lo),
                        coverage,
                    )
                    self.assertAlmostEqual(beta.cdf(lo), (1. - coverage) / 2.)
                    self.assertAlmostEqual(beta.cdf(hi), (1. + coverage) / 2.)
                # when coverage is an array.
                k = 5
                coverage = np.random.rand(k)
                lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                self.assertEqual(lo.shape, (k,))
                self.assertEqual(hi.shape, (k,))
                self.assertTrue(np.allclose(
                    beta.cdf(hi) - beta.cdf(lo),
                    coverage,
                ))
                self.assertTrue(np.all(
                    np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                    < 1e-10
                ))
                self.assertTrue(np.all(
                    np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                    < 1e-10
                ))
        # Test when a and b are 1D arrays.
        n = 10
        a = np.arange(1, n + 1)
        b = np.arange(n + 1, 1, -1)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n,))
            self.assertEqual(hi.shape, (n,))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                < 1e-10
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                < 1e-10
            ))
        #   when coverage is an array.
        coverage = np.random.rand(n)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n,))
        self.assertEqual(hi.shape, (n,))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10
        ))
        # Test when a and b are 2D arrays.
        n, m = 5, 2
        a = np.arange(1, n * m + 1).reshape(n, m)
        b = np.arange(n * m + 1, 1, -1).reshape(n, m)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n, m))
            self.assertEqual(hi.shape, (n, m))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                < 1e-10
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                < 1e-10
            ))
        #   when coverage is an array.
        coverage = np.random.rand(n, m)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10
        ))
        # Test when a and b broadcast over each other.
        n, m = 5, 2
        a = np.arange(1, n + 1).reshape(n, 1)
        b = np.arange(m + 1, 1, -1).reshape(1, m)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n, m))
            self.assertEqual(hi.shape, (n, m))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                < 1e-10
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                < 1e-10
            ))
        #   when coverage is an array.
        coverage = np.random.rand(n, m)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10
        ))
        # Test when coverage broadcasts over a and b.
        #   when a and b have the same shape.
        n = 10
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        beta = stats.beta(a, b)
        k = 5
        coverage = np.random.rand(k)[None, :]
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, k))
        self.assertEqual(hi.shape, (n, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, 1))
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        beta = stats.beta(a, b)
        k = 5
        coverage = np.random.rand(k)[None, None, :]
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m, k))
        self.assertEqual(hi.shape, (n, m, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, m, 1))
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10
        ))


class BetaHpdIntervalTestCase(unittest.TestCase):
    """Test ersa.utils.beta_hpd_interval."""

    @pytest.mark.level(1)
    def test_beta_hpd_interval(self):
        # Test when a and b are scalars.
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                if a == 1. and b == 1.:
                    # No HPD interval exists when a <= 1 and b <= 1.
                    continue
                beta = stats.beta(a, b)
                # when coverage is a scalar.
                for coverage in [0.25, 0.50, 0.75]:
                    lo, hi = utils.beta_hpd_interval(a, b, coverage)
                    self.assertEqual(lo.shape, ())
                    self.assertEqual(hi.shape, ())
                    self.assertAlmostEqual(
                        beta.cdf(hi) - beta.cdf(lo),
                        coverage,
                    )
                    equal_tailed_lo, equal_tailed_hi =\
                        utils.beta_equal_tailed_interval(a, b, coverage)
                    self.assertLess(
                        (hi - lo) - (equal_tailed_hi - equal_tailed_lo),
                        1e-12,
                    )
                # when coverage is an array.
                k = 5
                coverage = np.random.rand(k)
                lo, hi = utils.beta_hpd_interval(a, b, coverage)
                self.assertEqual(lo.shape, (k,))
                self.assertEqual(hi.shape, (k,))
                self.assertTrue(np.allclose(
                    beta.cdf(hi) - beta.cdf(lo),
                    coverage,
                ))
                equal_tailed_lo, equal_tailed_hi =\
                    utils.beta_equal_tailed_interval(a, b, coverage)
                self.assertTrue(np.all(
                    (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
                ))
        # Test when a and b are 1D arrays.
        n = 10
        a = np.arange(1, n + 1)
        b = np.arange(n + 1, 1, -1)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_hpd_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n,))
            self.assertEqual(hi.shape, (n,))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10
            ))
            equal_tailed_lo, equal_tailed_hi =\
                utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertTrue(np.all(
                (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
            ))
            self.assertTrue(np.any(
                (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
            ))
        #   when coverage is an array.
        coverage = np.random.rand(n)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n,))
        self.assertEqual(hi.shape, (n,))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
        ))
        # Test when a and b are 2D arrays.
        n, m = 5, 2
        a = np.arange(1, n * m + 1).reshape(n, m)
        b = np.arange(n * m + 1, 1, -1).reshape(n, m)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_hpd_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n, m))
            self.assertEqual(hi.shape, (n, m))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10
            ))
            equal_tailed_lo, equal_tailed_hi =\
                utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertTrue(np.all(
                (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
            ))
            self.assertTrue(np.any(
                (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
            ))
        #   when coverage is an array.
        coverage = np.random.rand(n, m)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
        ))
        # Test when a and b broadcast over each other.
        n, m = 5, 2
        a = np.arange(1, n + 1).reshape(n, 1)
        b = np.arange(m + 1, 1, -1).reshape(1, m)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_hpd_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n, m))
            self.assertEqual(hi.shape, (n, m))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10
            ))
            equal_tailed_lo, equal_tailed_hi =\
                utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertTrue(np.all(
                (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
            ))
            self.assertTrue(np.any(
                (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
            ))
        #   when coverage is an array.
        coverage = np.random.rand(n, m)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
        ))
        # Test when coverage broadcasts over a and b.
        #   when a and b have the same shape.
        n = 10
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        beta = stats.beta(a, b)
        k = 5
        coverage = np.random.rand(k)[None, :]
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, k))
        self.assertEqual(hi.shape, (n, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, 1)),
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        beta = stats.beta(a, b)
        k = 5
        coverage = np.random.rand(k)[None, None, :]
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m, k))
        self.assertEqual(hi.shape, (n, m, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, m, 1)),
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5
        ))

    def test_converges_for_short_intervals(self):
        coverage = 1e-8
        for a in [2., 5., 10.]:
            for b in [2., 5., 10.]:
                beta = stats.beta(a, b)
                lo, hi = utils.beta_hpd_interval(a, b, coverage)
                self.assertEqual(lo.shape, ())
                self.assertEqual(hi.shape, ())
                self.assertLess(lo, hi)
                self.assertAlmostEqual(beta.cdf(hi) - beta.cdf(lo), coverage)


class BetaEqualTailedCoverageTestCase(unittest.TestCase):
    """Test ersa.utils.beta_equal_tailed_coverage."""

    def test_beta_equal_tailed_coverage(self):
        # Test when a and b are scalars.
        for a in [1., 2., 3.]:
            for b in [1., 2., 3.]:
                # when x is a scalar.
                for x in [0.25, 0.50, 0.75]:
                    coverage = utils.beta_equal_tailed_coverage(a, b, x)
                    lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                    self.assertEqual(coverage.shape, ())
                    self.assertTrue(np.all(
                        np.isclose(lo, x) | np.isclose(hi, x)
                    ))
                # when x is an array.
                k = 5
                x = np.random.rand(k)
                coverage = utils.beta_equal_tailed_coverage(a, b, x)
                lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                self.assertEqual(coverage.shape, (k,))
                self.assertTrue(np.all(
                    np.isclose(lo, x) | np.isclose(hi, x)
                ))
        # Test when a and b are 1D arrays.
        n = 3
        a = np.arange(1, n + 1)
        b = np.arange(n + 1, 1, -1)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_equal_tailed_coverage(a, b, x)
            lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n,))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x)
            ))
        #   when x is an array.
        x = np.random.rand(n)
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n,))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        # Test when a and b are 2D arrays.
        n, m = 3, 2
        a = np.arange(1, n * m + 1).reshape(n, m)
        b = np.arange(n * m + 1, 1, -1).reshape(n, m)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_equal_tailed_coverage(a, b, x)
            lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n, m))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x)
            ))
        #   when x is an array.
        x = np.random.rand(n, m)
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        # Test when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)
        b = np.arange(m + 1, 1, -1).reshape(1, m)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_equal_tailed_coverage(a, b, x)
            lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n, m))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x)
            ))
        #   when x is an array.
        x = np.random.rand(n, m)
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        # Test when x broadcasts over a and b.
        #   when a and b have the same shape.
        n = 3
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        k = 5
        x = np.random.rand(k)[None, :]
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        k = 5
        x = np.random.rand(k)[None, None, :]
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))


class BetaHpdCoverageTestCase(unittest.TestCase):
    """Test ersa.utils.beta_hpd_coverage."""

    @pytest.mark.level(1)
    def test_beta_hpd_coverage(self):
        # Test when a and b are scalars.
        for a in [1., 2., 3.]:
            for b in [1., 2., 3.]:
                if a == 1. and b == 1.:
                    # No HPD interval exists when a <= 1 and b <= 1.
                    continue
                # when x is a scalar.
                for x in [0.25, 0.50, 0.75]:
                    coverage = utils.beta_hpd_coverage(a, b, x)
                    lo, hi = utils.beta_hpd_interval(a, b, coverage)
                    self.assertEqual(coverage.shape, ())
                    self.assertTrue(np.all(
                        np.isclose(lo, x) | np.isclose(hi, x)
                    ))
                # when x is an array.
                k = 5
                x = np.random.rand(k)
                coverage = utils.beta_hpd_coverage(a, b, x)
                lo, hi = utils.beta_hpd_interval(a, b, coverage)
                self.assertEqual(coverage.shape, (k,))
                self.assertTrue(np.all(
                    np.isclose(lo, x) | np.isclose(hi, x)
                ))
        # Test when a and b are 1D arrays.
        n = 3
        a = np.arange(1, n + 1)
        b = np.arange(n + 1, 1, -1)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_hpd_coverage(a, b, x)
            lo, hi = utils.beta_hpd_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n,))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x)
            ))
        #   when x is an array.
        x = np.random.rand(n)
        coverage = utils.beta_hpd_coverage(a, b, x)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n,))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        # Test when a and b are 2D arrays.
        n, m = 3, 2
        a = np.arange(1, n * m + 1).reshape(n, m)
        b = np.arange(n * m + 1, 1, -1).reshape(n, m)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_hpd_coverage(a, b, x)
            lo, hi = utils.beta_hpd_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n, m))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x)
            ))
        #   when x is an array.
        x = np.random.rand(n, m)
        coverage = utils.beta_hpd_coverage(a, b, x)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        # Test when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)
        b = np.arange(m + 1, 1, -1).reshape(1, m)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_hpd_coverage(a, b, x)
            lo, hi = utils.beta_hpd_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n, m))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x)
            ))
        #   when x is an array.
        x = np.random.rand(n, m)
        coverage = utils.beta_hpd_coverage(a, b, x)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        # Test when x broadcasts over a and b.
        #   when a and b have the same shape.
        n = 3
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        k = 5
        x = np.random.rand(k)[None, :]
        coverage = utils.beta_hpd_coverage(a, b, x)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        k = 5
        x = np.random.rand(k)[None, None, :]
        coverage = utils.beta_hpd_coverage(a, b, x)
        lo, hi = utils.beta_hpd_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x)
        ))
