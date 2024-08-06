"""Tests for opda.utils."""

import unittest

import numpy as np
import pytest
from scipy import stats

from opda import utils

from tests import testcases


class SortByFirstTestCase(unittest.TestCase):
    """Test opda.utils.sort_by_first."""

    def test_sort_by_first(self):
        def tolist(a):
            return a.tolist()

        # Test on no arguments.
        self.assertEqual(utils.sort_by_first(), ())
        # Test on empty lists.
        self.assertEqual(
            tuple(map(tolist, utils.sort_by_first([]))),
            ([],),
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
            tuple(map(tolist, utils.sort_by_first([1.], [2.], ["a"]))),
            ([1.], [2.], ["a"]),
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
                ["a", "c", "b"],
            ))),
            (
                [1., 2., 3.],
                [3., 2., 1.],
                ["a", "c", "b"],
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
                ["a", "c", "b"],
            ))),
            (
                [1., 2., 3.],
                [2., 3., 1.],
                ["c", "a", "b"],
            ),
        )


class DkwEpsilonTestCase(unittest.TestCase):
    """Test opda.utils.dkw_epsilon."""

    def test_dkw_epsilon(self):
        # Test when confidence = 0.
        self.assertEqual(utils.dkw_epsilon(1, 0.), np.sqrt(np.log(2) / 2))
        self.assertEqual(utils.dkw_epsilon(2, 0.), np.sqrt(np.log(2) / 4))
        self.assertEqual(utils.dkw_epsilon(4, 0.), np.sqrt(np.log(2) / 8))
        self.assertEqual(utils.dkw_epsilon(8, 0.), np.sqrt(np.log(2) / 16))

        # Test when confidence = 1.
        self.assertEqual(utils.dkw_epsilon(1, 1.), np.inf)
        self.assertEqual(utils.dkw_epsilon(2, 1.), np.inf)
        self.assertEqual(utils.dkw_epsilon(4, 1.), np.inf)
        self.assertEqual(utils.dkw_epsilon(8, 1.), np.inf)

        # Test when 0 < confidence < 1.
        self.assertAlmostEqual(utils.dkw_epsilon(2, 1. - 2./np.e), 0.5)
        self.assertAlmostEqual(utils.dkw_epsilon(8, 1. - 2./np.e), 0.25)
        self.assertAlmostEqual(utils.dkw_epsilon(1, 1. - 2./np.e**2), 1.)
        self.assertAlmostEqual(utils.dkw_epsilon(4, 1. - 2./np.e**2), 0.5)

        # Test error conditions.
        #   when n is not a scalar
        with self.assertRaises(ValueError):
            utils.dkw_epsilon([1], 0.5)
        with self.assertRaises(ValueError):
            utils.dkw_epsilon([[1]], 0.5)
        #   when n is not positive
        with self.assertRaises(ValueError):
            utils.dkw_epsilon(0, 0.5)
        with self.assertRaises(ValueError):
            utils.dkw_epsilon(-1, 0.5)
        #   when confidence is not a scalar
        with self.assertRaises(ValueError):
            utils.dkw_epsilon(10, [0.5])
        with self.assertRaises(ValueError):
            utils.dkw_epsilon(10, [[0.5]])
        #   when confidence is not between 0 and 1, inclusive
        with self.assertRaises(ValueError):
            utils.dkw_epsilon(10, -0.1)
        with self.assertRaises(ValueError):
            utils.dkw_epsilon(10, 1.1)


class BetaEqualTailedIntervalTestCase(testcases.RandomTestCase):
    """Test opda.utils.beta_equal_tailed_interval."""

    # backwards compatibility (scipy == 1.14.0)
    @pytest.mark.skipif(
        np.isnan(stats.beta(5., 5.).ppf(0.5)),
        reason="stats.beta(5., 5.).ppf(0.5) incorrectly returns NaN in"
               " scipy 1.14.0 on certain platforms. For more details,"
               " see https://github.com/scipy/scipy/issues/21303",
    )
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
                    self.assertLess(lo, beta.ppf(0.5))
                    self.assertGreater(hi, beta.ppf(0.5))
                    self.assertAlmostEqual(beta.cdf(lo), (1. - coverage) / 2.)
                    self.assertAlmostEqual(beta.cdf(hi), (1. + coverage) / 2.)
                # when coverage is an array.
                k = 5
                coverage = self.generator.random(size=k)
                lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                self.assertEqual(lo.shape, (k,))
                self.assertEqual(hi.shape, (k,))
                self.assertTrue(np.allclose(
                    beta.cdf(hi) - beta.cdf(lo),
                    coverage,
                ))
                self.assertTrue(np.all(
                    (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
                ))
                self.assertTrue(np.all(
                    np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                    < 1e-10,
                ))
                self.assertTrue(np.all(
                    np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                    < 1e-10,
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
                < 1e-10,
            ))
            self.assertTrue(np.all(
                (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                < 1e-10,
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                < 1e-10,
            ))
        #   when coverage is an array.
        coverage = self.generator.random(size=n)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n,))
        self.assertEqual(hi.shape, (n,))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10,
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
                < 1e-10,
            ))
            self.assertTrue(np.all(
                (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                < 1e-10,
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                < 1e-10,
            ))
        #   when coverage is an array.
        coverage = self.generator.random(size=(n, m))
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10,
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
                < 1e-10,
            ))
            self.assertTrue(np.all(
                (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
                < 1e-10,
            ))
            self.assertTrue(np.all(
                np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
                < 1e-10,
            ))
        #   when coverage is an array.
        coverage = self.generator.random(size=(n, m))
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10,
        ))
        # Test when coverage broadcasts over a and b.
        #   when a and b have the same shape.
        n = 10
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        beta = stats.beta(a, b)
        k = 5
        coverage = self.generator.random(size=k)[None, :]
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, k))
        self.assertEqual(hi.shape, (n, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, 1)),
        ))
        self.assertTrue(np.all(
            (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10,
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        beta = stats.beta(a, b)
        k = 5
        coverage = self.generator.random(size=k)[None, None, :]
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m, k))
        self.assertEqual(hi.shape, (n, m, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, m, 1)),
        ))
        self.assertTrue(np.all(
            (lo < beta.ppf(0.5)) & (hi > beta.ppf(0.5)),
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(lo) - (1. - coverage) / 2.)
            < 1e-10,
        ))
        self.assertTrue(np.all(
            np.abs(beta.cdf(hi) - (1. + coverage) / 2.)
            < 1e-10,
        ))

    # backwards compatibility (scipy == 1.14.0)
    @pytest.mark.skipif(
        np.isnan(stats.beta(5., 5.).ppf(0.5)),
        reason="stats.beta(5., 5.).ppf(0.5) incorrectly returns NaN in"
               " scipy 1.14.0 on certain platforms. For more details,"
               " see https://github.com/scipy/scipy/issues/21303",
    )
    @pytest.mark.level(1)
    def test_on_small_confidences(self):
        for coverage in [1e-8, 1e-12, 1e-16]:
            for a in [1., 5., 10.]:
                for b in [1., 5., 10.]:
                    beta = stats.beta(a, b)
                    lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                    self.assertEqual(lo.shape, ())
                    self.assertEqual(hi.shape, ())
                    self.assertLessEqual(lo, hi)
                    self.assertAlmostEqual(
                        beta.cdf(hi) - beta.cdf(lo),
                        coverage,
                    )
                    self.assertLessEqual(lo, beta.ppf(0.5))
                    self.assertGreaterEqual(hi, beta.ppf(0.5))

    # backwards compatibility (scipy == 1.14.0)
    @pytest.mark.skipif(
        np.isnan(stats.beta(5., 5.).ppf(0.5)),
        reason="stats.beta(5., 5.).ppf(0.5) incorrectly returns NaN in"
               " scipy 1.14.0 on certain platforms. For more details,"
               " see https://github.com/scipy/scipy/issues/21303",
    )
    def test_on_zero_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                beta = stats.beta(a, b)
                lo, hi = utils.beta_equal_tailed_interval(a, b, 0.)
                self.assertEqual(lo.shape, ())
                self.assertEqual(hi.shape, ())
                self.assertLessEqual(lo, hi)
                self.assertAlmostEqual(lo, hi)
                self.assertAlmostEqual(beta.cdf(lo), 0.5)
                self.assertAlmostEqual(beta.cdf(hi), 0.5)
                self.assertAlmostEqual(beta.cdf(hi) - beta.cdf(lo), 0.)


class BetaHighestDensityIntervalTestCase(testcases.RandomTestCase):
    """Test opda.utils.beta_highest_density_interval."""

    # backwards compatibility (scipy == 1.14.0)
    @pytest.mark.skipif(
        np.isnan(stats.beta(5., 5.).ppf(0.5)),
        reason="stats.beta(5., 5.).ppf(0.5) incorrectly returns NaN in"
               " scipy 1.14.0 on certain platforms. For more details,"
               " see https://github.com/scipy/scipy/issues/21303",
    )
    def test_beta_highest_density_interval(self):
        # Test when a and b are scalars.
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                if a == 1. and b == 1.:
                    # No highest density interval exists when a <= 1 and b <= 1.
                    continue
                mode = (a - 1) / (a + b - 2)
                beta = stats.beta(a, b)
                # when coverage is a scalar.
                for coverage in [0.25, 0.50, 0.75]:
                    lo, hi = utils.beta_highest_density_interval(a, b, coverage)
                    self.assertEqual(lo.shape, ())
                    self.assertEqual(hi.shape, ())
                    self.assertAlmostEqual(
                        beta.cdf(hi) - beta.cdf(lo),
                        coverage,
                    )
                    self.assertLessEqual(lo, mode)
                    self.assertGreaterEqual(hi, mode)
                    equal_tailed_lo, equal_tailed_hi =\
                        utils.beta_equal_tailed_interval(a, b, coverage)
                    self.assertLess(
                        (hi - lo) - (equal_tailed_hi - equal_tailed_lo),
                        1e-12,
                    )
                # when coverage is an array.
                k = 5
                coverage = self.generator.random(size=k)
                lo, hi = utils.beta_highest_density_interval(a, b, coverage)
                self.assertEqual(lo.shape, (k,))
                self.assertEqual(hi.shape, (k,))
                self.assertTrue(np.allclose(
                    beta.cdf(hi) - beta.cdf(lo),
                    coverage,
                ))
                self.assertTrue(np.all(
                    (lo <= mode) & (hi >= mode),
                ))
                equal_tailed_lo, equal_tailed_hi =\
                    utils.beta_equal_tailed_interval(a, b, coverage)
                self.assertTrue(np.all(
                    (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
                ))
        # Test when a and b are 1D arrays.
        n = 10
        a = np.arange(1, n + 1)
        b = np.arange(n + 1, 1, -1)
        mode = (a - 1) / (a + b - 2)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_highest_density_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n,))
            self.assertEqual(hi.shape, (n,))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10,
            ))
            self.assertTrue(np.all(
                (lo <= mode) & (hi >= mode),
            ))
            equal_tailed_lo, equal_tailed_hi =\
                utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertTrue(np.all(
                (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
            ))
            self.assertTrue(np.any(
                (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
            ))
        #   when coverage is an array.
        coverage = self.generator.random(size=n)
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n,))
        self.assertEqual(hi.shape, (n,))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            (lo <= mode) & (hi >= mode),
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
        ))
        # Test when a and b are 2D arrays.
        n, m = 5, 2
        a = np.arange(1, n * m + 1).reshape(n, m)
        b = np.arange(n * m + 1, 1, -1).reshape(n, m)
        mode = (a - 1) / (a + b - 2)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_highest_density_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n, m))
            self.assertEqual(hi.shape, (n, m))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10,
            ))
            self.assertTrue(np.all(
                (lo <= mode) & (hi >= mode),
            ))
            equal_tailed_lo, equal_tailed_hi =\
                utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertTrue(np.all(
                (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
            ))
            self.assertTrue(np.any(
                (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
            ))
        #   when coverage is an array.
        coverage = self.generator.random(size=(n, m))
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            (lo <= mode) & (hi >= mode),
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
        ))
        # Test when a and b broadcast over each other.
        n, m = 5, 2
        a = np.arange(1, n + 1).reshape(n, 1)
        b = np.arange(m + 1, 1, -1).reshape(1, m)
        mode = (a - 1) / (a + b - 2)
        beta = stats.beta(a, b)
        #   when coverage is a scalar.
        for coverage in [0.25, 0.50, 0.75]:
            lo, hi = utils.beta_highest_density_interval(a, b, coverage)
            self.assertEqual(lo.shape, (n, m))
            self.assertEqual(hi.shape, (n, m))
            self.assertTrue(np.all(
                np.abs((beta.cdf(hi) - beta.cdf(lo)) - coverage)
                < 1e-10,
            ))
            self.assertTrue(np.all(
                (lo <= mode) & (hi >= mode),
            ))
            equal_tailed_lo, equal_tailed_hi =\
                utils.beta_equal_tailed_interval(a, b, coverage)
            self.assertTrue(np.all(
                (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
            ))
            self.assertTrue(np.any(
                (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
            ))
        #   when coverage is an array.
        coverage = self.generator.random(size=(n, m))
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m))
        self.assertEqual(hi.shape, (n, m))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            coverage,
        ))
        self.assertTrue(np.all(
            (lo <= mode) & (hi >= mode),
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
        ))
        # Test when coverage broadcasts over a and b.
        #   when a and b have the same shape.
        n = 10
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        mode = (a - 1) / (a + b - 2)
        beta = stats.beta(a, b)
        k = 5
        coverage = self.generator.random(size=k)[None, :]
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, k))
        self.assertEqual(hi.shape, (n, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, 1)),
        ))
        self.assertTrue(np.all(
            (lo <= mode) & (hi >= mode),
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        mode = (a - 1) / (a + b - 2)
        beta = stats.beta(a, b)
        k = 5
        coverage = self.generator.random(size=k)[None, None, :]
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(lo.shape, (n, m, k))
        self.assertEqual(hi.shape, (n, m, k))
        self.assertTrue(np.allclose(
            beta.cdf(hi) - beta.cdf(lo),
            np.tile(coverage, (n, m, 1)),
        ))
        self.assertTrue(np.all(
            (lo <= mode) & (hi >= mode),
        ))
        equal_tailed_lo, equal_tailed_hi =\
            utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertTrue(np.all(
            (hi - lo) - (equal_tailed_hi - equal_tailed_lo) < 1e-12,
        ))
        self.assertTrue(np.any(
            (equal_tailed_hi - equal_tailed_lo) - (hi - lo) > 1e-5,
        ))

    @pytest.mark.level(1)
    def test_on_small_confidences(self):
        for coverage in [1e-8, 1e-12, 1e-16]:
            for a in [1., 5., 10.]:
                for b in [1., 5., 10.]:
                    if a == 1 and b == 1:
                        # No highest density interval exists when
                        # a <= 1 and b <= 1.
                        continue
                    mode = (a - 1) / (a + b - 2)
                    beta = stats.beta(a, b)
                    lo, hi = utils.beta_highest_density_interval(a, b, coverage)
                    self.assertEqual(lo.shape, ())
                    self.assertEqual(hi.shape, ())
                    self.assertLessEqual(lo, hi)
                    self.assertAlmostEqual(
                        beta.cdf(hi) - beta.cdf(lo),
                        coverage,
                    )
                    self.assertLessEqual(lo, mode)
                    self.assertGreaterEqual(hi, mode)

    # backwards compatibility (scipy == 1.14.0)
    @pytest.mark.skipif(
        np.isnan(stats.beta(5., 5.).ppf(0.5)),
        reason="stats.beta(5., 5.).ppf(0.5) incorrectly returns NaN in"
               " scipy 1.14.0 on certain platforms. For more details,"
               " see https://github.com/scipy/scipy/issues/21303",
    )
    def test_on_zero_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                if a == 1 and b == 1:
                    # No highest density interval exists when a <= 1 and b <= 1.
                    continue
                mode = (a - 1) / (a + b - 2)
                beta = stats.beta(a, b)
                lo, hi = utils.beta_highest_density_interval(a, b, 0.)
                self.assertEqual(lo.shape, ())
                self.assertEqual(hi.shape, ())
                self.assertLessEqual(lo, hi)
                self.assertAlmostEqual(lo, hi)
                self.assertAlmostEqual(lo, mode)
                self.assertAlmostEqual(hi, mode)
                self.assertAlmostEqual(beta.cdf(hi) - beta.cdf(lo), 0.)


class BetaEqualTailedCoverageTestCase(testcases.RandomTestCase):
    """Test opda.utils.beta_equal_tailed_coverage."""

    def test_beta_equal_tailed_coverage(self):
        x_lo, x_hi = 0.01, 0.99
        # NOTE: These checks must use values of x between 0.01 and 0.99,
        # otherwise they will spuriously fail. The reason is that these
        # tests work by taking the coverage returned by
        # beta_equal_tailed_coverage, constructing the equal-tailed
        # interval with that coverage, and verifying that x is one of
        # the endpoints. If x is too close to 0 or 1 and the beta
        # distribution is highly concentrated, then x can change by a
        # large amount *without* changing the coverage much. This
        # happens because the distribution will have almost no
        # probability density near the endpoints. This invalidates our
        # testing strategy. Instead, we test when x is near the
        # endpoints in a separate test (See
        # test_when_interval_has_large_coverage on
        # BetaEqualTailedCoverageTestCase).

        # Test when a and b are scalars.
        for a in [1., 2., 3.]:
            for b in [1., 2., 3.]:
                # when x is a scalar.
                for x in [0.25, 0.50, 0.75]:
                    coverage = utils.beta_equal_tailed_coverage(a, b, x)
                    lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                    self.assertEqual(coverage.shape, ())
                    self.assertTrue(np.all(
                        np.isclose(lo, x) | np.isclose(hi, x),
                    ))
                # when x is an array.
                k = 5
                x = self.generator.uniform(x_lo, x_hi, size=k)
                coverage = utils.beta_equal_tailed_coverage(a, b, x)
                lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
                self.assertEqual(coverage.shape, (k,))
                self.assertTrue(np.all(
                    np.isclose(lo, x) | np.isclose(hi, x),
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
                np.isclose(lo, x) | np.isclose(hi, x),
            ))
        #   when x is an array.
        x = self.generator.uniform(x_lo, x_hi, size=n)
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n,))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
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
                np.isclose(lo, x) | np.isclose(hi, x),
            ))
        #   when x is an array.
        x = self.generator.uniform(x_lo, x_hi, size=(n, m))
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
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
                np.isclose(lo, x) | np.isclose(hi, x),
            ))
        #   when x is an array.
        x = self.generator.uniform(x_lo, x_hi, size=(n, m))
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))
        # Test when x broadcasts over a and b.
        #   when a and b have the same shape.
        n = 3
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        k = 5
        x = self.generator.uniform(x_lo, x_hi, size=(1, k))
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        k = 5
        x = self.generator.uniform(x_lo, x_hi, size=(1, 1, k))
        coverage = utils.beta_equal_tailed_coverage(a, b, x)
        lo, hi = utils.beta_equal_tailed_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))

    @pytest.mark.level(1)
    def test_when_interval_has_large_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                for x_less_than_median in [False, True]:
                    for eps in [1e-8, 1e-12, 1e-16]:
                        x = (
                            eps
                            if x_less_than_median else
                            1. - eps
                        )
                        coverage = utils.beta_equal_tailed_coverage(a, b, x)
                        lo, hi =\
                            utils.beta_equal_tailed_interval(a, b, coverage)
                        self.assertEqual(coverage.shape, ())
                        self.assertAlmostEqual(
                            x,
                            lo if x_less_than_median else hi,
                        )

    def test_when_x_is_on_the_boundary(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                for x_less_than_median in [False, True]:
                    x = 0. if x_less_than_median else 1.
                    coverage = utils.beta_equal_tailed_coverage(a, b, x)
                    self.assertEqual(coverage.shape, ())
                    self.assertAlmostEqual(coverage, 1.)

    # backwards compatibility (scipy == 1.14.0)
    @pytest.mark.skipif(
        np.isnan(stats.beta(5., 5.).ppf(0.5)),
        reason="stats.beta(5., 5.).ppf(0.5) incorrectly returns NaN in"
               " scipy 1.14.0 on certain platforms. For more details,"
               " see https://github.com/scipy/scipy/issues/21303",
    )
    @pytest.mark.level(1)
    def test_when_interval_has_small_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                median = stats.beta(a, b).ppf(0.5)
                for x_less_than_median in [False, True]:
                    for eps in [1e-8, 1e-12, 1e-16]:
                        x = np.clip(
                            median - eps
                            if x_less_than_median else
                            median + eps,
                            0.,
                            1.,
                        )
                        coverage = utils.beta_equal_tailed_coverage(a, b, x)
                        lo, hi =\
                            utils.beta_equal_tailed_interval(a, b, coverage)
                        self.assertEqual(coverage.shape, ())
                        self.assertAlmostEqual(
                            x,
                            lo if x_less_than_median else hi,
                        )

    # backwards compatibility (scipy == 1.14.0)
    @pytest.mark.skipif(
        np.isnan(stats.beta(5., 5.).ppf(0.5)),
        reason="stats.beta(5., 5.).ppf(0.5) incorrectly returns NaN in"
               " scipy 1.14.0 on certain platforms. For more details,"
               " see https://github.com/scipy/scipy/issues/21303",
    )
    def test_when_interval_has_zero_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                # Set x equal to the median.
                x = stats.beta(a, b).ppf(0.5)
                coverage = utils.beta_equal_tailed_coverage(a, b, x)
                self.assertEqual(coverage.shape, ())
                self.assertAlmostEqual(coverage, 0.)


class BetaHighestDensityCoverageTestCase(testcases.RandomTestCase):
    """Test opda.utils.beta_highest_density_coverage."""

    def test_beta_highest_density_coverage(self):
        x_lo, x_hi = 0.01, 0.99
        # NOTE: These checks must use values of x between 0.01 and 0.99,
        # otherwise they will spuriously fail. The reason is that these
        # tests work by taking the coverage returned by
        # beta_highest_density_coverage, constructing the highest
        # density interval with that coverage, and verifying that x is
        # one of the endpoints. If x is too close to 0 or 1 and the beta
        # distribution is highly concentrated, then x can change by a
        # large amount *without* changing the coverage much. This happens
        # because the distribution will have almost no probability
        # density near the endpoints. This invalidates our testing
        # strategy. Instead, we test when x is near the endpoints in a
        # separate test (See test_when_interval_has_large_coverage on
        # BetaHighestDensityCoverageTestCase).

        # Test when a and b are scalars.
        for a in [1., 2., 3.]:
            for b in [1., 2., 3.]:
                if a == 1. and b == 1.:
                    # No highest density interval exists when a <= 1 and b <= 1.
                    continue
                # when x is a scalar.
                for x in [0.25, 0.50, 0.75]:
                    coverage = utils.beta_highest_density_coverage(a, b, x)
                    lo, hi = utils.beta_highest_density_interval(a, b, coverage)
                    self.assertEqual(coverage.shape, ())
                    self.assertTrue(np.all(
                        np.isclose(lo, x) | np.isclose(hi, x),
                    ))
                # when x is an array.
                k = 5
                x = self.generator.uniform(x_lo, x_hi, size=k)
                coverage = utils.beta_highest_density_coverage(a, b, x)
                lo, hi = utils.beta_highest_density_interval(a, b, coverage)
                self.assertEqual(coverage.shape, (k,))
                self.assertTrue(np.all(
                    np.isclose(lo, x) | np.isclose(hi, x),
                ))
        # Test when a and b are 1D arrays.
        n = 3
        a = np.arange(1, n + 1)
        b = np.arange(n + 1, 1, -1)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_highest_density_coverage(a, b, x)
            lo, hi = utils.beta_highest_density_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n,))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x),
            ))
        #   when x is an array.
        x = self.generator.uniform(x_lo, x_hi, size=n)
        coverage = utils.beta_highest_density_coverage(a, b, x)
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n,))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))
        # Test when a and b are 2D arrays.
        n, m = 3, 2
        a = np.arange(1, n * m + 1).reshape(n, m)
        b = np.arange(n * m + 1, 1, -1).reshape(n, m)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_highest_density_coverage(a, b, x)
            lo, hi = utils.beta_highest_density_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n, m))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x),
            ))
        #   when x is an array.
        x = self.generator.uniform(x_lo, x_hi, size=(n, m))
        coverage = utils.beta_highest_density_coverage(a, b, x)
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))
        # Test when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)
        b = np.arange(m + 1, 1, -1).reshape(1, m)
        #   when x is a scalar.
        for x in [0.25, 0.50, 0.75]:
            coverage = utils.beta_highest_density_coverage(a, b, x)
            lo, hi = utils.beta_highest_density_interval(a, b, coverage)
            self.assertEqual(coverage.shape, (n, m))
            self.assertTrue(np.all(
                np.isclose(lo, x) | np.isclose(hi, x),
            ))
        #   when x is an array.
        x = self.generator.uniform(x_lo, x_hi, size=(n, m))
        coverage = utils.beta_highest_density_coverage(a, b, x)
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))
        # Test when x broadcasts over a and b.
        #   when a and b have the same shape.
        n = 3
        a = np.arange(1, n + 1)[:, None]
        b = np.arange(n + 1, 1, -1)[:, None]
        k = 5
        x = self.generator.uniform(x_lo, x_hi, size=(1, k))
        coverage = utils.beta_highest_density_coverage(a, b, x)
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))
        #   when a and b broadcast over each other.
        n, m = 3, 2
        a = np.arange(1, n + 1).reshape(n, 1)[..., None]
        b = np.arange(m + 1, 1, -1).reshape(1, m)[..., None]
        k = 5
        x = self.generator.uniform(x_lo, x_hi, size=(1, 1, k))
        coverage = utils.beta_highest_density_coverage(a, b, x)
        lo, hi = utils.beta_highest_density_interval(a, b, coverage)
        self.assertEqual(coverage.shape, (n, m, k))
        self.assertTrue(np.all(
            np.isclose(lo, x) | np.isclose(hi, x),
        ))

    @pytest.mark.level(1)
    def test_when_interval_has_large_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                if a == 1 and b == 1:
                    # No highest density interval exists when a <= 1 and b <= 1.
                    continue
                for x_less_than_mode in [False, True]:
                    for eps in [1e-8, 1e-12, 1e-16]:
                        x = (
                            eps
                            if x_less_than_mode else
                            1. - eps
                        )
                        coverage = utils.beta_highest_density_coverage(a, b, x)
                        lo, hi =\
                            utils.beta_highest_density_interval(a, b, coverage)
                        self.assertEqual(coverage.shape, ())
                        self.assertAlmostEqual(
                            x,
                            lo if x_less_than_mode else hi,
                        )

    def test_when_x_is_on_the_boundary(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                if a == 1 and b == 1:
                    # No highest density interval exists when a <= 1 and b <= 1.
                    continue
                mode = (a - 1) / (a + b - 2)
                for x_less_than_mode in [False, True]:
                    x = 0. if x_less_than_mode else 1.
                    coverage = utils.beta_highest_density_coverage(a, b, x)
                    self.assertEqual(coverage.shape, ())
                    self.assertAlmostEqual(
                        coverage,
                        1. if x != mode else 0.,
                    )

    @pytest.mark.level(1)
    def test_when_interval_has_small_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                if a == 1 and b == 1:
                    # No highest density interval exists when a <= 1 and b <= 1.
                    continue
                mode = (a - 1) / (a + b - 2)
                for x_less_than_mode in [False, True]:
                    for eps in [1e-8, 1e-12, 1e-16]:
                        x = np.clip(
                            mode - eps
                            if x_less_than_mode else
                            mode + eps,
                            0.,
                            1.,
                        )
                        coverage = utils.beta_highest_density_coverage(a, b, x)
                        lo, hi =\
                            utils.beta_highest_density_interval(a, b, coverage)
                        self.assertEqual(coverage.shape, ())
                        self.assertAlmostEqual(
                            x,
                            lo if x_less_than_mode else hi,
                        )

    def test_when_interval_has_zero_coverage(self):
        for a in [1., 5., 10.]:
            for b in [1., 5., 10.]:
                if a == 1 and b == 1:
                    # No highest density interval exists when a <= 1 and b <= 1.
                    continue
                # Set x equal to the mode.
                x = (a - 1) / (a + b - 2)
                coverage = utils.beta_highest_density_coverage(a, b, x)
                self.assertEqual(coverage.shape, ())
                self.assertAlmostEqual(coverage, 0.)


class BinomialConfidenceIntervalTestCase(unittest.TestCase):
    """Test opda.utils.binomial_confidence_interval."""

    def test_binomial_confidence_interval(self):
        for n_successes, n_total, confidence, (lo_expected, hi_expected) in [
                # n_total == 1
                ( 0,  1, 0.0, (0.0000, 0.5000)),
                ( 0,  1, 0.5, (0.0000, 0.7500)),
                ( 0,  1, 1.0, (0.0000, 1.0000)),
                ( 1,  1, 0.0, (0.5000, 1.0000)),
                ( 1,  1, 0.5, (0.2500, 1.0000)),
                ( 1,  1, 1.0, (0.0000, 1.0000)),
                # n_successes == 0
                ( 0, 10, 0.0, (0.0000, 0.0670)),
                ( 0, 10, 0.5, (0.0000, 0.1295)),
                ( 0, 10, 1.0, (0.0000, 1.0000)),
                # 0 < n_successes < n_total
                ( 5, 10, 0.0, (0.4517, 0.5483)),
                ( 5, 10, 0.5, (0.3507, 0.6493)),
                ( 5, 10, 1.0, (0.0000, 1.0000)),
                # n_successes == n_total
                (10, 10, 0.0, (0.9330, 1.0000)),
                (10, 10, 0.5, (0.8705, 1.0000)),
                (10, 10, 1.0, (0.0000, 1.0000)),
        ]:
            lo_actual, hi_actual = utils.binomial_confidence_interval(
                n_successes, n_total, confidence,
            )
            self.assertAlmostEqual(lo_actual, lo_expected, places=3)
            self.assertAlmostEqual(hi_actual, hi_expected, places=3)

    def test_binomial_confidence_interval_broadcasts(self):
        # 1D array x scalar x scalar
        lo, hi = utils.binomial_confidence_interval(
            n_successes=[0, 5, 10],
            n_total=10,
            confidence=0.5,
        )
        self.assertTrue(np.allclose(
            lo,
            [0.0000, 0.3507, 0.8705],
            atol=5e-4,
        ))
        self.assertTrue(np.allclose(
            hi,
            [0.1295, 0.6493, 1.0000],
            atol=5e-4,
        ))
        # scalar x 1D array x scalar
        lo, hi = utils.binomial_confidence_interval(
            n_successes=0,
            n_total=[1, 10],
            confidence=0.5,
        )
        self.assertTrue(np.allclose(
            lo,
            [0.0000, 0.0000],
            atol=5e-4,
        ))
        self.assertTrue(np.allclose(
            hi,
            [0.7500, 0.1295],
            atol=5e-4,
        ))
        # scalar x scalar x 1D array
        lo, hi = utils.binomial_confidence_interval(
            n_successes=5,
            n_total=10,
            confidence=[0.0, 0.5, 1.0],
        )
        self.assertTrue(np.allclose(
            lo,
            [0.4517, 0.3507, 0.0000],
            atol=5e-4,
        ))
        self.assertTrue(np.allclose(
            hi,
            [0.5483, 0.6493, 1.0000],
            atol=5e-4,
        ))

    @pytest.mark.level(1)
    def test_binomial_confidence_interval_is_symmetric(self):
        for n_successes, n_total in [
                (  0,   1),
                (  0, 100),
                ( 25, 100),
                ( 50, 100),
        ]:
            for confidence in [0.00, 0.25, 0.50, 0.75, 1.00]:
                lo1, hi1 = utils.binomial_confidence_interval(
                    n_successes, n_total, confidence,
                )
                lo2, hi2 = utils.binomial_confidence_interval(
                    n_total - n_successes, n_total, confidence,
                )
                self.assertAlmostEqual(lo1, 1 - hi2)
                self.assertAlmostEqual(hi1, 1 - lo2)


class NormalPdfTestCase(unittest.TestCase):
    """Test opda.utils.normal_pdf."""

    def test_normal_pdf(self):
        norm = stats.norm(0., 1.)
        # scalar
        for x in [
                  0.,
                -1e2, -1e1, -1e0, -1e-1, 1e-1, 1e0, 1e1, 1e2,
                -5e2, -5e1, -5e0, -5e-1, 5e-1, 5e0, 5e1, 5e2,
        ]:
            self.assertTrue(np.isscalar(utils.normal_pdf(x)))
            self.assertAlmostEqual(utils.normal_pdf(x), norm.pdf(x))
        # 1D array
        xs = np.linspace(-10., 10., num=1_000)
        self.assertEqual(utils.normal_pdf(xs).shape, xs.shape)
        self.assertTrue(np.allclose(utils.normal_pdf(xs), norm.pdf(xs)))
        # 2D array
        xs = np.linspace(-10., 10., num=1_000).reshape((100, 10))
        self.assertEqual(utils.normal_pdf(xs).shape, xs.shape)
        self.assertTrue(np.allclose(utils.normal_pdf(xs), norm.pdf(xs)))


class NormalCdfTestCase(unittest.TestCase):
    """Test opda.utils.normal_cdf."""

    def test_normal_cdf(self):
        norm = stats.norm(0., 1.)
        # scalar
        for x in [
                  0.,
                -1e2, -1e1, -1e0, -1e-1, 1e-1, 1e0, 1e1, 1e2,
                -5e2, -5e1, -5e0, -5e-1, 5e-1, 5e0, 5e1, 5e2,
        ]:
            self.assertTrue(np.isscalar(utils.normal_cdf(x)))
            self.assertAlmostEqual(utils.normal_cdf(x), norm.cdf(x))
        # 1D array
        xs = np.linspace(-10., 10., num=1_000)
        self.assertEqual(utils.normal_cdf(xs).shape, xs.shape)
        self.assertTrue(np.allclose(utils.normal_cdf(xs), norm.cdf(xs)))
        # 2D array
        xs = np.linspace(-10., 10., num=1_000).reshape((100, 10))
        self.assertEqual(utils.normal_cdf(xs).shape, xs.shape)
        self.assertTrue(np.allclose(utils.normal_cdf(xs), norm.cdf(xs)))


class NormalPpfTestCase(unittest.TestCase):
    """Test opda.utils.normal_ppf."""

    def test_normal_ppf(self):
        norm = stats.norm(0., 1.)
        # scalar
        for q in [
                0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
                0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01,
                0.9999999, 0.999999, 0.99999, 0.9999, 0.999, 0.99,
        ]:
            self.assertTrue(np.isscalar(utils.normal_ppf(q)))
            self.assertAlmostEqual(utils.normal_ppf(q), norm.ppf(q))
        # 1D array
        qs = np.linspace(0., 1., num=1_000)
        self.assertEqual(utils.normal_ppf(qs).shape, qs.shape)
        self.assertTrue(np.allclose(utils.normal_ppf(qs), norm.ppf(qs)))
        # 2D array
        qs = np.linspace(0., 1., num=1_000).reshape((100, 10))
        self.assertEqual(utils.normal_ppf(qs).shape, qs.shape)
        self.assertTrue(np.allclose(utils.normal_ppf(qs), norm.ppf(qs)))
