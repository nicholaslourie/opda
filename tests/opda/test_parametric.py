"""Tests for opda.parametric."""

import warnings

import numpy as np
import pytest
from scipy import stats

from opda import exceptions, parametric, utils
import opda.random

from tests import testcases


class QuadraticDistributionTestCase(testcases.RandomTestCase):
    """Test opda.parametric.QuadraticDistribution."""

    def test_class_attributes(self):
        # Test C_MIN and C_MAX.

        c_min = parametric.QuadraticDistribution.C_MIN
        c_max = parametric.QuadraticDistribution.C_MAX

        # QuadraticDistribution should always support c = 1 to 5.
        self.assertEqual(c_min, 1)
        self.assertGreaterEqual(c_max, 5)

        # QuadraticDistribution should support all claimed values of c.
        for c in range(c_min, c_max + 1):
            dist = parametric.QuadraticDistribution(0., 1., c, False)
            self.assertGreater(dist.pdf(0.5), 0.)
            self.assertGreater(dist.cdf(0.5), 0.)

        # C_MAX should match the maximum value for c mentioned in the docstring.
        self.assertIn(
            f"Values of ``c`` greater than {c_max} are not supported.",
            " ".join(parametric.QuadraticDistribution.__doc__.split()),
        )

    @pytest.mark.level(2)
    def test_attributes(self):
        n_samples = 1_000_000

        bounds = [(-10., -1.), (-1., 0.), (0., 0.), (0., 1.), (1., 10.)]
        cs = [1, 2, 10]
        for a, b in bounds:
            for c in cs:
                for convex in [False, True]:
                    dist = parametric.QuadraticDistribution(a, b, c, convex)
                    ys = dist.sample(n_samples)
                    self.assertAlmostEqual(
                        dist.mean,
                        np.mean(ys),
                        delta=6 * np.std(ys) / n_samples**0.5,
                    )
                    self.assertAlmostEqual(
                        dist.variance,
                        np.mean((ys - dist.mean)**2),
                        delta=6 * np.std((ys - dist.mean)**2) / n_samples**0.5,
                    )

    def test___eq__(self):
        bounds = [(-10., -1.), (-1., 0.), (0., 0.), (0., 1.), (1., 10.)]
        cs = [1, 2, 10]
        for a, b in bounds:
            for c in cs:
                for convex in [False, True]:
                    # Test inequality with other objects.
                    self.assertNotEqual(
                        parametric.QuadraticDistribution(a, b, c, convex),
                        None,
                    )
                    self.assertNotEqual(
                        parametric.QuadraticDistribution(a, b, c, convex),
                        1.,
                    )
                    self.assertNotEqual(
                        parametric.QuadraticDistribution(a, b, c, convex),
                        set(),
                    )

                    # Test (in)equality between instances of the same class.
                    #   equality
                    self.assertEqual(
                        parametric.QuadraticDistribution(a, b, c, convex),
                        parametric.QuadraticDistribution(a, b, c, convex),
                    )
                    #   inequality
                    for a_, _ in bounds:
                        if a_ == a or a_ > b:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            parametric.QuadraticDistribution(a_, b, c, convex),
                        )
                    for _, b_ in bounds:
                        if b_ == b or b_ < a:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            parametric.QuadraticDistribution(a, b_, c, convex),
                        )
                    for c_ in cs:
                        if c_ == c:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            parametric.QuadraticDistribution(a, b, c_, convex),
                        )
                    for convex_ in [False, True]:
                        if convex_ == convex:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            parametric.QuadraticDistribution(a, b, c, convex_),
                        )

                    # Test (in)equality between instances of different classes.
                    class QuadraticDistributionSubclass(
                            parametric.QuadraticDistribution,
                    ):
                        pass
                    #   equality
                    self.assertEqual(
                        parametric.QuadraticDistribution(a, b, c, convex),
                        QuadraticDistributionSubclass(a, b, c, convex),
                    )
                    self.assertEqual(
                        QuadraticDistributionSubclass(a, b, c, convex),
                        parametric.QuadraticDistribution(a, b, c, convex),
                    )
                    #   inequality
                    for a_, _ in bounds:
                        if a_ == a or a_ > b:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            QuadraticDistributionSubclass(a_, b, c, convex),
                        )
                        self.assertNotEqual(
                            QuadraticDistributionSubclass(a_, b, c, convex),
                            parametric.QuadraticDistribution(a, b, c, convex),
                        )
                    for _, b_ in bounds:
                        if b_ == b or b_ < a:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            QuadraticDistributionSubclass(a, b_, c, convex),
                        )
                        self.assertNotEqual(
                            QuadraticDistributionSubclass(a, b_, c, convex),
                            parametric.QuadraticDistribution(a, b, c, convex),
                        )
                    for c_ in cs:
                        if c_ == c:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            QuadraticDistributionSubclass(a, b, c_, convex),
                        )
                        self.assertNotEqual(
                            QuadraticDistributionSubclass(a, b, c_, convex),
                            parametric.QuadraticDistribution(a, b, c, convex),
                        )
                    for convex_ in [False, True]:
                        if convex_ == convex:
                            continue
                        self.assertNotEqual(
                            parametric.QuadraticDistribution(a, b, c, convex),
                            QuadraticDistributionSubclass(a, b, c, convex_),
                        )
                        self.assertNotEqual(
                            QuadraticDistributionSubclass(a, b, c, convex_),
                            parametric.QuadraticDistribution(a, b, c, convex),
                        )

    def test___str__(self):
        self.assertEqual(
            str(parametric.QuadraticDistribution(0., 1., 1, convex=False)),
            f"QuadraticDistribution("
                f"a={np.array(0.)[()]!s},"
                f" b={np.array(1.)[()]!s},"
                f" c={np.array(1)[()]!s},"
                f" convex={False!s}"
            f")",
        )

    def test___repr__(self):
        self.assertEqual(
            repr(parametric.QuadraticDistribution(0., 1., 1, convex=False)),
            f"QuadraticDistribution("
                f"a={np.array(0.)[()]!r},"
                f" b={np.array(1.)[()]!r},"
                f" c={np.array(1)[()]!r},"
                f" convex={False!r}"
            f")",
        )

    def test_sample(self):
        # Test when a = b.
        a, b = 0., 0.
        for c in [1, 10]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                # without expicit value for size
                y = dist.sample()
                self.assertTrue(np.isscalar(y))
                self.assertEqual(y, 0.)
                # scalar
                y = dist.sample(None)
                self.assertTrue(np.isscalar(y))
                self.assertEqual(y, 0.)
                # 1D array
                ys = dist.sample(100)
                self.assertTrue(np.array_equal(ys, np.zeros(100)))
                # 2D array
                ys = dist.sample((10, 10))
                self.assertTrue(np.array_equal(ys, np.zeros((10, 10))))

        # Test when a != b.
        a, b = 0., 1.
        for c in [1, 10]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                # without explicit value for size
                y = dist.sample()
                self.assertTrue(np.isscalar(y))
                self.assertLess(a, y)
                self.assertGreater(b, y)
                # scalar
                y = dist.sample(None)
                self.assertTrue(np.isscalar(y))
                self.assertLess(a, y)
                self.assertGreater(b, y)
                # 1D array
                ys = dist.sample(100)
                self.assertEqual(ys.shape, (100,))
                self.assertLess(a, np.min(ys))
                self.assertGreater(b, np.max(ys))
                # 2D array
                ys = dist.sample((10, 10))
                self.assertEqual(ys.shape, (10, 10))
                self.assertLess(a, np.min(ys))
                self.assertGreater(b, np.max(ys))
        # Test c = 2, the samples should be uniformly distributed.
        a, b, c = 0., 1., 2
        ys = parametric.QuadraticDistribution(a, b, c).sample(2_500)
        self.assertLess(a, np.min(ys))
        self.assertGreater(b, np.max(ys))
        self.assertLess(abs(np.mean(ys < 0.5) - 0.5), 0.05)

    def test_pdf(self):
        # Test when a = b.

        # When a = b, the distribution is a point mass.
        a, b, c = 0., 0., 2
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # scalar
            self.assertTrue(np.isscalar(dist.pdf(a)))
            self.assertEqual(dist.pdf(a - 1.), 0.)
            self.assertEqual(dist.pdf(a), np.inf)
            self.assertEqual(dist.pdf(a + 1.), 0.)
            # broadcasting
            #   1D array
            self.assertEqual(
                dist.pdf([a - 1., a, a + 1.]).tolist(),
                [0., np.inf, 0.],
            )
            # 2D array
            self.assertEqual(
                dist.pdf([[a - 1.], [a], [a + 1.]]).tolist(),
                [[0.], [np.inf], [0.]],
            )
            # 3D array
            self.assertEqual(
                dist.pdf([[[a - 1.]], [[a]], [[a + 1.]]]).tolist(),
                [[[0.]], [[np.inf]], [[0.]]],
            )

        # Test when a != b.

        # Test inside of the distribution's support.
        a, b, c = 0., 1., 2
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.pdf(a + (n / 5.) * (b - a))))
                # When c = 2, the distribution is uniform.
                self.assertAlmostEqual(
                    dist.pdf(a + (n / 5.) * (b - a)),
                    1. / (b - a),
                )
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.full_like(us, 1. / (b - a)).tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.full_like(us, 1. / (b - a)).tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.full_like(us, 1. / (b - a)).tolist(),
                )

        # Test outside of the distribution's support.
        for a, b, c in [(0., 1., 1), (0., 1., 2)]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                self.assertEqual(dist.pdf(a - 1e-10), 0.)
                self.assertEqual(dist.pdf(a - 10), 0.)
                self.assertEqual(dist.pdf(b + 1e-10), 0.)
                self.assertEqual(dist.pdf(b + 10), 0.)

    def test_cdf(self):
        # Test when a = b.

        # When a = b, the distribution is a point mass.
        a, b, c = 0., 0., 2
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # scalar
            self.assertTrue(np.isscalar(dist.cdf(a)))
            self.assertEqual(dist.cdf(a - 1.), 0.)
            self.assertEqual(dist.cdf(a), 1.)
            self.assertEqual(dist.cdf(a + 1.), 1.)
            # broadcasting
            #   1D array
            self.assertEqual(
                dist.cdf([a - 1., a, a + 1.]).tolist(),
                [0., 1., 1.],
            )
            # 2D array
            self.assertEqual(
                dist.cdf([[a - 1.], [a], [a + 1.]]).tolist(),
                [[0.], [1.], [1.]],
            )
            # 3D array
            self.assertEqual(
                dist.cdf([[[a - 1.]], [[a]], [[a + 1.]]]).tolist(),
                [[[0.]], [[1.]], [[1.]]],
            )

        # Test when a != b.

        # Test inside of the distribution's support.
        a, b, c = 0., 1., 2
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.cdf(a + (n / 5.) * (b - a))))
                # When c = 2, the distribution is uniform.
                self.assertAlmostEqual(
                    dist.cdf(a + (n / 5.) * (b - a)),
                    n / 5.,
                )
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )

        # Test outside of the distribution's support.
        for a, b, c in [(0., 1., 1), (0., 1., 2)]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                self.assertEqual(dist.cdf(a - 1e-10), 0.)
                self.assertEqual(dist.cdf(a - 10), 0.)
                self.assertEqual(dist.cdf(b + 1e-10), 1.)
                self.assertEqual(dist.cdf(b + 10), 1.)

    def test_ppf(self):
        # Test when a = b.

        # When a = b, the distribution is a point mass.
        a, b, c = 0., 0., 2
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.ppf(n / 5.)))
                self.assertEqual(dist.ppf(n / 5.), a)
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(dist.ppf(us).tolist(), [a] * 5)
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(dist.ppf(us).tolist(), [[a] * 3] * 5)
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(dist.ppf(us).tolist(), [[[a] * 2] * 3] * 5)

        # Test when a != b.

        a, b, c = 0., 1., 2
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.ppf(n / 5.)))
                # When c = 2, the distribution is uniform.
                self.assertAlmostEqual(dist.ppf(n / 5.), a + (n / 5.) * (b - a))
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                )

    def test_quantile_tuning_curve(self):
        n_trials = 2_000
        q = 0.5
        a, b = 0., 1.
        for c in [1, 10]:
            for convex in [False, True]:
                for minimize in [None, False, True]:
                    # NOTE: When minimize is None, default to convex.
                    expect_minimize = (
                        minimize
                        if minimize is not None else
                        convex
                    )

                    dist = parametric.QuadraticDistribution(
                        a,
                        b,
                        c,
                        convex=convex,
                    )
                    yss = dist.sample((n_trials, 5))
                    # Use the binomial confidence interval for the quantile.
                    idx_lo, idx_pt, idx_hi = stats.binom(n_trials, q).ppf(
                        [1e-6 / 2, 0.5, 1 - 1e-6 / 2],
                    ).astype(int)
                    curve_lo, curve, curve_hi = np.sort(
                        np.minimum.accumulate(yss, axis=1)
                        if expect_minimize else
                        np.maximum.accumulate(yss, axis=1),
                        axis=0,
                    )[(
                        idx_lo,  # lower 1 - 1e-6 confidence bound
                        idx_pt,  # point estimate
                        idx_hi,  # upper 1 - 1e-6 confidence bound
                    ), :]
                    atol = np.max(curve_hi - curve_lo) / 2

                    # Test when n is integral.
                    #   scalar
                    for n in range(1, 6):
                        self.assertTrue(np.isscalar(
                            dist.quantile_tuning_curve(
                                n,
                                q=q,
                                minimize=minimize,
                            ),
                        ))
                        self.assertAlmostEqual(
                            dist.quantile_tuning_curve(
                                n,
                                q=q,
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=atol,
                        )
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [n],
                                q=q,
                                minimize=minimize,
                            ),
                            [
                                dist.quantile_tuning_curve(
                                    n,
                                    q=q,
                                    minimize=minimize,
                                ),
                            ],
                        ))
                    #   1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [1, 2, 3, 4, 5],
                            q=q,
                            minimize=minimize,
                        ),
                        curve,
                        atol=atol,
                    ))
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [3, 1, 5],
                            q=q,
                            minimize=minimize,
                        ),
                        [curve[2], curve[0], curve[4]],
                        atol=atol,
                    ))
                    #   2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [1, 2, 3],
                                [3, 1, 5],
                            ],
                            q=q,
                            minimize=minimize,
                        ),
                        [
                            [curve[0], curve[1], curve[2]],
                            [curve[2], curve[0], curve[4]],
                        ],
                        atol=atol,
                    ))

                    # Test when n is non-integral.
                    #   scalar
                    for n in range(1, 6):
                        self.assertTrue(np.isscalar(
                            dist.quantile_tuning_curve(
                                n/10.,
                                q=1 - (1 - q)**(1/10)
                                  if expect_minimize else
                                  q**(1/10),
                                minimize=minimize,
                            ),
                        ))
                        self.assertAlmostEqual(
                            dist.quantile_tuning_curve(
                                n/10.,
                                q=1 - (1 - q)**(1/10)
                                  if expect_minimize else
                                  q**(1/10),
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=atol,
                        )
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [n/10.],
                                q=1 - (1 - q)**(1/10)
                                  if expect_minimize else
                                  q**(1/10),
                                minimize=minimize,
                            ),
                            [
                                dist.quantile_tuning_curve(
                                    n/10.,
                                    q=1 - (1 - q)**(1/10)
                                      if expect_minimize else
                                      q**(1/10),
                                    minimize=minimize,
                                ),
                            ],
                        ))
                    #   1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [0.1, 0.2, 0.3, 0.4, 0.5],
                            q=1 - (1 - q)**(1/10)
                              if expect_minimize else
                              q**(1/10),
                            minimize=minimize,
                        ),
                        curve,
                        atol=atol,
                    ))
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [0.3, 0.1, 0.5],
                            q=1 - (1 - q)**(1/10)
                              if expect_minimize else
                              q**(1/10),
                            minimize=minimize,
                        ),
                        [curve[2], curve[0], curve[4]],
                        atol=atol,
                    ))
                    #   2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [0.1, 0.2, 0.3],
                                [0.3, 0.1, 0.5],
                            ],
                            q=1 - (1 - q)**(1/10)
                              if expect_minimize else
                              q**(1/10),
                            minimize=minimize,
                        ),
                        [
                            [curve[0], curve[1], curve[2]],
                            [curve[2], curve[0], curve[4]],
                        ],
                        atol=atol,
                    ))

                    # Test ns <= 0.
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            0,
                            q=q,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            -1,
                            q=q,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [0],
                            q=q,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [-2],
                            q=q,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [0, 1],
                            q=q,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [-2, 1],
                            q=q,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [[0], [1]],
                            q=q,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [[-2], [1]],
                            q=q,
                            minimize=minimize,
                        )

    def test_average_tuning_curve(self):
        a, b = 0., 1.
        for c in [1, 10]:
            for convex in [False, True]:
                for minimize in [None, False, True]:
                    # NOTE: When minimize is None, default to convex.
                    expect_minimize = (
                        minimize
                        if minimize is not None else
                        convex
                    )

                    dist = parametric.QuadraticDistribution(
                        a,
                        b,
                        c,
                        convex=convex,
                    )
                    yss = dist.sample((2_000, 5))
                    curve = np.mean(
                        np.minimum.accumulate(yss, axis=1)
                        if expect_minimize else
                        np.maximum.accumulate(yss, axis=1),
                        axis=0,
                    )

                    # Test when n is integral.
                    #   scalar
                    for n in range(1, 6):
                        self.assertTrue(np.isscalar(
                            dist.average_tuning_curve(
                                n,
                                minimize=minimize,
                            ),
                        ))
                        self.assertAlmostEqual(
                            dist.average_tuning_curve(
                                n,
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=0.075,
                        )
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [n],
                                minimize=minimize,
                            ),
                            [
                                dist.average_tuning_curve(
                                    n,
                                    minimize=minimize,
                                ),
                            ],
                        ))
                    #   1D array
                    self.assertTrue(np.allclose(
                        dist.average_tuning_curve(
                            [1, 2, 3, 4, 5],
                            minimize=minimize,
                        ),
                        curve,
                        atol=0.075,
                    ))
                    self.assertTrue(np.allclose(
                        dist.average_tuning_curve(
                            [3, 1, 5],
                            minimize=minimize,
                        ),
                        [curve[2], curve[0], curve[4]],
                        atol=0.075,
                    ))
                    #   2D array
                    self.assertTrue(np.allclose(
                        dist.average_tuning_curve(
                            [
                                [1, 2, 3],
                                [3, 1, 5],
                            ],
                            minimize=minimize,
                        ),
                        [
                            [curve[0], curve[1], curve[2]],
                            [curve[2], curve[0], curve[4]],
                        ],
                        atol=0.075,
                    ))

                    # Test when n is non-integral.
                    #   scalar
                    for n in range(1, 6):
                        self.assertTrue(np.isscalar(
                            dist.average_tuning_curve(
                                n + (0.5 if expect_minimize else -0.5),
                                minimize=minimize,
                            ),
                        ))
                        self.assertLess(
                            dist.average_tuning_curve(
                                n + (0.5 if expect_minimize else -0.5),
                                minimize=minimize,
                            ),
                            dist.average_tuning_curve(
                                n,
                                minimize=minimize,
                            ),
                        )
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [n - 0.5],
                                minimize=minimize,
                            ),
                            [
                                dist.average_tuning_curve(
                                    n - 0.5,
                                    minimize=minimize,
                                ),
                            ],
                        ))
                    #   1D array
                    self.assertTrue(np.all(
                        dist.average_tuning_curve(
                            np.arange(1, 6)
                              + (0.5 if expect_minimize else -0.5),
                            minimize=minimize,
                        )
                        < dist.average_tuning_curve(
                            np.arange(1, 6),
                            minimize=minimize,
                        ),
                    ))
                    #   2D array
                    self.assertTrue(np.all(
                        dist.average_tuning_curve(
                            np.arange(1, 11).reshape(5, 2)
                              + (0.5 if expect_minimize else -0.5),
                            minimize=minimize,
                        ) < dist.average_tuning_curve(
                            np.arange(1, 11).reshape(5, 2),
                            minimize=minimize,
                        ),
                    ))

                    # Test ns <= 0.
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            0,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            -1,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            [0],
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            [-2],
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            [0, 1],
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            [-2, 1],
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            [[0], [1]],
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.average_tuning_curve(
                            [[-2], [1]],
                            minimize=minimize,
                        )

    @pytest.mark.level(3)
    def test_fit(self):
        n_samples = 2_048

        for a, b in [
                # Test when a = b.
                (    0.,    0.),
                (    1.,    1.),
                # Test when a != b.
                (    0.,    1.),
                (   -1.,    1.),
                ( 1e-50, 2e-50),
                (  1e50,  2e50),
        ]:
            # NOTE: When c = 2, convex is unidentifiable because the
            # distribution is uniform whether convex is True or
            # False. Thus, we test c = 1 and 3.
            for c in [1, 3]:
                for convex in [False, True]:
                    dist = parametric.QuadraticDistribution(
                        a, b, c, convex,
                    )
                    ys = dist.sample(n_samples)

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore" if a == b else "default",
                            message=r"Parameters might be unidentifiable.",
                            category=RuntimeWarning,
                        )
                        dist_hat = parametric.QuadraticDistribution.fit(ys)

                    # Check the parameters are approximately correct.
                    self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
                    self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
                    # NOTE: When a = b, the distribution is a point mass
                    # for any c or convex, thus they are unidentifiable.
                    if a != b:
                        self.assertEqual(dist_hat.c, c)
                        self.assertEqual(dist_hat.convex, convex)

    def test_sample_defaults_to_global_random_number_generator(self):
        # sample should be deterministic if global seed is set.
        dist = parametric.QuadraticDistribution(0., 1., 2)
        #   Before setting the seed, two samples should be unequal.
        self.assertNotEqual(dist.sample(16).tolist(), dist.sample(16).tolist())
        #   After setting the seed, two samples should be unequal.
        opda.random.set_seed(0)
        self.assertNotEqual(dist.sample(16).tolist(), dist.sample(16).tolist())
        #   Resetting the seed should produce the same sample.
        opda.random.set_seed(0)
        first_sample = dist.sample(16)
        opda.random.set_seed(0)
        second_sample = dist.sample(16)
        self.assertEqual(first_sample.tolist(), second_sample.tolist())

    def test_sample_is_deterministic_given_generator_argument(self):
        dist = parametric.QuadraticDistribution(0., 1., 2)
        # Reusing the same generator, two samples should be unequal.
        generator = np.random.default_rng(0)
        self.assertNotEqual(
            dist.sample(16, generator=generator).tolist(),
            dist.sample(16, generator=generator).tolist(),
        )
        # Using generators in the same state should produce the same sample.
        self.assertEqual(
            dist.sample(16, generator=np.random.default_rng(0)).tolist(),
            dist.sample(16, generator=np.random.default_rng(0)).tolist(),
        )

    def test_pdf_on_boundary_of_support(self):
        for convex in [False, True]:
            a, b, c = 0., 1., 1
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            self.assertEqual(dist.pdf(a), np.inf if convex else 0.5)
            self.assertEqual(dist.pdf(b), 0.5 if convex else np.inf)

            a, b, c = 0., 1., 2
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            self.assertEqual(dist.pdf(a), 1.)
            self.assertEqual(dist.pdf(b), 1.)

    def test_pdf_is_consistent_across_scales(self):
        for c in [1, 2, 10]:
            for convex in [False, True]:
                ys = np.linspace(0., 1., num=100)

                ps = parametric.QuadraticDistribution(
                    0., 1., c, convex=convex,
                ).pdf(ys)
                for a, b in [(-1., 1.), (1., 10.)]:
                    dist = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    )
                    self.assertTrue(np.allclose(
                        ps / (b - a),
                        dist.pdf(a + (b - a) * ys),
                    ))

    def test_pdf_matches_numerical_derivative_of_cdf(self):
        for a, b in [(-1., 1.), (0., 1.), (1., 10.)]:
            for c in [1, 2, 10]:
                for convex in [False, True]:
                    dist = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    )

                    # Omit a and b from the numerical derivatives since
                    # they're on the boundary.
                    ys = np.linspace(a, b, num=102)[1:-1]
                    dy = 1e-7
                    self.assertTrue(np.allclose(
                        dist.pdf(ys),
                        (dist.cdf(ys + dy) - dist.cdf(ys - dy)) / (2 * dy),
                    ))

    def test_cdf_on_boundary_of_support(self):
        a, b = 0., 1.
        for c in [1, 2, 10]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                self.assertEqual(dist.cdf(a), 0.)
                self.assertEqual(dist.cdf(b), 1.)

    def test_cdf_is_consistent_across_scales(self):
        for c in [1, 2, 10]:
            for convex in [False, True]:
                ys = np.linspace(0., 1., num=100)

                ps = parametric.QuadraticDistribution(
                    0., 1., c, convex=convex,
                ).cdf(ys)
                for a, b in [(-1., 1.), (1., 10.)]:
                    dist = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    )
                    self.assertTrue(np.allclose(
                        ps,
                        dist.cdf(a + (b - a) * ys),
                    ))

    @pytest.mark.level(1)
    def test_cdf_agrees_with_sampling_definition(self):
        for a, b in [(-1., 1.), (0., 1.), (1., 10.)]:
            # NOTE: Keep c low because the rejection sampling below will
            # reject too many samples for large c.
            for c in [1, 2, 3]:
                for convex in [False, True]:
                    dist = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    )

                    # Sample from the quadratic distribution according
                    # to its derivation: uniform random variates passed
                    # through a quadratic function.
                    ys = np.sum(
                        self.generator.uniform(-1., 1., size=(150_000, c))**2,
                        axis=-1,
                    )
                    # Filter for points in the sphere of radius 1, to
                    # avoid distortions from the hypercube's boundary.
                    ys = ys[ys <= 1]
                    # Adjust the data for the distribution's parameters.
                    ys = (
                        a + (b - a) * ys
                        if convex else  # concave
                        b - (b - a) * ys
                    )

                    # Check the sample comes from the distribution using
                    # the KS test.
                    p_value = stats.kstest(
                        ys,
                        dist.cdf,
                        alternative="two-sided",
                    ).pvalue
                    self.assertGreater(p_value, 1e-6)

    def test_ppf_is_inverse_of_cdf(self):
        # NOTE: The quantile function is always an almost sure left
        # inverse of the CDF. For continuous distributions like the
        # quadratic distribution, the quantile function is also the
        # right inverse of the cumulative distribution function.
        for a, b in [(0., 0.), (0., 1.)]:
            for c in [1, 10]:
                for convex in [False, True]:
                    dist = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    )

                    ys = dist.sample(100)
                    self.assertTrue(np.allclose(dist.ppf(dist.cdf(ys)), ys))

                    if a == b:
                        # When a = b, the distribution is discrete and
                        # the quantile function is not a right inverse
                        # of the CDF.
                        continue

                    us = self.generator.uniform(0, 1, size=100)
                    self.assertTrue(np.allclose(dist.cdf(dist.ppf(us)), us))

    def test_ppf_at_extremes(self):
        for a, b in [(0., 0.), (0., 1.)]:
            for c in [1, 10]:
                for convex in [False, True]:
                    dist = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    )
                    self.assertEqual(dist.ppf(0. - 1e-12), a)
                    self.assertEqual(dist.ppf(0.), a)
                    self.assertEqual(dist.ppf(1.), b)
                    self.assertEqual(dist.ppf(1. + 1e-12), b)

    def test_quantile_tuning_curve_with_different_quantiles(self):
        n_trials = 2_000
        a, b, c, convex = 0., 1., 1, False
        for q in [0.25, 0.75]:
            for minimize in [None, False, True]:
                # NOTE: When minimize is None, default to convex.
                expect_minimize = (
                    minimize
                    if minimize is not None else
                    convex
                )

                dist = parametric.QuadraticDistribution(
                    a,
                    b,
                    c,
                    convex=convex,
                )
                yss = dist.sample((n_trials, 5))
                # Use the binomial confidence interval for the quantile.
                idx_lo, idx_pt, idx_hi = stats.binom(n_trials, q).ppf(
                    [1e-6 / 2, 0.5, 1 - 1e-6 / 2],
                ).astype(int)
                curve_lo, curve, curve_hi = np.sort(
                    np.minimum.accumulate(yss, axis=1)
                    if expect_minimize else
                    np.maximum.accumulate(yss, axis=1),
                    axis=0,
                )[(
                    idx_lo,  # lower 1 - 1e-6 confidence bound
                    idx_pt,  # point estimate
                    idx_hi,  # upper 1 - 1e-6 confidence bound
                ), :]
                atol = np.max(curve_hi - curve_lo) / 2

                self.assertTrue(np.allclose(
                    dist.quantile_tuning_curve(
                        [1, 2, 3, 4, 5],
                        q=q,
                        minimize=minimize,
                    ),
                    curve,
                    atol=atol,
                ))

    def test_quantile_tuning_curve_minimize_is_dual_to_maximize(self):
        for a, b, c in [(0., 1., 1), (-1., 10., 2)]:
            for convex in [False, True]:
                ns = np.arange(1, 17)

                self.assertTrue(np.allclose(
                    parametric
                      .QuadraticDistribution(a, b, c, convex=convex)
                      .quantile_tuning_curve(ns, minimize=False),
                    -parametric
                      .QuadraticDistribution(-b, -a, c, convex=not convex)
                      .quantile_tuning_curve(ns, minimize=True),
                ))
                self.assertTrue(np.allclose(
                    parametric
                      .QuadraticDistribution(a, b, c, convex=convex)
                      .quantile_tuning_curve(ns, minimize=True),
                    -parametric
                      .QuadraticDistribution(-b, -a, c, convex=not convex)
                      .quantile_tuning_curve(ns, minimize=False),
                ))

    def test_average_tuning_curve_minimize_is_dual_to_maximize(self):
        for a, b, c in [(0., 1., 1), (-1., 10., 2)]:
            for convex in [False, True]:
                ns = np.arange(1, 17)

                self.assertTrue(np.allclose(
                    parametric
                      .QuadraticDistribution(a, b, c, convex=convex)
                      .average_tuning_curve(ns, minimize=False),
                    -parametric
                      .QuadraticDistribution(-b, -a, c, convex=not convex)
                      .average_tuning_curve(ns, minimize=True),
                ))
                self.assertTrue(np.allclose(
                    parametric
                      .QuadraticDistribution(a, b, c, convex=convex)
                      .average_tuning_curve(ns, minimize=True),
                    -parametric
                      .QuadraticDistribution(-b, -a, c, convex=not convex)
                      .average_tuning_curve(ns, minimize=False),
                ))

    @pytest.mark.level(3)
    def test_fit_applies_limits_correctly(self):
        n_samples = 2_048

        c, convex = 1, False
        for a, b in [
                # Test when a = b.
                (0., 0.),
                # Test when a != b.
                (0., 1.),
        ]:
            dist = parametric.QuadraticDistribution(a, b, c, convex)
            ys = dist.sample(n_samples)

            # Test fit with limits.

            for censor_side in ["left", "right", "both"]:
                for censor_trivial in [False, True]:
                    limit_lower = (
                        -np.inf         if censor_side == "right" else
                        a - 0.1         if censor_trivial else
                        dist.ppf(0.1)   if a != b else
                        a + 0.1       # if a == b
                    )
                    limit_upper = (
                        np.inf          if censor_side == "left" else
                        b + 0.1         if censor_trivial else
                        dist.ppf(0.975) if a != b else
                        b - 0.1       # if a == b
                    )

                    if a == b and not censor_trivial:
                        # When a = b, a non-trivial censoring will
                        # censor all observations.
                        with self.assertRaises(ValueError):
                            dist_hat =\
                                parametric.QuadraticDistribution.fit(
                                    ys=ys,
                                    limits=(limit_lower, limit_upper),
                                )
                        continue

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore" if a == b else "default",
                            message=r"Parameters might be unidentifiable.",
                            category=RuntimeWarning,
                        )
                        dist_hat = parametric.QuadraticDistribution.fit(
                            ys=ys,
                            limits=(limit_lower, limit_upper),
                        )

                    self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
                    self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
                    # NOTE: When a = b, the distribution is a point mass
                    # for any c or convex, thus they are unidentifiable.
                    if a != b:
                        self.assertEqual(dist_hat.c, c)
                        self.assertEqual(dist_hat.convex, convex)

            # Test limits is a left-open interval.

            threshold = a if a == b else a + 0.75 * (b - a)

            if a == b:
                # Since limits is left-open and all observations equal a,
                # they are all censored when the lower limit is a.
                with self.assertRaises(ValueError):
                    dist_hat = parametric.QuadraticDistribution.fit(
                        ys=ys,
                        limits=(threshold, np.inf),
                    )
                # Since limits is right-closed, no observations are
                # censored when just the upper limit equals b.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Parameters might be unidentifiable.",
                        category=RuntimeWarning,
                    )
                    dist_hat = parametric.QuadraticDistribution.fit(
                        ys=ys,
                        limits=(-np.inf, threshold),
                    )
                self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
                self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
            else:  # a != b
                # Since limits is left-open, moving observations below
                # it to the lower endpoint will not effect the fit.
                dist_hat = parametric.QuadraticDistribution.fit(
                    ys=np.where(ys <= threshold, threshold, ys),
                    limits=(threshold, np.inf),
                )
                self.assertAlmostEqual(dist_hat.a, a, delta=2e-1 * (b - a))
                self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
                self.assertEqual(dist_hat.c, c)
                self.assertEqual(dist_hat.convex, convex)

    @pytest.mark.level(3)
    def test_fit_applies_constraints_correctly(self):
        n_samples = 2_048

        c, convex = 1, False
        for a, b in [
                # Test when a = b.
                (0., 0.),
                # Test when a != b.
                (0., 1.),
        ]:
            dist = parametric.QuadraticDistribution(a, b, c, convex)
            ys = dist.sample(n_samples)

            # Test fit with feasible constraints.

            for constraints in [
                    # Constrain a.
                    {"a": a},
                    {"a": (a, b)},
                    {"a": (-np.inf, a)},
                    # Constrain b.
                    {"b": b},
                    {"b": (a, b)},
                    {"b": (b, np.inf)},
                    # Constrain c.
                    {"c": c},
                    {"c": (0, c)},
                    {"c": (c, 100)},
                    # Constrain convex.
                    {"convex": convex},
                    {"convex": (convex,)},
                    {"convex": (False, True)},
                    # Fix two numeric parameters.
                    {"a": a, "b": b},
                    {"a": a, "c": c},
                    {"b": b, "c": c},
                    # Fix all three numeric parameters.
                    {"a": a, "b": b, "c": c},
                    # Fix all parameters.
                    {"a": a, "b": b, "c": c, "convex": convex},
            ]:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore" if a == b else "default",
                        message=r"Parameters might be unidentifiable.",
                        category=RuntimeWarning,
                    )
                    warnings.filterwarnings(
                        "ignore" if "c" in constraints else "default",
                        message=r"The constraint for c includes values"
                                r" outside of 1 to 10,",
                        category=RuntimeWarning,
                    )
                    dist_hat = parametric.QuadraticDistribution.fit(
                        ys=ys,
                        constraints=constraints,
                    )

                # Check the fit recovers the distribution.
                self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
                self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
                # NOTE: When a = b, the distribution is a point mass for
                # any c or convex, thus they are unidentifiable.
                if a != b:
                    self.assertEqual(dist_hat.c, c)
                    self.assertEqual(dist_hat.convex, convex)

                # Check the constraint was obeyed.
                for parameter, value in constraints.items():
                    estimate = getattr(dist_hat, parameter)
                    if np.isscalar(value):
                        self.assertEqual(estimate, value)
                    elif parameter in {"a", "b", "c"}:
                        lo, hi = value
                        self.assertGreaterEqual(estimate, lo)
                        self.assertLessEqual(estimate, hi)
                    elif parameter in {"convex"}:
                        self.assertIn(estimate, value)
                    else:
                        raise ValueError(
                            f"Unrecognized parameter: {parameter}",
                        )

            # Test fit with infeasible constraints.

            for censor in [False, True]:
                if a == b and censor:
                    # When a = b, all observations are equal thus it is
                    # impossible to censor some but not all of
                    # them. The case of censoring all observations is
                    # already covered in other tests.
                    continue

                limits = (
                    (a + 0.1 * (b - a), b - 0.1 * (b - a))
                    if censor else
                    (-np.inf, np.inf)
                )

                for constraints, exception in [
                        # Constrain a to be greater than observed ys.
                        ({"a": (max(a, limits[0]) + 0.1, np.inf)}, ValueError),
                        ({"a": (1e8, 1e9)}, ValueError),
                        # Constrain b to be less than observed ys.
                        ({"b": (-np.inf, min(b, limits[1]) - 0.1)}, ValueError),
                        ({"b": (-1e9, -1e8)}, ValueError),
                        # Constrain c to unsupported values.
                        ({"c": (-10, 0)}, ValueError),
                        ({"c": (11, 20)}, ValueError),
                        # Constrain a or b outside estimated bounds.
                        ({"a": (-1e9, -1e8)}, exceptions.OptimizationError),
                        ({"b": (1e8, 1e9)}, exceptions.OptimizationError),
                ]:
                    with warnings.catch_warnings(),\
                         self.assertRaises(exception):
                        warnings.filterwarnings(
                            "ignore" if a == b else "default",
                            message=r"Parameters might be unidentifiable.",
                            category=RuntimeWarning,
                        )
                        dist_hat = parametric.QuadraticDistribution.fit(
                            ys=ys,
                            limits=limits,
                            constraints=constraints,
                        )

    @pytest.mark.level(2)
    def test_fit_for_interactions_between_limits_and_constraints(self):
        n_samples = 2_048

        a, b, c, convex = 0., 1., 1, False
        dist = parametric.QuadraticDistribution(a, b, c, convex)

        # Test when constraints keeps a above the least of ys but not
        # the least *observed* y, thus the constraint is still feasible.
        y_min = -1.
        ys = dist.sample(n_samples)
        ys[np.argmin(ys)] = y_min
        dist_hat = parametric.QuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"a": ((y_min + a) / 2, np.inf)},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

        # Test when constraints keeps b below the greatest of ys but not
        # the greatest *observed* y, thus the constraint is still feasible.
        y_max = 2.
        ys = dist.sample(n_samples)
        ys[np.argmax(ys)] = y_max
        dist_hat = parametric.QuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"b": (-np.inf, (b + y_max) / 2)},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

        # Test when constraints prevents the distribution from putting
        # any probability in one of the censored tails. In that case,
        # observing a point in the tail is a zero probability event,
        # which can ruin the fit. If a point equals the lower limit (as
        # often happens with aggressive rounding), then it gets placed
        # in the left tail since limits defines a left-open interval.
        # Thus, it's important to test this edge case.
        #   zero probability mass in the left tail.
        y_min = a
        ys = dist.sample(n_samples)
        ys[np.argmin(ys)] = y_min
        dist_hat = parametric.QuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"a": (a, np.inf)},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)
        #   zero probability mass in the right tail.
        y_max = b
        ys = dist.sample(n_samples)
        ys[np.argmax(ys)] = y_max
        dist_hat = parametric.QuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"b": (-np.inf, b)},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

    @pytest.mark.level(2)
    def test_fit_with_ties(self):
        n_samples = 2_048

        a, b, c, convex = 0., 1., 1, False
        dist = parametric.QuadraticDistribution(a, b, c, convex)

        # Test fit with aggressive (1e-2) and light (1e-4) rounding.

        for decimals in [2, 4]:
            ys = np.round(dist.sample(n_samples), decimals=decimals)

            dist_hat = parametric.QuadraticDistribution.fit(ys)

            self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
            self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
            self.assertEqual(dist_hat.c, c)
            self.assertEqual(dist_hat.convex, convex)

        # Test fit with small spacings (e.g., floating point errors).

        ys = np.round(dist.sample(n_samples), decimals=2)
        ys += (
            self.generator.choice([-256, -16, -1, 0, 1, 16, 256], size=len(ys))
            * np.spacing(ys)
        )

        dist_hat = parametric.QuadraticDistribution.fit(ys)

        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

    def test_fit_accepts_integer_data(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Parameters might be unidentifiable.",
                category=RuntimeWarning,
            )
            dist_hat = parametric.QuadraticDistribution.fit([0] * 8)

        self.assertEqual(dist_hat.a, 0.)
        self.assertEqual(dist_hat.b, 0.)

    @pytest.mark.level(2)
    def test_fit_defaults_to_global_random_number_generator(self):
        n_samples = 256
        # fit should be deterministic if global seed is set.
        ys = parametric.QuadraticDistribution(
            0., 1., 1, False,
        ).sample(n_samples)
        #   Before setting the seed, two fits should be unequal.
        self.assertNotEqual(
            parametric.QuadraticDistribution.fit(ys),
            parametric.QuadraticDistribution.fit(ys),
        )
        #   After setting the seed, two fits should be unequal.
        opda.random.set_seed(0)
        self.assertNotEqual(
            parametric.QuadraticDistribution.fit(ys),
            parametric.QuadraticDistribution.fit(ys),
        )
        #   Resetting the seed should produce the same fit.
        opda.random.set_seed(0)
        first_fit = parametric.QuadraticDistribution.fit(ys)
        opda.random.set_seed(0)
        second_fit = parametric.QuadraticDistribution.fit(ys)
        self.assertEqual(first_fit, second_fit)

    @pytest.mark.level(2)
    def test_fit_is_deterministic_given_generator_argument(self):
        n_samples = 256
        ys = parametric.QuadraticDistribution(
            0., 1., 1, False,
        ).sample(n_samples)
        # Reusing the same generator, two fits should be unequal.
        generator = np.random.default_rng(0)
        self.assertNotEqual(
            parametric.QuadraticDistribution.fit(ys, generator=generator),
            parametric.QuadraticDistribution.fit(ys, generator=generator),
        )
        # Using generators in the same state should produce the same fit.
        self.assertEqual(
            parametric.QuadraticDistribution.fit(
                ys, generator=np.random.default_rng(0),
            ),
            parametric.QuadraticDistribution.fit(
                ys, generator=np.random.default_rng(0),
            ),
        )

    @pytest.mark.level(1)
    def test_fit_when_all_ys_are_negative(self):
        n_samples = 8

        # Due to the small sample size, fit might estimate that c = 2,
        # in which case it will fire a warning about identifiability.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Parameters might be unidentifiable.",
                category=RuntimeWarning,
            )

            # Test when a = b.
            ys = [-1.] * n_samples
            parametric.QuadraticDistribution.fit(
                ys,
                method="maximum_spacing",
            )

            # Test when a != b.
            ys = parametric.QuadraticDistribution(
                -2., -1., 1, False,
            ).sample(n_samples)
            parametric.QuadraticDistribution.fit(
                ys,
                method="maximum_spacing",
            )

    def test_fit_with_censored_infinity(self):
        n_samples = 2_048

        a, b, c, convex = 0., 1., 2, False
        dist = parametric.QuadraticDistribution(a, b, c, convex)
        ys = dist.sample(n_samples)

        # NOTE: Use constraints in fit below to make fit run faster.

        # Test when ys contains inf.
        #   positive infinitiy
        with self.assertRaises(ValueError):
            parametric.QuadraticDistribution.fit(
                ys=np.concatenate([[np.inf], ys]),
                limits=(-np.inf, np.inf),
                constraints={"c": c, "convex": convex},
            )
        #   negative infinitiy
        with self.assertRaises(ValueError):
            parametric.QuadraticDistribution.fit(
                ys=np.concatenate([[-np.inf], ys]),
                limits=(-np.inf, np.inf),
                constraints={"c": c, "convex": convex},
            )
        #   both positive and negative infinity
        with self.assertRaises(ValueError):
            parametric.QuadraticDistribution.fit(
                ys=np.concatenate([[-np.inf, np.inf], ys]),
                limits=(-np.inf, np.inf),
                constraints={"c": c, "convex": convex},
            )

        # Test when ys contains inf but it is censored.
        #   positive infinitiy
        dist_hat = parametric.QuadraticDistribution.fit(
            ys=np.concatenate([[np.inf], ys]),
            limits=(-np.inf, b),
            constraints={"c": c, "convex": convex},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        #   negative infinitiy
        dist_hat = parametric.QuadraticDistribution.fit(
            ys=np.concatenate([[-np.inf], ys]),
            limits=(a, np.inf),
            constraints={"c": c, "convex": convex},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        #   both positive and negative infinity
        dist_hat = parametric.QuadraticDistribution.fit(
            ys=np.concatenate([[-np.inf, np.inf], ys]),
            limits=(a, b),
            constraints={"c": c, "convex": convex},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))


class NoisyQuadraticDistributionTestCase(testcases.RandomTestCase):
    """Test opda.parametric.NoisyQuadraticDistribution."""

    def test_class_attributes(self):
        # Test C_MIN and C_MAX.

        c_min = parametric.NoisyQuadraticDistribution.C_MIN
        c_max = parametric.NoisyQuadraticDistribution.C_MAX

        # NoisyQuadraticDistribution should always support c = 1 to 5.
        self.assertEqual(c_min, 1)
        self.assertGreaterEqual(c_max, 5)

        # NoisyQuadraticDistribution should support all claimed values of c.
        for c in range(c_min, c_max + 1):
            dist = parametric.NoisyQuadraticDistribution(0., 1., c, 1e-2, False)
            self.assertGreater(dist.pdf(0.5), 0.)
            self.assertGreater(dist.cdf(0.5), 0.)

        # C_MAX should match the maximum value for c mentioned in the docstring.
        self.assertIn(
            f"Values of ``c`` greater than {c_max} are not supported.",
            " ".join(parametric.NoisyQuadraticDistribution.__doc__.split()),
        )

    @pytest.mark.level(2)
    def test_attributes(self):
        n_samples = 1_000_000

        bounds = [(-10., -1.), (-1., 0.), (0., 0.), (0., 1.), (1., 10.)]
        cs = [1, 2, 10]
        os = [1e-6, 1e-3, 1e0, 1e3]
        for a, b in bounds:
            for c in cs:
                for o in os:
                    for convex in [False, True]:
                        dist = parametric.NoisyQuadraticDistribution(
                            a, b, c, o, convex,
                        )
                        ys = dist.sample(n_samples)
                        self.assertAlmostEqual(
                            dist.mean,
                            np.mean(ys),
                            delta=6*np.std(ys)/n_samples**0.5,
                        )
                        self.assertAlmostEqual(
                            dist.variance,
                            np.mean((ys - dist.mean)**2),
                            delta=6*np.std((ys - dist.mean)**2)/n_samples**0.5,
                        )

    def test___eq__(self):
        bounds = [(-10., -1.), (-1., 0.), (0., 0.), (0., 1.), (1., 10.)]
        cs = [1, 2, 10]
        os = [1e-6, 1e-3, 1e0, 1e3]
        for a, b in bounds:
            for c in cs:
                for o in os:
                    for convex in [False, True]:
                        # Test inequality with other objects.
                        self.assertNotEqual(
                            parametric.NoisyQuadraticDistribution(
                                a, b, c, o, convex,
                            ),
                            None,
                        )
                        self.assertNotEqual(
                            parametric.NoisyQuadraticDistribution(
                                a, b, c, o, convex,
                            ),
                            1.,
                        )
                        self.assertNotEqual(
                            parametric.NoisyQuadraticDistribution(
                                a, b, c, o, convex,
                            ),
                            set(),
                        )

                        # Test (in)equality between instances of the same class.
                        #   equality
                        self.assertEqual(
                            parametric.NoisyQuadraticDistribution(
                                a, b, c, o, convex,
                            ),
                            parametric.NoisyQuadraticDistribution(
                                a, b, c, o, convex,
                            ),
                        )
                        #   inequality
                        for a_, _ in bounds:
                            if a_ == a or a_ > b:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a_, b, c, o, convex,
                                ),
                            )
                        for _, b_ in bounds:
                            if b_ == b or b_ < a:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b_, c, o, convex,
                                ),
                            )
                        for c_ in cs:
                            if c_ == c:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c_, o, convex,
                                ),
                            )
                        for o_ in os:
                            if o_ == o:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o_, convex,
                                ),
                            )
                        for convex_ in [False, True]:
                            if convex_ == convex:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex_,
                                ),
                            )

                        # Test (in)equality between instances of
                        # different classes.
                        class NoisyQuadraticDistributionSubclass(
                                parametric.NoisyQuadraticDistribution,
                        ):
                            pass
                        #   equality
                        self.assertEqual(
                            parametric.NoisyQuadraticDistribution(
                                a, b, c, o, convex,
                            ),
                            NoisyQuadraticDistributionSubclass(
                                a, b, c, o, convex,
                            ),
                        )
                        self.assertEqual(
                            NoisyQuadraticDistributionSubclass(
                                a, b, c, o, convex,
                            ),
                            parametric.NoisyQuadraticDistribution(
                                a, b, c, o, convex,
                            ),
                        )
                        #   inequality
                        for a_, _ in bounds:
                            if a_ == a or a_ > b:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                NoisyQuadraticDistributionSubclass(
                                    a_, b, c, o, convex,
                                ),
                            )
                            self.assertNotEqual(
                                NoisyQuadraticDistributionSubclass(
                                    a_, b, c, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                            )
                        for _, b_ in bounds:
                            if b_ == b or b_ < a:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                NoisyQuadraticDistributionSubclass(
                                    a, b_, c, o, convex,
                                ),
                            )
                            self.assertNotEqual(
                                NoisyQuadraticDistributionSubclass(
                                    a, b_, c, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                            )
                        for c_ in cs:
                            if c_ == c:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                NoisyQuadraticDistributionSubclass(
                                    a, b, c_, o, convex,
                                ),
                            )
                            self.assertNotEqual(
                                NoisyQuadraticDistributionSubclass(
                                    a, b, c_, o, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                            )
                        for o_ in os:
                            if o_ == o:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                NoisyQuadraticDistributionSubclass(
                                    a, b, c, o_, convex,
                                ),
                            )
                            self.assertNotEqual(
                                NoisyQuadraticDistributionSubclass(
                                    a, b, c, o_, convex,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                            )
                        for convex_ in [False, True]:
                            if convex_ == convex:
                                continue
                            self.assertNotEqual(
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                                NoisyQuadraticDistributionSubclass(
                                    a, b, c, o, convex_,
                                ),
                            )
                            self.assertNotEqual(
                                NoisyQuadraticDistributionSubclass(
                                    a, b, c, o, convex_,
                                ),
                                parametric.NoisyQuadraticDistribution(
                                    a, b, c, o, convex,
                                ),
                            )

    def test___str__(self):
        self.assertEqual(
            str(parametric.NoisyQuadraticDistribution(
                0., 1., 1, 1., convex=False,
            )),
            f"NoisyQuadraticDistribution("
                f"a={np.array(0.)[()]!s},"
                f" b={np.array(1.)[()]!s},"
                f" c={np.array(1)[()]!s},"
                f" o={np.array(1.)[()]!s},"
                f" convex={False!s}"
            f")",
        )

    def test___repr__(self):
        self.assertEqual(
            repr(parametric.NoisyQuadraticDistribution(
                0., 1., 1, 1., convex=False,
            )),
            f"NoisyQuadraticDistribution("
                f"a={np.array(0.)[()]!r},"
                f" b={np.array(1.)[()]!r},"
                f" c={np.array(1)[()]!r},"
                f" o={np.array(1.)[()]!r},"
                f" convex={False!r}"
            f")",
        )

    def test_sample(self):
        # Test sample when a = b and o = 0.
        a, b = 0., 0.
        for c in [1, 10]:
            for convex in [False, True]:
                dist = parametric.NoisyQuadraticDistribution(
                    a, b, c, 0., convex=convex,
                )
                # without expicit value for size
                y = dist.sample()
                self.assertTrue(np.isscalar(y))
                self.assertEqual(y, 0.)
                # scalar
                y = dist.sample(None)
                self.assertTrue(np.isscalar(y))
                self.assertEqual(y, 0.)
                # 1D array
                ys = dist.sample(100)
                self.assertTrue(np.array_equal(ys, np.zeros(100)))
                # 2D array
                ys = dist.sample((10, 10))
                self.assertTrue(np.array_equal(ys, np.zeros((10, 10))))

        # Test sample when a != b.
        a, b = 0., 1.
        for c in [1, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    # without explicit value for size
                    y = dist.sample()
                    self.assertTrue(np.isscalar(y))
                    self.assertLess(a - 6*o, y)
                    self.assertGreater(b + 6*o, y)
                    # scalar
                    y = dist.sample(None)
                    self.assertTrue(np.isscalar(y))
                    self.assertLess(a - 6*o, y)
                    self.assertGreater(b + 6*o, y)
                    # 1D array
                    ys = dist.sample(100)
                    self.assertEqual(ys.shape, (100,))
                    self.assertLess(a - 6*o, np.min(ys))
                    self.assertGreater(b + 6*o, np.max(ys))
                    # 2D array
                    ys = dist.sample((10, 10))
                    self.assertEqual(ys.shape, (10, 10))
                    self.assertLess(a - 6*o, np.min(ys))
                    self.assertGreater(b + 6*o, np.max(ys))
        # Test c = 2 and o = 0, the samples should be uniformly distributed.
        a, b, c, o = 0., 1., 2, 0.
        ys = parametric.NoisyQuadraticDistribution(a, b, c, o).sample(2_500)
        self.assertLess(a, np.min(ys))
        self.assertGreater(b, np.max(ys))
        self.assertLess(abs(np.mean(ys < 0.5) - 0.5), 0.05)
        # Test a = b and o > 0, the samples should be normally distributed.
        a, b, c, o = 0., 0., 1, 1.
        ys = parametric.NoisyQuadraticDistribution(a, b, c, o).sample(2_500)
        self.assertLess(a-7*o, np.min(ys))
        self.assertGreater(b+7*o, np.max(ys))
        self.assertLess(abs(np.mean(utils.normal_cdf(ys) < 0.5) - 0.5), 0.05)

    def test_pdf(self):
        # Test when a = b and o = 0.

        # When a = b and o = 0, the distribution is a point mass.
        a, b, c, o = 0., 0., 2, 0.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            self.assertTrue(np.isscalar(dist.pdf(a)))
            self.assertEqual(dist.pdf(a - 1.), 0.)
            self.assertEqual(dist.pdf(a), np.inf)
            self.assertEqual(dist.pdf(a + 1.), 0.)
            # broadcasting
            #   1D array
            self.assertEqual(
                dist.pdf([a - 1., a, a + 1.]).tolist(),
                [0., np.inf, 0.],
            )
            # 2D array
            self.assertEqual(
                dist.pdf([[a - 1.], [a], [a + 1.]]).tolist(),
                [[0.], [np.inf], [0.]],
            )
            # 3D array
            self.assertEqual(
                dist.pdf([[[a - 1.]], [[a]], [[a + 1.]]]).tolist(),
                [[[0.]], [[np.inf]], [[0.]]],
            )

        # Test when a = b and o > 0.

        a, b, c, o = 0., 0., 2, 1.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(
                    dist.pdf(
                        (n / 5.) * (a - 6*o)
                        + ((5. - n) / 5.) * (b + 6*o),
                    ),
                ))
                # When a = b and o > 0, the distribution is normal.
                self.assertEqual(
                    dist.pdf(
                        (n / 5.) * (a - 6*o)
                        + ((5. - n) / 5.) * (b + 6*o),
                    ),
                    utils.normal_pdf(
                        (n / 5.) * (a - 6*o)
                        + ((5. - n) / 5.) * (b + 6*o),
                    ),
                )
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.pdf(a - 6*o + us * 12*o).tolist(),
                    utils.normal_pdf(a - 6*o + us * 12*o).tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.pdf(a - 6*o + us * 12*o).tolist(),
                    utils.normal_pdf(a - 6*o + us * 12*o).tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.pdf(a - 6*o + us * 12*o).tolist(),
                    utils.normal_pdf(a - 6*o + us * 12*o).tolist(),
                )

        # Test when a != b and o = 0.

        # Test inside of the distribution's support.
        a, b, c, o = 0., 1., 2, 0.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.pdf(a + (n / 5.) * (b - a))))
                # When c = 2 and o = 0, the distribution is uniform.
                self.assertAlmostEqual(
                    dist.pdf(a + (n / 5.) * (b - a)),
                    1. / (b - a),
                )
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.full_like(us, 1. / (b - a)).tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.full_like(us, 1. / (b - a)).tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.full_like(us, 1. / (b - a)).tolist(),
                )

        # Test outside of the distribution's support.
        for a, b, c, o in [(0., 1., 1, 0.), (0., 1., 2, 0.)]:
            for convex in [False, True]:
                dist = parametric.NoisyQuadraticDistribution(
                    a, b, c, o, convex=convex,
                )
                self.assertEqual(dist.pdf(a - 1e-10), 0.)
                self.assertEqual(dist.pdf(a - 10), 0.)
                self.assertEqual(dist.pdf(b + 1e-10), 0.)
                self.assertEqual(dist.pdf(b + 10), 0.)

        # Test when a != b and o > 0.

        a, b = 0., 1.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    # scalar
                    for n in range(6):
                        self.assertTrue(np.isscalar(
                            dist.pdf(
                                (n / 5.) * (a - 6*o)
                                + ((5. - n) / 5.) * (b + 6*o),
                            ),
                        ))
                        self.assertTrue(np.all(
                            dist.pdf(
                                (n / 5.) * (a - 6*o)
                                + ((5. - n) / 5.) * (b + 6*o),
                            ) >= 0.,
                        ))
                    # broadcasting
                    for _ in range(7):
                        # 1D array
                        us = self.generator.uniform(0, 1, size=5)
                        self.assertEqual(
                            dist.pdf(a - 6*o + us * 12*o).shape,
                            us.shape,
                        )
                        self.assertTrue(np.all(
                            dist.pdf(a - 6*o + us * 12*o) >= 0.,
                        ))
                        # 2D array
                        us = self.generator.uniform(0, 1, size=(5, 3))
                        self.assertEqual(
                            dist.pdf(a - 6*o + us * 12*o).shape,
                            us.shape,
                        )
                        self.assertTrue(np.all(
                            dist.pdf(a - 6*o + us * 12*o) >= 0.,
                        ))
                        # 3D array
                        us = self.generator.uniform(0, 1, size=(5, 3, 2))
                        self.assertEqual(
                            dist.pdf(a - 6*o + us * 12*o).shape,
                            us.shape,
                        )
                        self.assertTrue(np.all(
                            dist.pdf(a - 6*o + us * 12*o) >= 0.,
                        ))

    def test_cdf(self):
        # Test when a = b and o = 0.

        # When a = b and o = 0, the distribution is a point mass.
        a, b, c, o = 0., 0., 2, 0.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            self.assertTrue(np.isscalar(dist.cdf(a)))
            self.assertEqual(dist.cdf(a - 1.), 0.)
            self.assertEqual(dist.cdf(a), 1.)
            self.assertEqual(dist.cdf(a + 1.), 1.)
            # broadcasting
            #   1D array
            self.assertEqual(
                dist.cdf([a - 1., a, a + 1.]).tolist(),
                [0., 1., 1.],
            )
            # 2D array
            self.assertEqual(
                dist.cdf([[a - 1.], [a], [a + 1.]]).tolist(),
                [[0.], [1.], [1.]],
            )
            # 3D array
            self.assertEqual(
                dist.cdf([[[a - 1.]], [[a]], [[a + 1.]]]).tolist(),
                [[[0.]], [[1.]], [[1.]]],
            )

        # Test when a = b and o > 0.

        a, b, c, o = 0., 0., 2, 1.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(
                    dist.cdf(
                        (n / 5.) * (a - 6*o)
                        + ((5. - n) / 5.) * (b + 6*o),
                    ),
                ))
                # When a = b, the distribution is normal.
                self.assertEqual(
                    dist.cdf(
                        (n / 5.) * (a - 6*o)
                        + ((5. - n) / 5.) * (b + 6*o),
                    ),
                    utils.normal_cdf(
                        (n / 5.) * (a - 6*o)
                        + ((5. - n) / 5.) * (b + 6*o),
                    ),
                )
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.cdf(a - 6*o + us * 12*o).tolist(),
                    utils.normal_cdf(a - 6*o + us * 12*o).tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.cdf(a - 6*o + us * 12*o).tolist(),
                    utils.normal_cdf(a - 6*o + us * 12*o).tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.cdf(a - 6*o + us * 12*o).tolist(),
                    utils.normal_cdf(a - 6*o + us * 12*o).tolist(),
                )

        # Test when a != b and o = 0.

        # Test inside of the distribution's support.
        a, b, c, o = 0., 1., 2, 0.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.cdf(a + (n / 5.) * (b - a))))
                # When c = 2 and o = 0, the distribution is uniform.
                self.assertAlmostEqual(
                    dist.cdf(a + (n / 5.) * (b - a)),
                    n / 5.,
                )
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )

        # Test outside of the distribution's support.
        for a, b, c, o in [(0., 1., 1, 0.), (0., 1., 2, 0.)]:
            for convex in [False, True]:
                dist = parametric.NoisyQuadraticDistribution(
                    a, b, c, o, convex=convex,
                )
                self.assertEqual(dist.cdf(a - 1e-10), 0.)
                self.assertEqual(dist.cdf(a - 10), 0.)
                self.assertEqual(dist.cdf(b + 1e-10), 1.)
                self.assertEqual(dist.cdf(b + 10), 1.)

        # Test when a != b and o > 0.

        a, b = 0., 1.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    # scalar
                    for n in range(6):
                        self.assertTrue(np.isscalar(
                            dist.cdf(
                                (n / 5.) * (a - 6*o)
                                + ((5. - n) / 5.) * (b + 6*o),
                            ),
                        ))
                        self.assertTrue(np.all(
                            dist.cdf(
                                (n / 5.) * (a - 6*o)
                                + ((5. - n) / 5.) * (b + 6*o),
                            ) >= 0.,
                        ))
                        self.assertTrue(np.all(
                            dist.cdf(
                                (n / 5.) * (a - 6*o)
                                + ((5. - n) / 5.) * (b + 6*o),
                            ) <= 1.,
                        ))
                    # broadcasting
                    for _ in range(7):
                        # 1D array
                        us = self.generator.uniform(0, 1, size=5)
                        self.assertEqual(
                            dist.cdf(a - 6*o + us * 12*o).shape,
                            us.shape,
                        )
                        self.assertTrue(np.all(
                            dist.cdf(a - 6*o + us * 12*o) >= 0.,
                        ))
                        self.assertTrue(np.all(
                            dist.cdf(a - 6*o + us * 12*o) <= 1.,
                        ))
                        # 2D array
                        us = self.generator.uniform(0, 1, size=(5, 3))
                        self.assertEqual(
                            dist.cdf(a - 6*o + us * 12*o).shape,
                            us.shape,
                        )
                        self.assertTrue(np.all(
                            dist.cdf(a - 6*o + us * 12*o) >= 0.,
                        ))
                        self.assertTrue(np.all(
                            dist.cdf(a - 6*o + us * 12*o) <= 1.,
                        ))
                        # 3D array
                        us = self.generator.uniform(0, 1, size=(5, 3, 2))
                        self.assertEqual(
                            dist.cdf(a - 6*o + us * 12*o).shape,
                            us.shape,
                        )
                        self.assertTrue(np.all(
                            dist.cdf(a - 6*o + us * 12*o) >= 0.,
                        ))
                        self.assertTrue(np.all(
                            dist.cdf(a - 6*o + us * 12*o) <= 1.,
                        ))

    @pytest.mark.level(3)
    def test_ppf(self):
        # Test when a = b and o = 0.

        # When a = b, the distribution is a point mass.
        a, b, c, o = 0., 0., 2, 0.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.ppf(n / 5.)))
                self.assertEqual(dist.ppf(n / 5.), a)
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(dist.ppf(us).tolist(), [a] * 5)
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(dist.ppf(us).tolist(), [[a] * 3] * 5)
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(dist.ppf(us).tolist(), [[[a] * 2] * 3] * 5)

        # Test when a = b and o > 0.

        a, b, c, o = 0., 0., 2, 1.
        normal = stats.norm(0., o)
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.ppf(n / 5.)))
                # When a = b, the distribution is normal.
                self.assertTrue(np.allclose(
                    dist.ppf(n / 5.),
                    normal.ppf(n / 5.),
                ))
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertTrue(np.allclose(
                    dist.ppf(us).tolist(),
                    normal.ppf(us).tolist(),
                ))
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertTrue(np.allclose(
                    dist.ppf(us).tolist(),
                    normal.ppf(us).tolist(),
                ))
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertTrue(np.allclose(
                    dist.ppf(us).tolist(),
                    normal.ppf(us).tolist(),
                ))

        # Test when a != b and o = 0.

        a, b, c, o = 0., 1., 2, 0.
        for convex in [False, True]:
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.ppf(n / 5.)))
                # When c = 2 and o = 0, the distribution is uniform.
                self.assertAlmostEqual(
                    dist.ppf(n / 5.),
                    a + (n / 5.) * (b - a),
                )
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertTrue(np.allclose(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                ))
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertTrue(np.allclose(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                ))
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertTrue(np.allclose(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                ))

        # Test when a != b and o > 0.

        a, b = 0., 1.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    # scalar
                    for n in range(6):
                        self.assertTrue(np.isscalar(dist.ppf(n / 5.)))
                        self.assertTrue(np.all(~np.isnan(
                            dist.ppf(n / 5.),
                        )))
                    # broadcasting
                    for _ in range(7):
                        # 1D array
                        us = self.generator.uniform(0, 1, size=5)
                        self.assertEqual(dist.ppf(us).shape, us.shape)
                        self.assertTrue(np.all(~np.isnan(dist.ppf(us))))
                        # 2D array
                        us = self.generator.uniform(0, 1, size=(5, 3))
                        self.assertEqual(dist.ppf(us).shape, us.shape)
                        self.assertTrue(np.all(~np.isnan(dist.ppf(us))))
                        # 3D array
                        us = self.generator.uniform(0, 1, size=(5, 3, 2))
                        self.assertEqual(dist.ppf(us).shape, us.shape)
                        self.assertTrue(np.all(~np.isnan(dist.ppf(us))))

    @pytest.mark.level(3)
    def test_quantile_tuning_curve(self):
        n_trials = 2_000
        q = 0.5
        a, b = 0., 1.
        for c in [1, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    for minimize in [None, False, True]:
                        # NOTE: When minimize is None, default to convex.
                        expect_minimize = (
                            minimize
                            if minimize is not None else
                            convex
                        )

                        dist = parametric.NoisyQuadraticDistribution(
                            a,
                            b,
                            c,
                            o,
                            convex=convex,
                        )
                        yss = dist.sample((n_trials, 5))
                        # Use the binomial confidence interval for the quantile.
                        idx_lo, idx_pt, idx_hi = stats.binom(n_trials, q).ppf(
                            [1e-6 / 2, 0.5, 1 - 1e-6 / 2],
                        ).astype(int)
                        curve_lo, curve, curve_hi = np.sort(
                            np.minimum.accumulate(yss, axis=1)
                            if expect_minimize else
                            np.maximum.accumulate(yss, axis=1),
                            axis=0,
                        )[(
                            idx_lo,  # lower 1 - 1e-6 confidence bound
                            idx_pt,  # point estimate
                            idx_hi,  # upper 1 - 1e-6 confidence bound
                        ), :]
                        atol = np.max(curve_hi - curve_lo) / 2

                        # Test when n is integral.
                        #   scalar
                        for n in range(1, 6):
                            self.assertTrue(np.isscalar(
                                dist.quantile_tuning_curve(
                                    n,
                                    q=q,
                                    minimize=minimize,
                                ),
                            ))
                            self.assertAlmostEqual(
                                dist.quantile_tuning_curve(
                                    n,
                                    q=q,
                                    minimize=minimize,
                                ),
                                curve[n-1],
                                delta=atol,
                            )
                            self.assertTrue(np.allclose(
                                dist.quantile_tuning_curve(
                                    [n],
                                    q=q,
                                    minimize=minimize,
                                ),
                                [
                                    dist.quantile_tuning_curve(
                                        n,
                                        q=q,
                                        minimize=minimize,
                                    ),
                                ],
                            ))
                        #   1D array
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [1, 2, 3, 4, 5],
                                q=q,
                                minimize=minimize,
                            ),
                            curve,
                            atol=atol,
                        ))
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [3, 1, 5],
                                q=q,
                                minimize=minimize,
                            ),
                            [curve[2], curve[0], curve[4]],
                            atol=atol,
                        ))
                        #   2D array
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [
                                    [1, 2, 3],
                                    [3, 1, 5],
                                ],
                                q=q,
                                minimize=minimize,
                            ),
                            [
                                [curve[0], curve[1], curve[2]],
                                [curve[2], curve[0], curve[4]],
                            ],
                            atol=atol,
                        ))

                        # Test when n is non-integral.
                        #   scalar
                        for n in range(1, 6):
                            self.assertTrue(np.isscalar(
                                dist.quantile_tuning_curve(
                                    n/10.,
                                    q=1 - (1 - q)**(1/10)
                                      if expect_minimize else
                                      q**(1/10),
                                    minimize=minimize,
                                ),
                            ))
                            self.assertAlmostEqual(
                                dist.quantile_tuning_curve(
                                    n/10.,
                                    q=1 - (1 - q)**(1/10)
                                      if expect_minimize else
                                      q**(1/10),
                                    minimize=minimize,
                                ),
                                curve[n-1],
                                delta=atol,
                            )
                            self.assertTrue(np.allclose(
                                dist.quantile_tuning_curve(
                                    [n/10.],
                                    q=1 - (1 - q)**(1/10)
                                      if expect_minimize else
                                      q**(1/10),
                                    minimize=minimize,
                                ),
                                [
                                    dist.quantile_tuning_curve(
                                        n/10.,
                                        q=1 - (1 - q)**(1/10)
                                          if expect_minimize else
                                          q**(1/10),
                                        minimize=minimize,
                                    ),
                                ],
                            ))
                        #   1D array
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [0.1, 0.2, 0.3, 0.4, 0.5],
                                q=1 - (1 - q)**(1/10)
                                  if expect_minimize else
                                  q**(1/10),
                                minimize=minimize,
                            ),
                            curve,
                            atol=atol,
                        ))
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [0.3, 0.1, 0.5],
                                q=1 - (1 - q)**(1/10)
                                  if expect_minimize else
                                  q**(1/10),
                                minimize=minimize,
                            ),
                            [curve[2], curve[0], curve[4]],
                            atol=atol,
                        ))
                        #   2D array
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [
                                    [0.1, 0.2, 0.3],
                                    [0.3, 0.1, 0.5],
                                ],
                                q=1 - (1 - q)**(1/10)
                                  if expect_minimize else
                                  q**(1/10),
                                minimize=minimize,
                            ),
                            [
                                [curve[0], curve[1], curve[2]],
                                [curve[2], curve[0], curve[4]],
                            ],
                            atol=atol,
                        ))

                        # Test ns <= 0.
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                0,
                                q=q,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                -1,
                                q=q,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [0],
                                q=q,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [-2],
                                q=q,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [0, 1],
                                q=q,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [-2, 1],
                                q=q,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [[0], [1]],
                                q=q,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [[-2], [1]],
                                q=q,
                                minimize=minimize,
                            )

    @pytest.mark.level(3)
    def test_average_tuning_curve(self):
        # NOTE: The average tuning curve is very expensive to
        # compute. Thus, to keep this test reasonably fast, only try c
        # in [2] and o in [1e-3]. For more thorough coverage, you can
        # change these to c in [1, 10] and o in [1e-6, 1e-3, 1e0, 1e3]
        # as in other similar tests.
        a, b = 0., 1.
        for c in [2]:
            for o in [1e-3]:
                for convex in [False, True]:
                    for minimize in [None, False, True]:
                        # NOTE: When minimize is None, default to convex.
                        expect_minimize = (
                            minimize
                            if minimize is not None else
                            convex
                        )

                        dist = parametric.NoisyQuadraticDistribution(
                            a,
                            b,
                            c,
                            o,
                            convex=convex,
                        )
                        yss = dist.sample((2_000, 5))
                        curves = (
                            np.minimum.accumulate(yss, axis=1)
                            if expect_minimize else
                            np.maximum.accumulate(yss, axis=1)
                        )
                        curve = np.mean(curves, axis=0)
                        atol = 6 * np.max(np.std(curves, axis=0) / 2_000**0.5)

                        # Test when n is integral.
                        #   scalar
                        for n in range(1, 6):
                            self.assertTrue(np.isscalar(
                                dist.average_tuning_curve(
                                    n,
                                    minimize=minimize,
                                ),
                            ))
                            self.assertAlmostEqual(
                                dist.average_tuning_curve(
                                    n,
                                    minimize=minimize,
                                ),
                                curve[n-1],
                                delta=atol,
                            )
                            self.assertTrue(np.allclose(
                                dist.average_tuning_curve(
                                    [n],
                                    minimize=minimize,
                                ),
                                [
                                    dist.average_tuning_curve(
                                        n,
                                        minimize=minimize,
                                    ),
                                ],
                            ))
                        #   1D array
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [1, 2, 3, 4, 5],
                                minimize=minimize,
                            ),
                            curve,
                            atol=atol,
                        ))
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [3, 1, 5],
                                minimize=minimize,
                            ),
                            [curve[2], curve[0], curve[4]],
                            atol=atol,
                        ))
                        #   2D array
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [
                                    [1, 2, 3],
                                    [3, 1, 5],
                                ],
                                minimize=minimize,
                            ),
                            [
                                [curve[0], curve[1], curve[2]],
                                [curve[2], curve[0], curve[4]],
                            ],
                            atol=atol,
                        ))

                        # Test when n is non-integral.
                        #   scalar
                        for n in range(1, 6):
                            self.assertTrue(np.isscalar(
                                dist.average_tuning_curve(
                                    n + (0.5 if expect_minimize else -0.5),
                                    minimize=minimize,
                                ),
                            ))
                            self.assertLess(
                                dist.average_tuning_curve(
                                    n + (0.5 if expect_minimize else -0.5),
                                    minimize=minimize,
                                ),
                                dist.average_tuning_curve(
                                    n,
                                    minimize=minimize,
                                ),
                            )
                            self.assertTrue(np.allclose(
                                dist.average_tuning_curve(
                                    [n - 0.5],
                                    minimize=minimize,
                                ),
                                [
                                    dist.average_tuning_curve(
                                        n - 0.5,
                                        minimize=minimize,
                                    ),
                                ],
                            ))
                        #   1D array
                        self.assertTrue(np.all(
                            dist.average_tuning_curve(
                                np.arange(1, 6)
                                  + (0.5 if expect_minimize else -0.5),
                                minimize=minimize,
                            )
                            < dist.average_tuning_curve(
                                np.arange(1, 6),
                                minimize=minimize,
                            ),
                        ))
                        #   2D array
                        self.assertTrue(np.all(
                            dist.average_tuning_curve(
                                np.arange(1, 11).reshape(5, 2)
                                  + (0.5 if expect_minimize else -0.5),
                                minimize=minimize,
                            ) < dist.average_tuning_curve(
                                np.arange(1, 11).reshape(5, 2),
                                minimize=minimize,
                            ),
                        ))

                        # Test ns <= 0.
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                0,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                -1,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                [0],
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                [-2],
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                [0, 1],
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                [-2, 1],
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                [[0], [1]],
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.average_tuning_curve(
                                [[-2], [1]],
                                minimize=minimize,
                            )

    @pytest.mark.level(4)
    def test_fit(self):
        n_samples = 2_048

        for a, b, o in [
                # Test when a = b and o = 0.
                (    0.,    0.,    0.),
                # Test when a = b and o > 0.
                (    1.,    1.,    1.),
                # Test when a != b.
                (    0.,    1.,  1e-7),
                (   -1.,    1.,  2e-3),
                ( 1e-50, 2e-50, 1e-51),
                (  1e50,  2e50,  1e51),
        ]:
            # NOTE: When c = 2, convex is unidentifiable because the
            # distribution is the same whether convex is True or
            # False. Thus, we test c = 1 and 3.
            for c in [1, 3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex,
                    )
                    ys = dist.sample(n_samples)

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore"
                            if a == b or o > 1e-2 * (b - a) else
                            "default",
                            message=r"Parameters might be unidentifiable.",
                            category=RuntimeWarning,
                        )
                        dist_hat = parametric.NoisyQuadraticDistribution.fit(ys)

                    # Check the fitted distribution approximates the true CDF.
                    grid = np.linspace(a - 6 * o, b + 6 * o, num=128)
                    self.assertAlmostEqual(
                        np.max(np.abs(dist.cdf(grid) - dist_hat.cdf(grid))), 0.,
                        delta=2**0.5 * utils.dkw_epsilon(n_samples, 1 - 1e-6),
                    )
                    # Check the parameters are approximately correct.
                    self.assertAlmostEqual(
                        dist_hat.a, a,
                        delta=5e-2 * (b - a) + 4 * o,
                    )
                    self.assertAlmostEqual(
                        dist_hat.b, b,
                        delta=5e-2 * (b - a) + 4 * o,
                    )
                    self.assertAlmostEqual(
                        dist_hat.mean, dist.mean,
                        delta=5e-2 * (b - a) + 1e-1 * o,
                    )
                    if o >= 1e-2 * (b - a):
                        # o is hard to estimate if it's small relative to b - a.
                        self.assertGreaterEqual(dist_hat.o, o / 5.)
                        self.assertLessEqual(dist_hat.o, 5. * o)
                    # NOTE: When a = b, the distribution is a point mass or a
                    # normal for any c or convex, thus they are unidentifiable.
                    if a != b and o <= 1e-2 * (b - a):
                        # c and convex are hard to identify when o is
                        # large relative to b - a.
                        self.assertEqual(dist_hat.c, c)
                        self.assertEqual(dist_hat.convex, convex)

    def test_sample_defaults_to_global_random_number_generator(self):
        # sample should be deterministic if global seed is set.
        dist = parametric.NoisyQuadraticDistribution(0., 1., 2, 1.)
        #   Before setting the seed, two samples should be unequal.
        self.assertNotEqual(dist.sample(16).tolist(), dist.sample(16).tolist())
        #   After setting the seed, two samples should be unequal.
        opda.random.set_seed(0)
        self.assertNotEqual(dist.sample(16).tolist(), dist.sample(16).tolist())
        #   Resetting the seed should produce the same sample.
        opda.random.set_seed(0)
        first_sample = dist.sample(16)
        opda.random.set_seed(0)
        second_sample = dist.sample(16)
        self.assertEqual(first_sample.tolist(), second_sample.tolist())

    def test_sample_is_deterministic_given_generator_argument(self):
        dist = parametric.NoisyQuadraticDistribution(0., 1., 2, 1.)
        # Reusing the same generator, two samples should be unequal.
        generator = np.random.default_rng(0)
        self.assertNotEqual(
            dist.sample(16, generator=generator).tolist(),
            dist.sample(16, generator=generator).tolist(),
        )
        # Using generators in the same state should produce the same sample.
        self.assertEqual(
            dist.sample(16, generator=np.random.default_rng(0)).tolist(),
            dist.sample(16, generator=np.random.default_rng(0)).tolist(),
        )

    def test_pdf_on_boundary_of_support(self):
        # Test when a = b and o > 0.
        a, b = 0., 0.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    self.assertEqual(dist.pdf(-np.inf), 0.)
                    self.assertEqual(dist.pdf(np.inf), 0.)

        # Test when a != b and o = 0.
        for convex in [False, True]:
            a, b, c, o = 0., 1., 1, 0.
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            self.assertEqual(dist.pdf(a), np.inf if convex else 0.5)
            self.assertEqual(dist.pdf(b), 0.5 if convex else np.inf)

            a, b, c, o = 0., 1., 2, 0.
            dist = parametric.NoisyQuadraticDistribution(
                a, b, c, o, convex=convex,
            )
            self.assertEqual(dist.pdf(a), 1.)
            self.assertEqual(dist.pdf(b), 1.)

        # Test when a != b and o > 0.
        a, b = 0., 1.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    self.assertEqual(dist.pdf(-np.inf), 0.)
                    self.assertEqual(dist.pdf(np.inf), 0.)

    def test_pdf_is_consistent_across_scales(self):
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    ys = np.linspace(0. - 6 * o, 1. + 6 * o, num=100)

                    ps = parametric.NoisyQuadraticDistribution(
                        0., 1., c, o, convex=convex,
                    ).pdf(ys)
                    for a, b in [(-1., 1.), (1., 10.)]:
                        dist = parametric.NoisyQuadraticDistribution(
                            a, b, c, o * (b - a), convex=convex,
                        )
                        self.assertTrue(np.allclose(
                            ps / (b - a),
                            dist.pdf(a + (b - a) * ys),
                            atol=1e-5,
                        ))

    def test_pdf_matches_numerical_derivative_of_cdf(self):
        # Test when a = b and o > 0.
        a, b = 0, 0
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )

                    ys = np.linspace(a - 3 * o, b + 3 * o, num=100)
                    dy = 1e-9
                    self.assertTrue(np.allclose(
                        dist.pdf(ys),
                        (dist.cdf(ys + dy) - dist.cdf(ys - dy)) / (2 * dy),
                        atol=1e-7,
                    ))

        # Test when a != b and o = 0.
        for a, b in [(-1., 1.), (0., 1.), (1., 10.)]:
            for c in [1, 2, 10]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, 0., convex=convex,
                    )

                    # Omit a and b from the numerical derivatives since
                    # they're on the boundary.
                    ys = np.linspace(a, b, num=102)[1:-1]
                    dy = 1e-7
                    self.assertTrue(np.allclose(
                        dist.pdf(ys),
                        (dist.cdf(ys + dy) - dist.cdf(ys - dy)) / (2 * dy),
                    ))

        # Test when a != b and o > 0.
        for a, b in [(-1., 1.), (0., 1.), (1., 10.)]:
            for c in [1, 2, 10]:
                for o in [1e-6, 1e-3, 1e0, 1e3]:
                    for convex in [False, True]:
                        dist = parametric.NoisyQuadraticDistribution(
                            a, b, c, o, convex=convex,
                        )

                        ys = np.linspace(a - 3 * o, b + 3 * o, num=100)
                        ys = ys[
                            (np.abs(ys - a) > 5e-2)
                            & (np.abs(ys - b) > 5e-2)
                        ]
                        # NOTE: Numerical derivatives have high bias near
                        # spikes in the PDF that can occur at a and b.

                        dy = 3e-4 * (b - a)
                        self.assertTrue(np.allclose(
                            dist.pdf(ys),
                            (dist.cdf(ys + dy) - dist.cdf(ys - dy)) / (2 * dy),
                            atol=5e-3,
                        ))

    @pytest.mark.level(2)
    def test_pdf_agrees_with_pdf_computed_via_monte_carlo_integration(self):
        # NOTE: Use Monte Carlo integration instead of numerical
        # integration because Monte Carlo integration is unbiased while
        # numerical integration has a large bias near peaks in the PDF.
        a, b = 0., 1.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )

                    # Mathematically, the PDF of the noisy quadratic
                    # distribution is: E[f(y - Z)], where f is the
                    # normal distribution's PDF and Z is a random
                    # variable with the quadratic distribution.
                    def pdf_monte_carlo_integration(
                            ys, a=a, b=b, c=c, o=o, convex=convex,
                    ):
                        # Compute the PDF via Monte Carlo integration.
                        n_samples = 100_000

                        zs = parametric.QuadraticDistribution(
                            a, b, c, convex=convex,
                        ).sample(n_samples)

                        return np.mean(
                            utils.normal_pdf((ys[..., None] - zs) / o) / o,
                            axis=-1,
                        )

                    ys = np.linspace(a - 9 * o, b + 9 * o, num=100)
                    self.assertTrue(np.allclose(
                        dist.pdf(ys),
                        pdf_monte_carlo_integration(ys),
                        atol=2e-3 / o,
                        # The Monte Carlo integration has limited
                        # precision, becoming less precise as o shrinks.
                    ))

    def test_cdf_on_boundary_of_support(self):
        # Test when a = b and o > 0.
        a, b = 0., 0.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    self.assertEqual(dist.cdf(-np.inf), 0.)
                    self.assertEqual(dist.cdf(np.inf), 1.)

        # Test when a != b and o = 0.
        a, b, o = 0., 1., 0.
        for c in [1, 2, 10]:
            for convex in [False, True]:
                dist = parametric.NoisyQuadraticDistribution(
                    a, b, c, o, convex=convex,
                )
                self.assertEqual(dist.cdf(a), 0.)
                self.assertEqual(dist.cdf(b), 1.)

        # Test when a != b and o > 0.
        a, b = 0., 1.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )
                    self.assertEqual(dist.cdf(-np.inf), 0.)
                    self.assertEqual(dist.cdf(np.inf), 1.)

    def test_cdf_is_consistent_across_scales(self):
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    ys = np.linspace(0. - 6 * o, 1. + 6 * o, num=100)

                    ps = parametric.NoisyQuadraticDistribution(
                        0., 1., c, o, convex=convex,
                    ).cdf(ys)
                    for a, b in [(-1., 1.), (1., 10.)]:
                        dist = parametric.NoisyQuadraticDistribution(
                            a, b, c, o * (b - a), convex=convex,
                        )
                        self.assertTrue(np.allclose(
                            ps,
                            dist.cdf(a + (b - a) * ys),
                            atol=1e-5,
                        ))

    @pytest.mark.level(2)
    def test_cdf_agrees_with_sampling_definition(self):
        for a, b in [(-1., 1.), (0., 1.), (1., 10.)]:
            # NOTE: Keep c low because the rejection sampling below
            # will reject too many samples for large c.
            for c in [1, 2, 3]:
                for o in [1e-6, 1e-3, 1e0, 1e3]:
                    for convex in [False, True]:
                        dist = parametric.NoisyQuadraticDistribution(
                            a, b, c, o, convex=convex,
                        )

                        # Sample from the noisy quadratic distribution
                        # according to its derivation: uniform random
                        # variates passed through a quadratic function
                        # with additive normal noise.
                        ys = np.sum(
                            self.generator.uniform(
                                -1., 1.,
                                size=(150_000, c),
                            )**2,
                            axis=-1,
                        )
                        # Filter for points in the sphere of radius 1, to
                        # avoid distortions from the hypercube's boundary.
                        ys = ys[ys <= 1]
                        # Adjust the data for the distribution's parameters.
                        ys = (
                            a + (b - a) * ys
                            if convex else  # concave
                            b - (b - a) * ys
                        )
                        # Add normal noise.
                        ys += self.generator.normal(0, o, size=ys.shape)

                        # Check the sample comes from the distribution
                        # using the KS test.
                        p_value = stats.kstest(
                            ys,
                            dist.cdf,
                            alternative="two-sided",
                        ).pvalue
                        self.assertGreater(p_value, 1e-6)

    @pytest.mark.level(2)
    def test_cdf_agrees_with_cdf_computed_via_numerical_integration(self):
        a, b = 0., 1.
        for c in [1, 2, 10]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    dist = parametric.NoisyQuadraticDistribution(
                        a, b, c, o, convex=convex,
                    )

                    # Mathematically, the CDF of the noisy quadratic
                    # distribution is: E[F(y - E)], where F is the
                    # quadratic distribution's CDF and E is normally
                    # distributed noise with mean 0 and variance o.
                    def cdf_numerical_integration(
                            ys, a=a, b=b, c=c, o=o, convex=convex,
                    ):
                        # Compute the CDF using the composite
                        # trapezoid rule for numerical integration.
                        noiseless_cdf = parametric.QuadraticDistribution(
                            a, b, c, convex=convex,
                        ).cdf

                        atol = 1e-6
                        lo, hi =  -6 * o, 6 * o
                        h = hi - lo
                        xs = np.array([lo, hi])
                        ps = 0.5 * h * np.sum(
                            utils.normal_pdf(xs / o) / o
                            * noiseless_cdf(ys[..., None] - xs),
                            axis=-1,
                        )
                        for i in range(1, 21):
                            h *= 0.5
                            xs = lo + np.arange(1, 2**i, 2) * h
                            ps_prev, ps = ps, (
                                0.5 * ps + h * np.sum(
                                    utils.normal_pdf(xs / o) / o
                                    * noiseless_cdf(ys[..., None] - xs),
                                    axis=-1,
                                )
                            )
                            err = np.max(np.abs(ps - ps_prev)) / 3
                            if err < atol:
                                break

                        return ps

                    ys = np.linspace(a - 9 * o, b + 9 * o, num=100)
                    self.assertTrue(np.allclose(
                        dist.cdf(ys),
                        cdf_numerical_integration(ys),
                        atol=1e-5,
                    ))

    @pytest.mark.level(1)
    def test_ppf_is_inverse_of_cdf(self):
        # NOTE: The quantile function is always an almost sure left
        # inverse of the CDF. For continuous distributions like the
        # noisy quadratic distribution, the quantile function is also
        # the right inverse of the cumulative distribution function.
        for a, b in [(0., 0.), (0., 1.)]:
            for c in [1, 10]:
                for o in [0., 1e-6, 1e-3, 1e0, 1e3]:
                    for convex in [False, True]:
                        dist = parametric.NoisyQuadraticDistribution(
                            a, b, c, o, convex=convex,
                        )

                        ys = dist.sample(100)
                        qs = dist.cdf(ys)

                        # Remove points where the CDF gets rounded to 0
                        # or 1 since they'll be mapped to the lower or
                        # upper bound on the support (-inf and inf when
                        # o > 0) and thus won't invert.
                        ys = ys[(0. < qs) & (qs < 1.)]
                        qs = qs[(0. < qs) & (qs < 1.)]

                        ys_center = ys[(ys >= a) & (ys <= b)]
                        qs_center = qs[(ys >= a) & (ys <= b)]
                        self.assertTrue(np.allclose(
                            dist.ppf(qs_center),
                            ys_center,
                            atol=1e-5,
                        ))
                        # NOTE: Precision of the ppf is more difficult
                        # in the tails because the CDF is nearly flat.
                        ys_tails = ys[(ys < a) | (ys > b)]
                        qs_tails = qs[(ys < a) | (ys > b)]
                        self.assertTrue(np.allclose(
                            dist.ppf(qs_tails),
                            ys_tails,
                            atol=1e-2,
                        ))

                        if a == b and o == 0.:
                            # When a = b and o = 0, the distribution
                            # is discrete and the quantile function
                            # is not a right inverse of the CDF.
                            continue

                        us = self.generator.uniform(0, 1, size=100)
                        self.assertTrue(np.allclose(
                            dist.cdf(dist.ppf(us)),
                            us,
                            atol=1e-5,
                        ))

    @pytest.mark.level(1)
    def test_ppf_at_extremes(self):
        for a, b in [(0., 0.), (0., 1.)]:
            for c in [1, 10]:
                for o in [0., 1e-6, 1e-3, 1e0, 1e3]:
                    for convex in [False, True]:
                        dist = parametric.NoisyQuadraticDistribution(
                            a, b, c, o, convex=convex,
                        )
                        lo = a if o == 0 else -np.inf
                        hi = b if o == 0 else np.inf
                        self.assertEqual(dist.ppf(0. - 1e-12), lo)
                        self.assertEqual(dist.ppf(0.), lo)
                        self.assertEqual(dist.ppf(1.), hi)
                        self.assertEqual(dist.ppf(1. + 1e-12), hi)

    @pytest.mark.level(1)
    def test_quantile_tuning_curve_with_different_quantiles(self):
        n_trials = 2_000
        a, b, c, o, convex = 0., 1., 1, 1e-2, False
        for q in [0.25, 0.75]:
            for minimize in [None, False, True]:
                # NOTE: When minimize is None, default to convex.
                expect_minimize = (
                    minimize
                    if minimize is not None else
                    convex
                )

                dist = parametric.NoisyQuadraticDistribution(
                    a,
                    b,
                    c,
                    o,
                    convex=convex,
                )
                yss = dist.sample((n_trials, 5))
                # Use the binomial confidence interval for the quantile.
                idx_lo, idx_pt, idx_hi = stats.binom(n_trials, q).ppf(
                    [1e-6 / 2, 0.5, 1 - 1e-6 / 2],
                ).astype(int)
                curve_lo, curve, curve_hi = np.sort(
                    np.minimum.accumulate(yss, axis=1)
                    if expect_minimize else
                    np.maximum.accumulate(yss, axis=1),
                    axis=0,
                )[(
                    idx_lo,  # lower 1 - 1e-6 confidence bound
                    idx_pt,  # point estimate
                    idx_hi,  # upper 1 - 1e-6 confidence bound
                ), :]
                atol = np.max(curve_hi - curve_lo) / 2

                self.assertTrue(np.allclose(
                    dist.quantile_tuning_curve(
                        [1, 2, 3, 4, 5],
                        q=q,
                        minimize=minimize,
                    ),
                    curve,
                    atol=atol,
                ))

    @pytest.mark.level(3)
    def test_quantile_tuning_curve_minimize_is_dual_to_maximize(self):
        for a, b, c in [(0., 1., 1), (-1., 10., 2)]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    ns = np.arange(1, 17)

                    self.assertTrue(np.allclose(
                        parametric
                          .NoisyQuadraticDistribution(
                              a, b, c, o, convex=convex,
                          ).quantile_tuning_curve(ns, minimize=False),
                        -parametric
                          .NoisyQuadraticDistribution(
                              -b, -a, c, o, convex=not convex,
                          ).quantile_tuning_curve(ns, minimize=True),
                        atol=1e-4,
                    ))
                    self.assertTrue(np.allclose(
                        parametric
                          .NoisyQuadraticDistribution(
                              a, b, c, o, convex=convex,
                          ).quantile_tuning_curve(ns, minimize=True),
                        -parametric
                          .NoisyQuadraticDistribution(
                              -b, -a, c, o, convex=not convex,
                          ).quantile_tuning_curve(ns, minimize=False),
                        atol=1e-4,
                    ))

    @pytest.mark.level(3)
    def test_average_tuning_curve_minimize_is_dual_to_maximize(self):
        for a, b, c in [(0., 1., 1), (-1., 10., 2)]:
            for o in [1e-6, 1e-3, 1e0, 1e3]:
                for convex in [False, True]:
                    ns = np.arange(1, 17)

                    self.assertTrue(np.allclose(
                        parametric
                          .NoisyQuadraticDistribution(
                              a, b, c, o, convex=convex,
                          ).average_tuning_curve(ns, minimize=False),
                        -parametric
                          .NoisyQuadraticDistribution(
                              -b, -a, c, o, convex=not convex,
                          ).average_tuning_curve(ns, minimize=True),
                        atol=1e-4 * max(o, 1),
                    ))
                    self.assertTrue(np.allclose(
                        parametric
                          .NoisyQuadraticDistribution(
                              a, b, c, o, convex=convex,
                          ).average_tuning_curve(ns, minimize=True),
                        -parametric
                          .NoisyQuadraticDistribution(
                              -b, -a, c, o, convex=not convex,
                          ).average_tuning_curve(ns, minimize=False),
                        atol=1e-4 * max(o, 1),
                    ))

    @pytest.mark.level(4)
    def test_fit_applies_limits_correctly(self):
        n_samples = 2_048

        c, convex = 1, False
        for a, b, o in [
                # Test when a = b and o = 0.
                (0., 0.,   0.),
                # Test when a != b and o > 0.
                (0., 1., 1e-2),
        ]:
            dist = parametric.NoisyQuadraticDistribution(a, b, c, o, convex)
            ys = dist.sample(n_samples)

            # Test fit with limits.

            for censor_side in ["left", "right", "both"]:
                for censor_trivial in [False, True]:
                    limit_lower = (
                        -np.inf         if censor_side == "right" else
                        a - 0.1         if censor_trivial else
                        dist.ppf(0.1)   if a != b else
                        a + 0.1       # if a == b
                    )
                    limit_upper = (
                        np.inf          if censor_side == "left" else
                        b + 0.1         if censor_trivial else
                        dist.ppf(0.975) if a != b else
                        b - 0.1       # if a == b
                    )

                    if a == b and o == 0. and not censor_trivial:
                        # When a = b and o = 0, a non-trivial censoring
                        # will censor all observations.
                        with self.assertRaises(ValueError):
                            dist_hat =\
                                parametric.NoisyQuadraticDistribution.fit(
                                    ys=ys,
                                    limits=(limit_lower, limit_upper),
                                )
                        continue

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore" if a == b else "default",
                            message=r"Parameters might be unidentifiable.",
                            category=RuntimeWarning,
                        )
                        dist_hat = parametric.NoisyQuadraticDistribution.fit(
                            ys=ys,
                            limits=(limit_lower, limit_upper),
                        )

                    self.assertAlmostEqual(
                        dist_hat.a, a,
                        delta=5e-2 * (b - a) + 4 * o,
                    )
                    self.assertAlmostEqual(
                        dist_hat.b, b,
                        delta=5e-2 * (b - a) + 4 * o,
                    )
                    self.assertGreaterEqual(dist_hat.o, o / 5.)
                    self.assertLessEqual(dist_hat.o, 5. * o)
                    # NOTE: When a = b, the distribution is a point mass or a
                    # normal for any c or convex, thus they are unidentifiable.
                    if a != b:
                        self.assertEqual(dist_hat.c, c)
                        self.assertEqual(dist_hat.convex, convex)

            # Test limits is a left-open interval.

            threshold = a if a == b else a + 0.75 * (b - a)

            if a == b:
                # Since limits is left-open and all observations equal a,
                # they are all censored when the lower limit is a.
                with self.assertRaises(ValueError):
                    dist_hat = parametric.NoisyQuadraticDistribution.fit(
                        ys=ys,
                        limits=(threshold, np.inf),
                    )
                # Since limits is right-closed, no observations are
                # censored when just the upper limit equals b.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Parameters might be unidentifiable.",
                        category=RuntimeWarning,
                    )
                    dist_hat = parametric.NoisyQuadraticDistribution.fit(
                        ys=ys,
                        limits=(-np.inf, threshold),
                    )
                self.assertAlmostEqual(
                    dist_hat.a, a,
                    delta=5e-2 * (b - a) + 4 * o,
                )
                self.assertAlmostEqual(
                    dist_hat.b, b,
                    delta=5e-2 * (b - a) + 4 * o,
                )
            else:  # a != b
                # Since limits is left-open, moving observations below
                # it to the lower endpoint will not effect the fit.
                dist_hat = parametric.NoisyQuadraticDistribution.fit(
                    ys=np.where(ys <= threshold, threshold, ys),
                    limits=(threshold, np.inf),
                )
                self.assertAlmostEqual(
                    dist_hat.a, a,
                    delta=2e-1 * (b - a) + 4 * o,
                )
                self.assertAlmostEqual(
                    dist_hat.b, b,
                    delta=5e-2 * (b - a) + 4 * o,
                )
                self.assertEqual(dist_hat.c, c)
                self.assertGreaterEqual(dist_hat.o, o / 5.)
                self.assertLessEqual(dist_hat.o, 5. * o)
                self.assertEqual(dist_hat.convex, convex)

    @pytest.mark.level(4)
    def test_fit_applies_constraints_correctly(self):
        n_samples = 2_048

        c, convex = 1, False
        for a, b, o in [
                # Test when a = b and o = 0.
                (0., 0.,   0.),
                # Test when a != b and o > 0.
                (0., 1., 1e-2),
        ]:
            dist = parametric.NoisyQuadraticDistribution(a, b, c, o, convex)
            ys = dist.sample(n_samples)

            # Test fit with feasible constraints.

            for constraints in [
                    # Constrain a.
                    {"a": a},
                    {"a": (a, b)},
                    {"a": (-np.inf, a)},
                    # Constrain b.
                    {"b": b},
                    {"b": (a, b)},
                    {"b": (b, np.inf)},
                    # Constrain c.
                    {"c": c},
                    {"c": (0, c)},
                    {"c": (c, 100)},
                    # Constrain o.
                    {"o": o},
                    {"o": (o, np.inf)},
                    {"o": (0., o)},
                    # Constrain convex.
                    {"convex": convex},
                    {"convex": (convex,)},
                    {"convex": (False, True)},
                    # Fix two numeric parameters.
                    {"a": a, "b": b},
                    {"a": a, "c": c},
                    {"a": a, "o": o},
                    {"b": b, "c": c},
                    {"b": b, "o": o},
                    {"c": c, "o": o},
                    # Fix all four numeric parameters.
                    {"a": a, "b": b, "c": c, "o": o},
                    # Fix all parameters.
                    {"a": a, "b": b, "c": c, "o": o, "convex": convex},
            ]:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore" if a == b else "default",
                        message=r"Parameters might be unidentifiable.",
                        category=RuntimeWarning,
                    )
                    warnings.filterwarnings(
                        "ignore" if "c" in constraints else "default",
                        message=r"The constraint for c includes values"
                                r" outside of 1 to 10,",
                        category=RuntimeWarning,
                    )
                    dist_hat = parametric.NoisyQuadraticDistribution.fit(
                        ys=ys,
                        constraints=constraints,
                    )

                # Check the fit recovers the distribution.
                self.assertAlmostEqual(
                    dist_hat.a, a,
                    delta=5e-2 * (b - a) + 4 * o,
                )
                self.assertAlmostEqual(
                    dist_hat.b, b,
                    delta=5e-2 * (b - a) + 4 * o,
                )
                self.assertGreaterEqual(dist_hat.o, o / 5.)
                self.assertLessEqual(dist_hat.o, 5. * o)
                # NOTE: When a = b, the distribution is a point mass or a
                # normal for any c or convex, thus they are unidentifiable.
                if a != b:
                    self.assertEqual(dist_hat.c, c)
                    self.assertEqual(dist_hat.convex, convex)

                # Check the constraint was obeyed.
                for parameter, value in constraints.items():
                    estimate = getattr(dist_hat, parameter)
                    if np.isscalar(value):
                        self.assertEqual(estimate, value)
                    elif parameter in {"a", "b", "c", "o"}:
                        lo, hi = value
                        self.assertGreaterEqual(estimate, lo)
                        self.assertLessEqual(estimate, hi)
                    elif parameter in {"convex"}:
                        self.assertIn(estimate, value)
                    else:
                        raise ValueError(
                            f"Unrecognized parameter: {parameter}",
                        )

            # Test fit with infeasible constraints.

            for censor in [False, True]:
                if a == b and o == 0. and censor:
                    # When a = b and o = 0, all observations are equal
                    # thus it is impossible to censor some but not all
                    # of them. The case of censoring all observations
                    # is already covered in other tests.
                    continue

                limits = (
                    (a + 0.1 * (b - a), b - 0.1 * (b - a))
                    if censor else
                    (-np.inf, np.inf)
                )
                for constraints, exception in [
                        # Constrain a to be greater than observed ys.
                        (
                            {"a": (max(a, limits[0]) + 0.1, np.inf), "o": 0.},
                            ValueError,
                        ),
                        (
                            {"a": (1e8, 1e9), "o": 0.},
                            ValueError,
                        ),
                        # Constrain b to be less than observed ys.
                        (
                            {"b": (-np.inf, min(b, limits[1]) - 0.1), "o": 0.},
                            ValueError,
                        ),
                        (
                            {"b": (-1e9, -1e8), "o": 0.},
                            ValueError,
                        ),
                        # Constrain c to unsupported values.
                        (
                            {"c": (-10, 0)},
                            ValueError,
                        ),
                        (
                            {"c": (11, 20)},
                            ValueError,
                        ),
                        # Constrain a or b outside estimated bounds.
                        (
                            {"a": (-1e9, -1e8)},
                            exceptions.OptimizationError,
                        ),
                        (
                            {"b": (1e8, 1e9)},
                            exceptions.OptimizationError,
                        ),
                ]:
                    with warnings.catch_warnings(),\
                         self.assertRaises(exception):
                        warnings.filterwarnings(
                            "ignore" if a == b else "default",
                            message=r"Parameters might be unidentifiable.",
                            category=RuntimeWarning,
                        )
                        dist_hat = parametric.NoisyQuadraticDistribution.fit(
                            ys=ys,
                            limits=limits,
                            constraints=constraints,
                        )

    @pytest.mark.level(3)
    def test_fit_for_interactions_between_limits_and_constraints(self):
        n_samples = 2_048

        a, b, c, o, convex = 0., 1., 1, 0., False
        dist = parametric.NoisyQuadraticDistribution(a, b, c, o, convex)

        # Test when constraints keeps a above the least of ys but not
        # the least *observed* y, thus the constraint is still feasible.
        y_min = -1.
        ys = dist.sample(n_samples)
        ys[np.argmin(ys)] = y_min
        dist_hat = parametric.NoisyQuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"a": ((y_min + a) / 2, np.inf), "o": 0.},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.o, 0.)
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

        # Test when constraints keeps b below the greatest of ys but not
        # the greatest *observed* y, thus the constraint is still feasible.
        y_max = 2.
        ys = dist.sample(n_samples)
        ys[np.argmax(ys)] = y_max
        dist_hat = parametric.NoisyQuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"b": (-np.inf, (b + y_max) / 2), "o": 0.},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.o, 0.)
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

        # Test when constraints prevents the distribution from putting
        # any probability in one of the censored tails. In that case,
        # observing a point in the tail is a zero probability event,
        # which can ruin the fit. If a point equals the lower limit (as
        # often happens with aggressive rounding), then it gets placed
        # in the left tail since limits defines a left-open interval.
        # Thus, it's important to test this edge case.
        #   zero probability mass in the left tail.
        y_min = a
        ys = dist.sample(n_samples)
        ys[np.argmin(ys)] = y_min
        dist_hat = parametric.NoisyQuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"a": (a, np.inf), "o": 0.},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.o, 0.)
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)
        #   zero probability mass in the right tail.
        y_max = b
        ys = dist.sample(n_samples)
        ys[np.argmax(ys)] = y_max
        dist_hat = parametric.NoisyQuadraticDistribution.fit(
            ys=ys,
            limits=(a, b),
            constraints={"b": (-np.inf, b), "o": 0.},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a))
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a))
        self.assertEqual(dist_hat.o, 0.)
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

    @pytest.mark.level(3)
    def test_fit_with_ties(self):
        n_samples = 2_048

        a, b, c, o, convex = 0., 1., 1, 1e-2, False
        dist = parametric.NoisyQuadraticDistribution(a, b, c, o, convex)

        # Test fit with aggressive (1e-2) and light (1e-4) rounding.

        for decimals in [2, 4]:
            ys = np.round(dist.sample(n_samples), decimals=decimals)

            dist_hat = parametric.NoisyQuadraticDistribution.fit(ys)

            self.assertAlmostEqual(
                dist_hat.a, a,
                delta=5e-2 * (b - a) + 4 * o,
            )
            self.assertAlmostEqual(
                dist_hat.b, b,
                delta=5e-2 * (b - a) + 4 * o,
            )
            self.assertGreaterEqual(dist_hat.o, o / 5.)
            self.assertLessEqual(dist_hat.o, 5. * o)
            self.assertEqual(dist_hat.c, c)
            self.assertEqual(dist_hat.convex, convex)

        # Test fit with small spacings (e.g., floating point errors).

        ys = np.round(dist.sample(n_samples), decimals=2)
        ys += (
            self.generator.choice([-256, -16, -1, 0, 1, 16, 256], size=len(ys))
            * np.spacing(ys)
        )

        dist_hat = parametric.NoisyQuadraticDistribution.fit(ys)

        self.assertAlmostEqual(
            dist_hat.a, a,
            delta=5e-2 * (b - a) + 4 * o,
        )
        self.assertAlmostEqual(
            dist_hat.b, b,
            delta=5e-2 * (b - a) + 4 * o,
        )
        self.assertGreaterEqual(dist_hat.o, o / 5.)
        self.assertLessEqual(dist_hat.o, 5. * o)
        self.assertEqual(dist_hat.c, c)
        self.assertEqual(dist_hat.convex, convex)

    @pytest.mark.level(2)
    def test_fit_accepts_integer_data(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Parameters might be unidentifiable.",
                category=RuntimeWarning,
            )
            dist_hat = parametric.NoisyQuadraticDistribution.fit([0] * 8)

        self.assertEqual(dist_hat.a, 0.)
        self.assertEqual(dist_hat.b, 0.)
        self.assertEqual(dist_hat.o, 0.)

    @pytest.mark.level(4)
    def test_fit_defaults_to_global_random_number_generator(self):
        n_samples = 256
        # fit should be deterministic if global seed is set.
        ys = parametric.NoisyQuadraticDistribution(
            0., 1., 1, 1e-2, False,
        ).sample(n_samples)
        #   Before setting the seed, two fits should be unequal.
        self.assertTrue(
            # Occasionally but rarely, the fits can be identical even
            # though the random seeds are not. Thus, automatically try a
            # second time in case this occurs so as to avoid spuriously
            # failing the test.
            (
                parametric.NoisyQuadraticDistribution.fit(ys)
                !=
                parametric.NoisyQuadraticDistribution.fit(ys)
            ) or (
                parametric.NoisyQuadraticDistribution.fit(ys)
                !=
                parametric.NoisyQuadraticDistribution.fit(ys)
            ),
        )
        #   After setting the seed, two fits should be unequal.
        opda.random.set_seed(0)
        self.assertTrue(
            # Occasionally but rarely, the fits can be identical even
            # though the random seeds are not. Thus, automatically try a
            # second time in case this occurs so as to avoid spuriously
            # failing the test.
            (
                parametric.NoisyQuadraticDistribution.fit(ys)
                !=
                parametric.NoisyQuadraticDistribution.fit(ys)
            ) or (
                parametric.NoisyQuadraticDistribution.fit(ys)
                !=
                parametric.NoisyQuadraticDistribution.fit(ys)
            ),
        )
        #   Resetting the seed should produce the same fit.
        opda.random.set_seed(0)
        first_fit = parametric.NoisyQuadraticDistribution.fit(ys)
        opda.random.set_seed(0)
        second_fit = parametric.NoisyQuadraticDistribution.fit(ys)
        self.assertEqual(first_fit, second_fit)

    @pytest.mark.level(3)
    def test_fit_is_deterministic_given_generator_argument(self):
        n_samples = 256
        ys = parametric.NoisyQuadraticDistribution(
            0., 1., 1, 1e-2, False,
        ).sample(n_samples)
        # Reusing the same generator, two fits should be unequal.
        generator = np.random.default_rng(0)
        self.assertTrue(
            # Occasionally but rarely, the fits can be identical even
            # though the random seeds are not. Thus, automatically try
            # a second time in case this occurs so as to avoid
            # spuriously failing the test.
            (
                parametric.NoisyQuadraticDistribution.fit(
                    ys, generator=generator,
                ) != parametric.NoisyQuadraticDistribution.fit(
                    ys, generator=generator,
                )
            ) or (
                parametric.NoisyQuadraticDistribution.fit(
                    ys, generator=generator,
                ) != parametric.NoisyQuadraticDistribution.fit(
                    ys, generator=generator,
                )
            ),
        )
        # Using generators in the same state should produce the same fit.
        self.assertEqual(
            parametric.NoisyQuadraticDistribution.fit(
                ys, generator=np.random.default_rng(0),
            ),
            parametric.NoisyQuadraticDistribution.fit(
                ys, generator=np.random.default_rng(0),
            ),
        )

    @pytest.mark.level(3)
    def test_fit_when_all_ys_are_negative(self):
        n_samples = 8

        # Due to the small sample size, fit might estimate that c = 2,
        # in which case it will fire a warning about identifiability.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Parameters might be unidentifiable.",
                category=RuntimeWarning,
            )

            # Test when a = b and o = 0.
            ys = [-1.] * n_samples
            parametric.NoisyQuadraticDistribution.fit(
                ys,
                method="maximum_spacing",
            )

            # Test when a != b and o > 0.
            ys = parametric.NoisyQuadraticDistribution(
                -2., -1., 1, 1e-2, False,
            ).sample(n_samples)
            parametric.NoisyQuadraticDistribution.fit(
                ys,
                method="maximum_spacing",
            )

    @pytest.mark.level(1)
    def test_fit_with_censored_infinity(self):
        n_samples = 2_048

        a, b, c, o, convex = 0., 1., 2, 4e-2, False
        dist = parametric.NoisyQuadraticDistribution(a, b, c, o, convex)
        ys = dist.sample(n_samples)

        # NOTE: Use constraints in fit below to make fit run faster.

        # Test when ys contains inf.
        #   positive infinitiy
        with self.assertRaises(ValueError):
            parametric.NoisyQuadraticDistribution.fit(
                ys=np.concatenate([[np.inf], ys]),
                limits=(-np.inf, np.inf),
                constraints={"c": c, "convex": convex},
            )
        #   negative infinitiy
        with self.assertRaises(ValueError):
            parametric.NoisyQuadraticDistribution.fit(
                ys=np.concatenate([[-np.inf], ys]),
                limits=(-np.inf, np.inf),
                constraints={"c": c, "convex": convex},
            )
        #   both positive and negative infinity
        with self.assertRaises(ValueError):
            parametric.NoisyQuadraticDistribution.fit(
                ys=np.concatenate([[-np.inf, np.inf], ys]),
                limits=(-np.inf, np.inf),
                constraints={"c": c, "convex": convex},
            )

        # Test when ys contains inf but it is censored.
        #   positive infinitiy
        dist_hat = parametric.NoisyQuadraticDistribution.fit(
            ys=np.concatenate([[np.inf], ys]),
            limits=(-np.inf, b),
            constraints={"c": c, "convex": convex},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a) + o)
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a) + o)
        self.assertGreaterEqual(dist_hat.o, o / 5.)
        self.assertLessEqual(dist_hat.o, 5. * o)
        #   negative infinitiy
        dist_hat = parametric.NoisyQuadraticDistribution.fit(
            ys=np.concatenate([[-np.inf], ys]),
            limits=(a, np.inf),
            constraints={"c": c, "convex": convex},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a) + o)
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a) + o)
        self.assertGreaterEqual(dist_hat.o, o / 5.)
        self.assertLessEqual(dist_hat.o, 5. * o)
        #   both positive and negative infinity
        dist_hat = parametric.NoisyQuadraticDistribution.fit(
            ys=np.concatenate([[-np.inf, np.inf], ys]),
            limits=(a, b),
            constraints={"c": c, "convex": convex},
        )
        self.assertAlmostEqual(dist_hat.a, a, delta=5e-2 * (b - a) + o)
        self.assertAlmostEqual(dist_hat.b, b, delta=5e-2 * (b - a) + o)
        self.assertGreaterEqual(dist_hat.o, o / 5.)
        self.assertLessEqual(dist_hat.o, 5. * o)
