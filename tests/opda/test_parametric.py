"""Tests for opda.parametric."""

import numpy as np
import pytest
from scipy import stats

from opda import parametric, utils
import opda.random

from tests import testcases


class QuadraticDistributionTestCase(testcases.RandomTestCase):
    """Test opda.parametric.QuadraticDistribution."""

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
            "QuadraticDistribution(a=0.0, b=1.0, c=1, convex=False)",
        )

    def test___repr__(self):
        self.assertEqual(
            repr(parametric.QuadraticDistribution(0., 1., 1, convex=False)),
            "QuadraticDistribution(a=0.0, b=1.0, c=1, convex=False)",
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
                    curve = np.median(
                        np.minimum.accumulate(yss, axis=1)
                        if expect_minimize else
                        np.maximum.accumulate(yss, axis=1),
                        axis=0,
                    )

                    # Test when n is integral.
                    #   scalar
                    for n in range(1, 6):
                        self.assertTrue(np.isscalar(
                            dist.quantile_tuning_curve(
                                n,
                                q=0.5,
                                minimize=minimize,
                            ),
                        ))
                        self.assertAlmostEqual(
                            dist.quantile_tuning_curve(
                                n,
                                q=0.5,
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=0.075,
                        )
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [n],
                                q=0.5,
                                minimize=minimize,
                            ),
                            [
                                dist.quantile_tuning_curve(
                                    n,
                                    q=0.5,
                                    minimize=minimize,
                                ),
                            ],
                        ))
                    #   1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [1, 2, 3, 4, 5],
                            q=0.5,
                            minimize=minimize,
                        ),
                        curve,
                        atol=0.075,
                    ))
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [3, 1, 5],
                            q=0.5,
                            minimize=minimize,
                        ),
                        [curve[2], curve[0], curve[4]],
                        atol=0.075,
                    ))
                    #   2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [1, 2, 3],
                                [3, 1, 5],
                            ],
                            q=0.5,
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
                            dist.quantile_tuning_curve(
                                n/10.,
                                q=0.5**(1/10),
                                minimize=minimize,
                            ),
                        ))
                        self.assertAlmostEqual(
                            dist.quantile_tuning_curve(
                                n/10.,
                                q=0.5**(1/10),
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=0.075,
                        )
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [n/10.],
                                q=0.5**(1/10),
                                minimize=minimize,
                            ),
                            [
                                dist.quantile_tuning_curve(
                                    n/10.,
                                    q=0.5**(1/10),
                                    minimize=minimize,
                                ),
                            ],
                        ))
                    #   1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [0.1, 0.2, 0.3, 0.4, 0.5],
                            q=0.5**(1/10),
                            minimize=minimize,
                        ),
                        curve,
                        atol=0.075,
                    ))
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [0.3, 0.1, 0.5],
                            q=0.5**(1/10),
                            minimize=minimize,
                        ),
                        [curve[2], curve[0], curve[4]],
                        atol=0.075,
                    ))
                    #   2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [0.1, 0.2, 0.3],
                                [0.3, 0.1, 0.5],
                            ],
                            q=0.5**(1/10),
                            minimize=minimize,
                        ),
                        [
                            [curve[0], curve[1], curve[2]],
                            [curve[2], curve[0], curve[4]],
                        ],
                        atol=0.075,
                    ))

                    # Test ns <= 0.
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            0,
                            q=0.5,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            -1,
                            q=0.5,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [0],
                            q=0.5,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [-2],
                            q=0.5,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [0, 1],
                            q=0.5,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [-2, 1],
                            q=0.5,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [[0], [1]],
                            q=0.5,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [[-2], [1]],
                            q=0.5,
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

    def test_estimate_initial_parameters_and_bounds(self):
        # Test when a = b.
        a, b = 0., 0.
        for c in [1, 10]:
            for convex in [False, True]:
                for fraction in [0.5, 1.]:
                    ys = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    ).sample(5)
                    init_params, bounds = parametric.QuadraticDistribution\
                        .estimate_initial_parameters_and_bounds(
                            ys,
                            fraction=fraction,
                            convex=convex,
                        )
                    self.assertEqual(init_params[0], a)
                    self.assertEqual(init_params[1], b)
                    self.assertTrue(np.isnan(init_params[2]))
                    self.assertGreaterEqual(a, bounds[0, 0])
                    self.assertLessEqual(a, bounds[0, 1])
                    self.assertGreaterEqual(b, bounds[1, 0])
                    self.assertLessEqual(b, bounds[1, 1])
                    self.assertEqual(0, bounds[2, 0])
                    self.assertEqual(np.inf, bounds[2, 1])

        # Test when a != b.
        for a, b, c in [(0., 1., 1), (-1., 1., 2)]:
            for convex in [False, True]:
                for fraction in [0.5, 1.]:
                    ys = parametric.QuadraticDistribution(
                        a, b, c, convex=convex,
                    ).sample(5_000)
                    init_params, bounds = parametric.QuadraticDistribution\
                        .estimate_initial_parameters_and_bounds(
                            ys,
                            fraction=fraction,
                            convex=convex,
                        )
                    self.assertLess(abs(init_params[0] - a), 0.35)
                    self.assertLess(abs(init_params[1] - b), 0.35)
                    self.assertLess(abs(init_params[2] - c), 0.25)
                    self.assertGreaterEqual(a, bounds[0, 0])
                    self.assertLessEqual(a, bounds[0, 1])
                    self.assertGreaterEqual(b, bounds[1, 0])
                    self.assertLessEqual(b, bounds[1, 1])
                    self.assertGreaterEqual(c, bounds[2, 0])
                    self.assertLessEqual(c, bounds[2, 1])

        # Test error conditions.
        for convex in [False, True]:
            # when ys is not 1D
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        0,
                        fraction=0.5,
                        convex=convex,
                    )
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [[0]],
                        fraction=0.5,
                        convex=convex,
                    )
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [[0], [1]],
                        fraction=0.5,
                        convex=convex,
                    )
            # when ys is empty
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [],
                        fraction=0.5,
                        convex=convex,
                    )
            # when fraction is not a scalar
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=[0.5],
                        convex=convex,
                    )
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=[[0.5]],
                        convex=convex,
                    )
            # when fraction is not between 0 and 1, inclusive of 1
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=-0.1,
                        convex=convex,
                    )
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=0.,
                        convex=convex,
                    )
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=1.1,
                        convex=convex,
                    )
            # when convex is not a bool
            with self.assertRaises(TypeError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=0.5,
                        convex=None,
                    )
            with self.assertRaises(TypeError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=0.5,
                        convex=0.,
                    )
            with self.assertRaises(TypeError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2, 3, 4, 5],
                        fraction=0.5,
                        convex=[True],
                    )
            # when fraction * len(ys) < 1
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0],
                        fraction=0.99,
                        convex=convex,
                    )
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1],
                        fraction=0.49,
                        convex=convex,
                    )
            with self.assertRaises(ValueError):
                parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(
                        [0, 1, 2],
                        fraction=0.32,
                        convex=convex,
                    )

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
                        if convex else
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


class NoisyQuadraticDistributionTestCase(testcases.RandomTestCase):
    """Test opda.parametric.NoisyQuadraticDistribution."""

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
            "NoisyQuadraticDistribution("
                "a=0.0, b=1.0, c=1, o=1.0, convex=False"
            ")",
        )

    def test___repr__(self):
        self.assertEqual(
            repr(parametric.NoisyQuadraticDistribution(
                0., 1., 1, 1., convex=False,
            )),
            "NoisyQuadraticDistribution("
                "a=0.0, b=1.0, c=1, o=1.0, convex=False"
            ")",
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
                        yss = dist.sample((2_000, 5))
                        curve_lo, curve, curve_hi = np.sort(
                            np.minimum.accumulate(yss, axis=1)
                            if expect_minimize else
                            np.maximum.accumulate(yss, axis=1),
                            axis=0,
                        )[(  # Use the binomial confidence interval for
                             # the median.
                             891,  # lower 1 - 1e-6 confidence bound
                            1000,  # point estimate
                            1109,  # upper 1 - 1e-6 confidence bound
                        ), :]
                        atol = np.max(curve_hi - curve_lo) / 2

                        # Test when n is integral.
                        #   scalar
                        for n in range(1, 6):
                            self.assertTrue(np.isscalar(
                                dist.quantile_tuning_curve(
                                    n,
                                    q=0.5,
                                    minimize=minimize,
                                ),
                            ))
                            self.assertAlmostEqual(
                                dist.quantile_tuning_curve(
                                    n,
                                    q=0.5,
                                    minimize=minimize,
                                ),
                                curve[n-1],
                                delta=atol,
                            )
                            self.assertTrue(np.allclose(
                                dist.quantile_tuning_curve(
                                    [n],
                                    q=0.5,
                                    minimize=minimize,
                                ),
                                [
                                    dist.quantile_tuning_curve(
                                        n,
                                        q=0.5,
                                        minimize=minimize,
                                    ),
                                ],
                            ))
                        #   1D array
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [1, 2, 3, 4, 5],
                                q=0.5,
                                minimize=minimize,
                            ),
                            curve,
                            atol=atol,
                        ))
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [3, 1, 5],
                                q=0.5,
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
                                q=0.5,
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
                                    q=0.5**(1/10),
                                    minimize=minimize,
                                ),
                            ))
                            self.assertAlmostEqual(
                                dist.quantile_tuning_curve(
                                    n/10.,
                                    q=0.5**(1/10),
                                    minimize=minimize,
                                ),
                                curve[n-1],
                                delta=atol,
                            )
                            self.assertTrue(np.allclose(
                                dist.quantile_tuning_curve(
                                    [n/10.],
                                    q=0.5**(1/10),
                                    minimize=minimize,
                                ),
                                [
                                    dist.quantile_tuning_curve(
                                        n/10.,
                                        q=0.5**(1/10),
                                        minimize=minimize,
                                    ),
                                ],
                            ))
                        #   1D array
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [0.1, 0.2, 0.3, 0.4, 0.5],
                                q=0.5**(1/10),
                                minimize=minimize,
                            ),
                            curve,
                            atol=atol,
                        ))
                        self.assertTrue(np.allclose(
                            dist.quantile_tuning_curve(
                                [0.3, 0.1, 0.5],
                                q=0.5**(1/10),
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
                                q=0.5**(1/10),
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
                                q=0.5,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                -1,
                                q=0.5,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [0],
                                q=0.5,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [-2],
                                q=0.5,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [0, 1],
                                q=0.5,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [-2, 1],
                                q=0.5,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [[0], [1]],
                                q=0.5,
                                minimize=minimize,
                            )
                        with self.assertRaises(ValueError):
                            dist.quantile_tuning_curve(
                                [[-2], [1]],
                                q=0.5,
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
                        atol=1e-3 / o,
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
                            if convex else
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

                        ys_center = ys[(ys >= a) & (ys <= b)]
                        self.assertTrue(np.allclose(
                            dist.ppf(dist.cdf(ys_center)),
                            ys_center,
                            atol=1e-5,
                        ))
                        # NOTE: Precision of the ppf is more difficult
                        # in the tails because the CDF is nearly flat.
                        ys_tails = ys[(ys < a) | (ys > b)]
                        self.assertTrue(np.allclose(
                            dist.ppf(dist.cdf(ys_tails)),
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
