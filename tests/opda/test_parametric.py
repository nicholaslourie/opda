"""Tests for opda.parametric"""

import unittest

import numpy as np

from opda import parametric


class QuadraticDistributionTestCase(unittest.TestCase):
    """Test opda.parametric.QuadraticDistribution."""

    def test___repr__(self):
        self.assertEqual(
            repr(parametric.QuadraticDistribution(0., 1., 0.5, convex=False)),
            "QuadraticDistribution(a=0.0, b=1.0, c=0.5, convex=False)",
        )

    def test_sample(self):
        a, b = 0., 1.
        for c in [0.5, 10.]:
            for convex in [False, True]:
                ys = parametric.QuadraticDistribution(
                    a, b, c, convex=convex,
                ).sample(2_500)
                self.assertLess(a, np.min(ys))
                self.assertGreater(b, np.max(ys))
        # Test when c = 1. and the samples should be uniformly distributed.
        a, b, c = 0., 1., 1.
        ys = parametric.QuadraticDistribution(a, b, c).sample(2_500)
        self.assertLess(a, np.min(ys))
        self.assertGreater(b, np.max(ys))
        self.assertLess(abs(np.mean(ys < 0.5) - 0.5), 0.05)

    def test_pdf(self):
        a, b, c = 0., 1., 1.
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # not broadcasting
            for n in range(6):
                # When c = 1., the distribution is uniform.
                self.assertEqual(dist.pdf(a + (n / 5.) * (b - a)), np.array(1.))
            # broadcasting
            for _ in range(7):
                us = np.random.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.ones_like(us).tolist(),
                )
                us = np.random.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.ones_like(us).tolist(),
                )
                us = np.random.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.ones_like(us).tolist(),
                )

        # Test outside of the distribution's support.
        for a, b, c in [(0., 1., 0.5), (0., 1., 1.)]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                self.assertEqual(dist.pdf(a - 1e-10), np.array(0.))
                self.assertEqual(dist.pdf(a - 10), np.array(0.))
                self.assertEqual(dist.pdf(b + 1e-10), np.array(0.))
                self.assertEqual(dist.pdf(b + 10), np.array(0.))

    def test_cdf(self):
        a, b, c = 0., 1., 1.
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # not broadcasting
            for n in range(6):
                # When c = 1., the distribution is uniform.
                self.assertAlmostEqual(
                    dist.cdf(a + (n / 5.) * (b - a)),
                    np.array(n / 5.),
                )
            # broadcasting
            for _ in range(7):
                us = np.random.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )
                us = np.random.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )
                us = np.random.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.cdf(a + us * (b - a)).tolist(),
                    us.tolist(),
                )

        # Test outside of the distribution's support.
        for a, b, c in [(0., 1., 0.5), (0., 1., 1.)]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                self.assertEqual(dist.cdf(a - 1e-10), 0.)
                self.assertEqual(dist.cdf(a - 10), 0.)
                self.assertEqual(dist.cdf(b + 1e-10), 1.)
                self.assertEqual(dist.cdf(b + 10), 1.)

    def test_ppf(self):
        a, b, c = 0., 1., 1.
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # not broadcasting
            for n in range(6):
                # When c = 1., the distribution is uniform.
                self.assertAlmostEqual(dist.ppf(n / 5.), a + (n / 5.) * (b - a))
            # broadcasting
            for _ in range(7):
                us = np.random.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                )
                us = np.random.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                )
                us = np.random.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.ppf(us).tolist(),
                    (a + us * (b - a)).tolist(),
                )

    def test_quantile_tuning_curve(self):
        a, b = 0., 1.
        for c in [0.5, 10.]:
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
                        self.assertAlmostEqual(
                            dist.quantile_tuning_curve(
                                n,
                                q=0.5,
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=0.075,
                        )
                        self.assertEqual(
                            dist.quantile_tuning_curve(
                                [n],
                                q=0.5,
                                minimize=minimize,
                            ).tolist(),
                            [
                                dist.quantile_tuning_curve(
                                    n,
                                    q=0.5,
                                    minimize=minimize,
                                ),
                            ],
                        )
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
                        self.assertAlmostEqual(
                            dist.quantile_tuning_curve(
                                n/10.,
                                q=0.5**(1/10),
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=0.075,
                        )
                        self.assertEqual(
                            dist.quantile_tuning_curve(
                                [n/10],
                                q=0.5**(1/10),
                                minimize=minimize,
                            ).tolist(),
                            [
                                dist.quantile_tuning_curve(
                                    n/10.,
                                    q=0.5**(1/10),
                                    minimize=minimize,
                                ),
                            ],
                        )
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
        for c in [0.5, 10.]:
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
                        self.assertAlmostEqual(
                            dist.average_tuning_curve(
                                n,
                                minimize=minimize,
                            ),
                            curve[n-1],
                            delta=0.075,
                        )
                        self.assertEqual(
                            dist.average_tuning_curve(
                                [n],
                                minimize=minimize,
                            ).tolist(),
                            [
                                dist.average_tuning_curve(
                                    n,
                                    minimize=minimize,
                                )],
                        )
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
                        self.assertEqual(
                            dist.average_tuning_curve(
                                [n - 0.5],
                                minimize=minimize,
                            ).tolist(),
                            [
                                dist.average_tuning_curve(
                                    n - 0.5,
                                    minimize=minimize,
                                ),
                            ],
                        )
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
        for a, b, c in [(0., 1., 0.5), (-1., 1., 1.)]:
            for fraction in [0.5, 1.]:
                for convex in [False, True]:
                    ys = parametric.QuadraticDistribution(
                        a,
                        b,
                        c,
                        convex=convex,
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

    def test_pdf_on_boundary_of_support(self):
        for convex in [False, True]:
            a, b, c = 0., 1., 0.5
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            self.assertEqual(dist.pdf(a), np.inf if convex else 0.5)
            self.assertEqual(dist.pdf(b), 0.5 if convex else np.inf)

            a, b, c = 0., 1., 1.
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            self.assertEqual(dist.pdf(a), 1.)
            self.assertEqual(dist.pdf(b), 1.)

    def test_cdf_on_boundary_of_support(self):
        for convex in [False, True]:
            a, b, c = 0., 1., 0.5
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            self.assertEqual(dist.cdf(a), 0.)
            self.assertEqual(dist.cdf(b), 1.)

            a, b, c = 0., 1., 1.
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            self.assertEqual(dist.cdf(a), 0.)
            self.assertEqual(dist.cdf(b), 1.)

    def test_ppf_is_inverse_of_cdf(self):
        # NOTE: For continuous distributions like the quadratic
        # distribution, the quantile function is the inverse of the
        # cumulative distribution function.
        a, b = 0., 1.
        for c in [0.5, 10.]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                for _ in range(5):
                    ys = dist.sample(100)
                    self.assertTrue(np.allclose(dist.ppf(dist.cdf(ys)), ys))
                    us = np.random.uniform(0, 1, size=100)
                    self.assertTrue(np.allclose(dist.cdf(dist.ppf(us)), us))

    def test_ppf_at_extremes(self):
        a, b = 0., 1.
        for c in [0.5, 10.]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                self.assertEqual(dist.ppf(0. - 1e-12), a)
                self.assertEqual(dist.ppf(0.), a)
                self.assertEqual(dist.ppf(1.), b)
                self.assertEqual(dist.ppf(1. + 1e-12), b)

    def test_quantile_tuning_curve_minimize_is_dual_to_maximize(self):
        for _ in range(4):
            for a, b, c in [(-1., 1., 0.5), (-1., 1., 1.)]:
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
        for _ in range(4):
            for a, b, c in [(-1., 1., 0.5), (-1., 1., 1.)]:
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
