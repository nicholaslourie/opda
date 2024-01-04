"""Tests for opda.parametric."""

import numpy as np

from opda import parametric
import opda.random

from tests import testcases


class QuadraticDistributionTestCase(testcases.RandomTestCase):
    """Test opda.parametric.QuadraticDistribution."""

    def test___eq__(self):
        bounds = [(-10., -1.), (-1., 0.), (-1., 1.), (0., 1.), (1., 10.)]
        cs = [0.5, 1., 10.]
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
            str(parametric.QuadraticDistribution(0., 1., 0.5, convex=False)),
            "QuadraticDistribution(a=0.0, b=1.0, c=0.5, convex=False)",
        )

    def test___repr__(self):
        self.assertEqual(
            repr(parametric.QuadraticDistribution(0., 1., 0.5, convex=False)),
            "QuadraticDistribution(a=0.0, b=1.0, c=0.5, convex=False)",
        )

    def test_sample(self):
        a, b = 0., 1.
        # Test sample for various values of a, b, and c.
        for c in [0.5, 10.]:
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
                self.assertLess(a, np.min(ys))
                self.assertGreater(b, np.max(ys))
                # 2D array
                ys = dist.sample((10, 10))
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
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.pdf(a + (n / 5.) * (b - a))))
                # When c = 1., the distribution is uniform.
                self.assertEqual(dist.pdf(a + (n / 5.) * (b - a)), 1.)
            # broadcasting
            for _ in range(7):
                # 1D array
                us = self.generator.uniform(0, 1, size=5)
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.ones_like(us).tolist(),
                )
                # 2D array
                us = self.generator.uniform(0, 1, size=(5, 3))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.ones_like(us).tolist(),
                )
                # 3D array
                us = self.generator.uniform(0, 1, size=(5, 3, 2))
                self.assertEqual(
                    dist.pdf(a + us * (b - a)).tolist(),
                    np.ones_like(us).tolist(),
                )

        # Test outside of the distribution's support.
        for a, b, c in [(0., 1., 0.5), (0., 1., 1.)]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
                self.assertEqual(dist.pdf(a - 1e-10), 0.)
                self.assertEqual(dist.pdf(a - 10), 0.)
                self.assertEqual(dist.pdf(b + 1e-10), 0.)
                self.assertEqual(dist.pdf(b + 10), 0.)

    def test_cdf(self):
        a, b, c = 0., 1., 1.
        for convex in [False, True]:
            dist = parametric.QuadraticDistribution(a, b, c, convex=convex)
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.cdf(a + (n / 5.) * (b - a))))
                # When c = 1., the distribution is uniform.
                self.assertAlmostEqual(
                    dist.cdf(a + (n / 5.) * (b - a)),
                    n / 5.,
                )
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
            # scalar
            for n in range(6):
                self.assertTrue(np.isscalar(dist.ppf(n / 5.)))
                # When c = 1., the distribution is uniform.
                self.assertAlmostEqual(dist.ppf(n / 5.), a + (n / 5.) * (b - a))
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

    def test_sample_defaults_to_global_random_number_generator(self):
        # sample should be deterministic if global seed is set.
        dist = parametric.QuadraticDistribution(0., 1., 1.)
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
        dist = parametric.QuadraticDistribution(0., 1., 1.)
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
                    us = self.generator.uniform(0, 1, size=100)
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
