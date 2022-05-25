"""Tests for ersa.parametric"""

import unittest

import numpy as np

from ersa import parametric


class QuadraticDistributionTestCase(unittest.TestCase):
    """Test ersa.parametric.QuadraticDistribution."""

    def test_estimate_initial_parameters_and_bounds(self):
        a, b, c = 0., 1., 1.
        ys = parametric.QuadraticDistribution(a, b, c).sample(1_000)
        for fraction in [0.5, 1.]:
            for convex in [False, True]:
                init_params, bounds = parametric.QuadraticDistribution\
                    .estimate_initial_parameters_and_bounds(ys, fraction=1.)
                self.assertLess(abs(init_params[0] - a), 0.1)
                self.assertLess(abs(init_params[1] - b), 0.1)
                self.assertLess(abs(init_params[2] - c), 0.25)
                self.assertGreater(a, bounds[0, 0])
                self.assertLess(a, bounds[0, 1])
                self.assertGreater(b, bounds[1, 0])
                self.assertLess(b, bounds[1, 1])
                self.assertGreater(c, bounds[2, 0])
                self.assertLess(c, bounds[2, 1])

    def test_sample(self):
        a, b = 0., 1.
        for c in [0.5, 10.]:
            for convex in [False, True]:
                ys = parametric.QuadraticDistribution(a, b, c).sample(2_500)
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
        dist = parametric.QuadraticDistribution(a, b, c)
        for convex in [False, True]:
            # not broadcasting
            for n in range(6):
                # When c = 1., the distribution is uniform.
                self.assertEqual(dist.pdf(a + (n / 5.) * (b - a)), np.array(1.))
            self.assertEqual(dist.pdf(a - 1e-10), np.array(0.))
            self.assertEqual(dist.pdf(b + 1e-10), np.array(0.))
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

    def test_cdf(self):
        a, b, c = 0., 1., 1.
        dist = parametric.QuadraticDistribution(a, b, c)
        for convex in [False, True]:
            # not broadcasting
            for n in range(6):
                # When c = 1., the distribution is uniform.
                self.assertAlmostEqual(dist.cdf(a + (n / 5.) * (b - a)), np.array(n / 5.))
            self.assertEqual(dist.cdf(a - 1e-10), 0.)
            self.assertEqual(dist.cdf(b + 1e-10), 1.)
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

    def test_ppf(self):
        a, b, c = 0., 1., 1.
        dist = parametric.QuadraticDistribution(a, b, c)
        for convex in [False, True]:
            # not broadcasting
            for n in range(6):
                # When c = 1., the distribution is uniform.
                self.assertAlmostEqual(dist.ppf(n / 5.), a + (n / 5.) * (b - a))
            self.assertEqual(dist.ppf(0. - 1e-10), a)
            self.assertEqual(dist.ppf(0.), a)
            self.assertEqual(dist.ppf(1.), b)
            self.assertEqual(dist.ppf(1. + 1e-10), b)
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
                dist = parametric.QuadraticDistribution(a, b, c)
                yss = dist.sample((2_000, 5))
                self.assertTrue(np.allclose(
                    np.median(np.maximum.accumulate(yss, axis=1), axis=0),
                    dist.quantile_tuning_curve(list(range(1, 6)), 0.5),
                    atol=0.075,
                ))

    def test_average_tuning_curve(self):
        a, b = 0., 1.
        for c in [0.5, 10.]:
            for convex in [False, True]:
                dist = parametric.QuadraticDistribution(a, b, c)
                yss = dist.sample((2_000, 5))
                self.assertTrue(np.allclose(
                    np.mean(np.maximum.accumulate(yss, axis=1), axis=0),
                    dist.average_tuning_curve(list(range(1, 6))),
                    atol=0.075,
                ))
