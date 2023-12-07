"""Tests for experiments.analytic"""

import unittest

from autograd import numpy as npx
import numpy as np
import pytest

from opda import nonparametric, parametric, utils
from experiments import analytic, simulation


class EllipseVolumeTestCase(unittest.TestCase):
    """Test experiments.analytic.ellipse_volume."""

    def test_ellipse_volume(self):
        # Test the volume of an n-ball.
        #   1-ball
        self.assertAlmostEqual(
            analytic.ellipse_volume([1]),
            2.0,
        )
        #   2-ball
        self.assertAlmostEqual(
            analytic.ellipse_volume([1, 1]),
            np.pi,
        )
        #   3-ball
        self.assertAlmostEqual(
            analytic.ellipse_volume([1, 1, 1]),
            4./3 * np.pi,
        )
        #   4-ball
        self.assertAlmostEqual(
            analytic.ellipse_volume([1, 1, 1, 1]),
            1./2 * np.pi**2,
        )

        # Test the volume of an ellipse.
        #   1D
        self.assertAlmostEqual(
            analytic.ellipse_volume([2]),
            2 * 2.0,
        )
        #   2D
        self.assertAlmostEqual(
            analytic.ellipse_volume([2, 3]),
            6 * np.pi,
        )
        #   3D
        self.assertAlmostEqual(
            analytic.ellipse_volume([2, 3, 5]),
            30 * 4./3 * np.pi,
        )
        #   4D
        self.assertAlmostEqual(
            analytic.ellipse_volume([2, 3, 5, 7]),
            210 * 1./2 * np.pi**2,
        )


class GetApproximationParametersTestCase(unittest.TestCase):
    """Test experiments.analytic.get_approximation_parameters."""

    @pytest.mark.level(2)
    def test_get_approximation_parameters(self):
        # Quadratic Functions
        for n_dims in [1, 2, 3]:
            n_ball_vol = analytic.ellipse_volume([1] * n_dims)
            a, b, c = analytic.get_approximation_parameters(
                func=lambda xs: - npx.sum(xs**2, axis=-1),
                bounds=[(-0.5, 0.5)] * n_dims,
            )
            self.assertAlmostEqual(
                a,
                # minimum value (normalizer)
                - (1 / n_ball_vol)**(2 / n_dims),
            )
            self.assertAlmostEqual(
                b,
                # maximum value
                0.0,
            )
            self.assertAlmostEqual(
                c,
                # dimension / 2
                n_dims / 2,
            )
        # Non-Quadratic Functions
        n_trials = 10_000
        n_samples = 100
        for n_dims in [1, 2, 3]:
            s = simulation.Simulation.run(
                n_trials=n_trials,
                n_samples=n_samples,
                n_dims=n_dims,
                func=simulation.make_damped_linear_sin(
                    scale=1,
                    weights=[1] * n_dims,
                    bias=0,
                ),
                bounds=[(0.0, 0.5)] * n_dims,
            )
            approximate_distribution = parametric.QuadraticDistribution(
                *analytic.get_approximation_parameters(
                    func=s.func,
                    bounds=s.bounds,
                ),
            )
            empirical_distribution = nonparametric.EmpiricalDistribution(
                s.yss[:, 0],
            )
            empirical_max_distribution = nonparametric.EmpiricalDistribution(
                s.yss_cummax[:, n_samples - 1],
            )

            ns = np.arange(n_samples // 2, n_samples + 1)
            grid = np.linspace(s.y_min, s.y_max, num=10_000)
            #     Compare the approximation to the CDF of the max.
            self.assertTrue(np.allclose(
                approximate_distribution.cdf(grid)**n_samples,
                empirical_max_distribution.cdf(grid),
                atol=utils.dkw_epsilon(n_trials, 0.999999),
            ))
            #     Compare the approximation and the tail of the tuning curves.
            self.assertTrue(np.allclose(
                approximate_distribution.quantile_tuning_curve(ns),
                empirical_distribution.quantile_tuning_curve(ns),
                atol=0.025,
            ))
            self.assertTrue(np.allclose(
                approximate_distribution.average_tuning_curve(ns),
                empirical_distribution.average_tuning_curve(ns),
                atol=0.025,
            ))
