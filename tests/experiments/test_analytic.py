"""Tests for experiments.analytic"""

import unittest

import numpy as np

from experiments import analytic


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
