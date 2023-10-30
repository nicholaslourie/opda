"""Tests for experiments.visualization"""

import unittest

from experiments import visualization


class PlotRandomSearchTestCase(unittest.TestCase):
    """Test experiments.visualization.plot_random_search."""


class PlotCdfTestCase(unittest.TestCase):
    """Test experiments.visualization.plot_cdf."""


class PlotPdfTestCase(unittest.TestCase):
    """Test experiments.visualization.plot_pdf."""


class PlotDistributionTestCase(unittest.TestCase):
    """Test experiments.visualization.plot_distribution."""


class PlotDistributionApproximationTestCase(unittest.TestCase):
    """Test experiments.visualization.plot_distribution_approximation."""


class PlotTuningCurveApproximationTestCase(unittest.TestCase):
    """Test experiments.visualization.plot_tuning_curve_approximation."""


class ColorWithLightnessTestCase(unittest.TestCase):
    """Test experiments.visualization.color_with_lightness."""

    def test_color_with_lightness(self):
        colors_by = {
            'name': [
                'r',
                'g',
                'tab:blue',
            ],
            'hex': [
                '#f00',
                '#0f0',
                '#00f',
            ],
            'rgb': [
                (0.25, 0.00, 0.00),
                (0.00, 0.25, 0.00),
                (0.00, 0.00, 0.25),
            ],
        }
        for by, colors in colors_by.items():
            for color in colors:
                # Test maximum darkness.
                for channel in visualization.color_with_lightness(color, 0.):
                    self.assertAlmostEqual(channel, 0.)
                # Test maximum lightness.
                for channel in visualization.color_with_lightness(color, 1.):
                    self.assertAlmostEqual(channel, 1.)
                # Test intermediate lightness values
                if by == 'rgb':
                    for c1, c2 in zip(  # "c" for "channel"
                            color,
                            visualization.color_with_lightness(color, 0.1),
                    ):
                        # Only non-zero color channels should end up non-zero.
                        self.assertEqual(c1 > 0., c2 > 0.)
                        # No color channels should be turned all the way up.
                        self.assertLess(c2, 1.)
