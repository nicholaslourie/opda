"""Tests for experiments.simulation"""

import unittest

from experiments import simulation


class MakeDampedLinearSinTestCase(unittest.TestCase):
    """Test experiments.make_damped_linear_sin."""

    def test_make_damped_linear_sin(self):
        # Test 1 dimension.
        func = simulation.make_damped_linear_sin(
            weights=[1.5],
            bias=0.,
            scale=1.,
        )
        self.assertEqual(func([1.]).shape, ())
        self.assertEqual(func([[1.]]).shape, (1,))
        self.assertEqual(func([[1.], [2.]]).shape, (2,))

        # Test 3 dimensions.
        func = simulation.make_damped_linear_sin(
            weights=[1., 1., 1.],
            bias=0.,
            scale=1.,
        )
        self.assertEqual(func([1., 1., 1.]).shape, ())
        self.assertEqual(func([[1., 1., 1.]]).shape, (1,))
        self.assertEqual(
            func([
                [1., 1., 1.],
                [1., 2., 3.],
            ]).shape,
            (2,),
        )
