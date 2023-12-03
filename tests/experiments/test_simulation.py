"""Tests for experiments.simulation"""

import unittest

import numpy as np
import pytest

from experiments import simulation


class MakeDampedLinearSinTestCase(unittest.TestCase):
    """Test experiments.simulation.make_damped_linear_sin."""

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


class SimulationTestCase(unittest.TestCase):
    """Test experiments.simulation.Simulation."""

    @pytest.mark.level(1)
    def test_run(self):
        sim_kwargs = {
            'n_trials': 7,
            'n_samples': 5,
            'n_dims': 3,
            'func': simulation.make_damped_linear_sin(
                weights=[1., -1., 0.],
                bias=0.,
                scale=1.,
            ),
            'bounds': [[-1., 1.]] * 3,
        }
        sim = simulation.Simulation.run(**sim_kwargs)

        for key, expected in sim_kwargs.items():
            actual = getattr(sim, key)
            if isinstance(actual, np.ndarray):
                self.assertEqual(actual.tolist(), expected)
            else:
                self.assertEqual(actual, expected)

        self.assertLessEqual(sim.y_min, np.min(sim.yss))
        self.assertGreaterEqual(sim.y_max, np.max(sim.yss))
        self.assertEqual(sim.y_min, sim.func(sim.y_argmin))
        self.assertEqual(sim.y_max, sim.func(sim.y_argmax))
        self.assertEqual(
            sim.xss.shape,
            (sim.n_trials, sim.n_samples, sim.n_dims),
        )
        self.assertEqual(
            sim.yss.shape,
            (sim.n_trials, sim.n_samples),
        )
        self.assertEqual(
            sim.yss_cummax.shape,
            (sim.n_trials, sim.n_samples),
        )
