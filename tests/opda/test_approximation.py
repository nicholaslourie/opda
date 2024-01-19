"""Tests for opda.approximation."""

import itertools
import unittest

import numpy as np

from opda import approximation


class LagrangeInterpolateTestCase(unittest.TestCase):
    """Test opda.approximation.lagrange_interpolate."""

    def test_lagrange_interpolate(self):
        # Test argument validation.
        #   when xs and ys are empty
        with self.assertRaises(ValueError):
            approximation.lagrange_interpolate([], [])
        #   when xs and ys are different lengths.
        with self.assertRaises(ValueError):
            approximation.lagrange_interpolate([], [0.])
        with self.assertRaises(ValueError):
            approximation.lagrange_interpolate([0.], [])
        with self.assertRaises(ValueError):
            approximation.lagrange_interpolate([0.], [1., 2., 3.])
        with self.assertRaises(ValueError):
            approximation.lagrange_interpolate([1., 2., 3.], [0.])

        # Test interpolating points that yield full-degree polynomials.
        grid = np.linspace(-10., 10., num=100)
        for xs, ys, p in [
                # 1 point
                ([0.], [-1.], lambda x: np.full_like(x, -1.)),
                ([0.], [ 1.], lambda x: np.full_like(x,  1.)),
                # 2 points
                ([-1., 1.], [ 1., -1.], lambda x: -(  x    )),
                ([-1., 1.], [ 0., -2.], lambda x: -(  x + 1)),
                ([-1., 1.], [ 2., -2.], lambda x: -(2*x    )),
                ([-1., 1.], [ 1., -3.], lambda x: -(2*x + 1)),
                ([-1., 1.], [-1.,  1.], lambda x:  (  x    )),
                ([-1., 1.], [ 0.,  2.], lambda x:  (  x + 1)),
                ([-1., 1.], [-2.,  2.], lambda x:  (2*x    )),
                ([-1., 1.], [-1.,  3.], lambda x:  (2*x + 1)),
                # 3 points
                ([-1., 0., 1.], [-1.,  0., -1.], lambda x: -(  x**2        )),
                ([-1., 0., 1.], [-2., -1., -2.], lambda x: -(  x**2     + 1)),
                ([-1., 0., 1.], [ 0.,  0., -2.], lambda x: -(  x**2 + x    )),
                ([-1., 0., 1.], [-1., -1., -3.], lambda x: -(  x**2 + x + 1)),
                ([-1., 0., 1.], [-2.,  0., -2.], lambda x: -(2*x**2        )),
                ([-1., 0., 1.], [-3., -1., -3.], lambda x: -(2*x**2     + 1)),
                ([-1., 0., 1.], [-1.,  0., -3.], lambda x: -(2*x**2 + x    )),
                ([-1., 0., 1.], [-2., -1., -4.], lambda x: -(2*x**2 + x + 1)),
                ([-1., 0., 1.], [ 1.,  0.,  1.], lambda x:  (  x**2        )),
                ([-1., 0., 1.], [ 2.,  1.,  2.], lambda x:  (  x**2     + 1)),
                ([-1., 0., 1.], [ 0.,  0.,  2.], lambda x:  (  x**2 + x    )),
                ([-1., 0., 1.], [ 1.,  1.,  3.], lambda x:  (  x**2 + x + 1)),
                ([-1., 0., 1.], [ 2.,  0.,  2.], lambda x:  (2*x**2        )),
                ([-1., 0., 1.], [ 3.,  1.,  3.], lambda x:  (2*x**2     + 1)),
                ([-1., 0., 1.], [ 1.,  0.,  3.], lambda x:  (2*x**2 + x    )),
                ([-1., 0., 1.], [ 2.,  1.,  4.], lambda x:  (2*x**2 + x + 1)),
        ]:
            interpolant = approximation.lagrange_interpolate(xs, ys)
            self.assertTrue(np.allclose(interpolant(xs), ys))
            self.assertTrue(np.allclose(interpolant(grid), p(grid)))

        # Test interpolating points that yield degenerate polynomials.
        grid = np.linspace(-10., 10., num=100)
        for n_deg in range(5):
            for n_extra_points in range(1, 6):
                def p(xs, n_deg=n_deg): return xs**n_deg

                xs = np.arange(n_deg + 1 + n_extra_points)
                ys = p(xs)

                interpolant = approximation.lagrange_interpolate(xs, ys)
                self.assertTrue(np.allclose(interpolant(xs), ys))
                self.assertTrue(np.allclose(interpolant(grid), p(grid)))

    def test_lagrange_interpolate_is_permutation_invariant(self):
        grid = np.linspace(-10., 10., num=100)
        for xs0 in itertools.permutations([-1., 0., 1.]):
            for xs1 in itertools.permutations([-1., 0., 1.]):
                if xs0 == xs1:
                    continue

                for f in [np.abs, np.exp, np.sin]:
                    interpolant0 =\
                        approximation.lagrange_interpolate(xs0, f(xs0))
                    interpolant1 =\
                        approximation.lagrange_interpolate(xs1, f(xs1))
                    self.assertTrue(np.allclose(
                        interpolant0(grid),
                        interpolant1(grid),
                    ))
