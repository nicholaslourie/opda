"""Tests for opda.approximation."""

import itertools
import unittest
import warnings

import numpy as np
import pytest

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


class RemezTestCase(unittest.TestCase):
    """Test opda.approximation.remez."""

    def test_remez(self):
        # Test argument validation.
        #   when f is not a callable
        with self.assertRaises(TypeError):
            approximation.remez(None, 0., 1., 1)
        with self.assertRaises(TypeError):
            approximation.remez(1., 0., 1., 1)
        with self.assertRaises(TypeError):
            approximation.remez("f", 0., 1., 1)
        #   when a is not a scalar
        with self.assertRaises(ValueError):
            approximation.remez(np.exp, [0.], 1., 1)
        #   when b is not a scalar
        with self.assertRaises(ValueError):
            approximation.remez(np.exp, 0., [1.], 1)
        #   when n is not a scalar
        with self.assertRaises(ValueError):
            approximation.remez(np.exp, 0., 1., [1])
        #   when n is not an integer
        with self.assertRaises(ValueError):
            approximation.remez(np.exp, 0., 1., 1.5)
        #   when n is negative.
        with self.assertRaises(ValueError):
            approximation.remez(np.exp, 0., 1., -1)
        #   when atol is negative.
        with self.assertRaises(ValueError):
            approximation.remez(np.exp, 0., 1., 1, atol=-1e-5)
        #   when a > b
        with self.assertRaises(ValueError):
            approximation.remez(np.exp, 1., 0., 1)

        # Test remez recovers the minimax approximation to x**n.
        # NOTE: The minimax approximation of x**n by a polynomial of
        # degree less than n is well-known. For example, see:
        # https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html#eqn49.
        a, b = -1., 1.
        for n, expected_rs, expected_ys, expected_err in [
                # x**1 ~ 0
                (1, [-1., 1.],  [-1., 1.], 1.),
                # x**2 ~ 1/2
                (2, [-1., 0., 1.], [1., 0., 1.], 0.5),
                # x**3 ~ 3/4x
                (3, [-1., -0.5, 0.5, 1.], [-1., -0.125, 0.125, 1.], 0.25),
        ]:
            def f(xs, n=n): return xs**n

            rs, ys, err = approximation.remez(f, a, b, n-1)
            self.assertTrue(np.allclose(rs, expected_rs))
            self.assertTrue(np.allclose(ys, expected_ys))
            self.assertAlmostEqual(err, expected_err)

        # Test remez on degenerate polynomial approximations.
        a, b = -1., 1.
        for n in [0, 1, 2]:
            for extra_n in [0, 1, 2]:
                def f(xs, n=n): return xs**n

                rs, ys, err = approximation.remez(f, a, b, n + extra_n)

                self.assertLess(np.abs(err), 1e-15)

                # Construct the minimax polynomial approximation.
                # See minimax_polynomial_approximation for details.
                ns = np.arange(n + extra_n + 2)
                p0 = approximation.lagrange_interpolate(rs[:-1], ys[:-1])
                p1 = approximation.lagrange_interpolate(rs[:-1], (-1)**ns[:-1])
                h = (p0(rs[-1]) - ys[-1]) / (p1(rs[-1]) + (-1)**n)
                p = approximation.lagrange_interpolate(
                    rs[:-1],
                    ys[:-1] - h * (-1)**ns[:-1],
                )

                grid = np.linspace(a, b, num=1_000)
                self.assertTrue(np.allclose(f(grid), p(grid)))

    @pytest.mark.level(2)
    def test_remez_on_general_functions(self):
        atol = 256. * np.spacing(1.)
        for f, a, b in [
                ( np.abs,      -1.,      1.),
                (np.sqrt,       0.,      1.),
                ( np.exp,      -1.,      1.),
                ( np.sin, -2*np.pi, 2*np.pi),
        ]:
            for n in [0, 1, 2, 5, 15]:
                rs, ys, err = approximation.remez(f, a, b, n, atol=atol)

                # Construct the minimax polynomial approximation.
                # See minimax_polynomial_approximation for details.
                ns = np.arange(n + 2)
                p0 = approximation.lagrange_interpolate(rs[:-1], ys[:-1])
                p1 = approximation.lagrange_interpolate(rs[:-1], (-1)**ns[:-1])
                h = (p0(rs[-1]) - ys[-1]) / (p1(rs[-1]) + (-1)**n)
                p = approximation.lagrange_interpolate(
                    rs[:-1],
                    ys[:-1] - h * (-1)**ns[:-1],
                )
                errs = f(rs) - p(rs)

                # Check values returned by remez.
                #   rs
                #     error on rs should be *equal*
                self.assertTrue(np.allclose(np.abs(errs[0]), np.abs(errs)))
                #     error on rs should be *oscillating*
                self.assertTrue(np.array_equal(
                    np.sign(errs),
                    np.sign(errs[0]) * (-1)**ns,
                ))
                #     error on rs should be close to err.
                self.assertTrue(np.allclose(np.abs(errs), err, atol=atol))
                #   ys
                self.assertTrue(np.allclose(ys, f(rs)))
                #   err
                grid = np.linspace(a, b, num=1_000)
                self.assertGreater(err, 0.)
                self.assertGreater(
                    err,
                    np.max(np.abs(f(grid) - p(grid))) - atol,
                )


class MinimaxPolynomialApproximationTestCase(unittest.TestCase):
    """Test opda.approximation.minimax_polynomial_approximation."""

    def test_minimax_polynomial_approximation(self):
        # Test argument validation.
        #   when f is not a callable
        with self.assertRaises(TypeError):
            approximation.minimax_polynomial_approximation(
                None, 0., 1., 1,
            )
        with self.assertRaises(TypeError):
            approximation.minimax_polynomial_approximation(
                1., 0., 1., 1,
            )
        with self.assertRaises(TypeError):
            approximation.minimax_polynomial_approximation(
                "f", 0., 1., 1,
            )
        #   when a is not a scalar
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_approximation(
                np.exp, [0.], 1., 1,
            )
        #   when b is not a scalar
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_approximation(
                np.exp, 0., [1.], 1,
            )
        #   when n is not a scalar
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_approximation(
                np.exp, 0., 1., [1],
            )
        #   when n is not an integer
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_approximation(
                np.exp, 0., 1., 1.5,
            )
        #   when n is negative.
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_approximation(
                np.exp, 0., 1., -1,
            )
        #   when atol is negative.
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_approximation(
                np.exp, 0., 1., 1, atol=-1e-5,
            )
        #   when a > b
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_approximation(
                np.exp, 1., 0., 1,
            )

        # Test minimax_polynomial_approximation against x**n.
        # NOTE: The minimax approximation of x**n by a polynomial of
        # degree less than n is well-known. For example, see:
        # https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html#eqn49.
        a, b = -1., 1.
        for n, expected_p, expected_err in [
                # x**1 ~ 0
                (1, lambda x: np.full_like(x,  0.),     1.),
                # x**2 ~ 1/2
                (2, lambda x: np.full_like(x, 0.5),    0.5),
                # x**3 ~ 3/4x
                (3, lambda x:               0.75*x,   0.25),
                # x**4 ~ x**2 - 1/8
                (4, lambda x:         x**2 - 0.125,  0.125),
                # x**5 ~ 5/4x**3 - 5/16x
                (5, lambda x: 1.25*x**3 - 0.3125*x, 0.0625),
        ]:
            def f(xs, n=n): return xs**n

            # NOTE: The best polynomial approximation to x**n of degree
            # less than n has degree n-2. Thus, we can find the same
            # approximation when considering polynomials of both degree
            # n-2 and n-1.

            # degree n-1
            p, err = approximation.minimax_polynomial_approximation(
                f, a, b, n-1,
            )

            grid = np.linspace(a, b, num=1_000)
            self.assertTrue(np.allclose(p(grid), expected_p(grid)))
            self.assertAlmostEqual(err, expected_err)

            # degree n-2
            if n < 2:
                continue

            p, err = approximation.minimax_polynomial_approximation(
                f, a, b, n-2,
            )

            grid = np.linspace(a, b, num=1_000)
            self.assertTrue(np.allclose(p(grid), expected_p(grid)))
            self.assertAlmostEqual(err, expected_err)

        # Test minimax_polynomial_approximation on lower degree polynomials.
        a, b = -1., 1.
        for n in [0, 1, 2]:
            for extra_n in [0, 1, 2]:
                def f(xs, n=n): return xs**n

                p, err = approximation.minimax_polynomial_approximation(
                    f, a, b, n + extra_n,
                )

                self.assertLess(np.abs(err), 1e-15)

                grid = np.linspace(a, b, num=1_000)
                self.assertTrue(np.allclose(f(grid), p(grid)))

    @pytest.mark.level(2)
    def test_minimax_polynomial_approximation_on_general_functions(self):
        atol = 256. * np.spacing(1.)
        for f, a, b in [
                ( np.abs,      -1.,      1.),
                (np.sqrt,       0.,      1.),
                ( np.exp,      -1.,      1.),
                ( np.sin, -2*np.pi, 2*np.pi),
        ]:
            for n in [0, 1, 2, 5, 15]:
                p, err = approximation.minimax_polynomial_approximation(
                    f, a, b, n, atol=atol,
                )

                grid = np.linspace(a, b, num=1_000)

                # Check minimax is better than Chebyshev approximation.
                ns = np.arange(n + 1)
                xs = a + (b - a) * 0.5 * (1 - np.cos(np.pi * (ns + 0.5)/(n+1)))
                p_cheb = approximation.lagrange_interpolate(xs, f(xs))
                self.assertLess(
                    np.max(np.abs(f(grid) - p(grid))),
                    np.max(np.abs(f(grid) - p_cheb(grid))) + atol,
                )
                # Check that err bounds the maximum error.
                self.assertGreater(err, 0.)
                self.assertGreater(
                    err,
                    np.max(np.abs(f(grid) - p(grid))) - atol,
                )


class MinimaxPolynomialCoefficientsTestCase(unittest.TestCase):
    """Test opda.approximation.minimax_polynomial_coefficients."""

    def test_minimax_polynomial_coefficients(self):
        # Test argument validation.
        #   when f is not a callable
        with self.assertRaises(TypeError):
            approximation.minimax_polynomial_coefficients(
                None, 0., 1., 1,
            )
        with self.assertRaises(TypeError):
            approximation.minimax_polynomial_coefficients(
                1., 0., 1., 1,
            )
        with self.assertRaises(TypeError):
            approximation.minimax_polynomial_coefficients(
                "f", 0., 1., 1,
            )
        #   when a is not a scalar
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, [0.], 1., 1,
            )
        #   when b is not a scalar
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., [1.], 1,
            )
        #   when n is not a scalar
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., [1],
            )
        #   when n is not an integer
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., 1.5,
            )
        #   when n is negative.
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., -1,
            )
        #   when transform is not length 2
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., 1, transform=(),
            )
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., 1, transform=(1.,),
            )
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., 1, transform=(0., 1., 2.),
            )
        #   when transform bounds are in wrong order
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., 1, transform=(1., -1.),
            )
        #   when atol is negative.
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 0., 1., 1, atol=-1e-5,
            )
        #   when a > b
        with self.assertRaises(ValueError):
            approximation.minimax_polynomial_coefficients(
                np.exp, 1., 0., 1,
            )

        # Test minimax_polynomial_coefficients against x**n.
        # NOTE: The minimax approximation of x**n by a polynomial of
        # degree less than n is well-known. For example, see:
        # https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html#eqn49.
        a, b = -1., 1.
        for n, expected_coeffs, expected_err in [
                # x**1 ~ 0
                (1, (0.,),                           1.),
                # x**2 ~ 1/2
                (2, (0.5, 0.),                      0.5),
                # x**3 ~ 3/4x
                (3, (0., 0.75, 0.),                0.25),
                # x**4 ~ x**2 - 1/8
                (4, (-0.125, 0., 1., 0.),         0.125),
                # x**5 ~ 5/4x**3 - 5/16x
                (5, (0., -0.3125, 0., 1.25, 0.), 0.0625),
        ]:
            def f(xs, n=n): return xs**n

            # NOTE: The best polynomial approximation to x**n of degree
            # less than n has degree n-2. Thus, we can find the same
            # approximation when considering polynomials of both degree
            # n-2 and n-1.

            # degree n-1
            coeffs, err = approximation.minimax_polynomial_coefficients(
                f, a, b, n-1,
            )

            self.assertTrue(np.allclose(coeffs, expected_coeffs))
            self.assertAlmostEqual(err, expected_err)

            # degree n-2
            if n < 2:
                continue

            coeffs, err = approximation.minimax_polynomial_coefficients(
                f, a, b, n-2,
            )

            self.assertTrue(np.allclose(coeffs, expected_coeffs[:-1]))
            self.assertAlmostEqual(err, expected_err)

        # Test minimax_polynomial_coefficients on lower degree polynomials.
        a, b = -1., 1.
        for n in [0, 1, 2]:
            for extra_n in [0, 1, 2]:
                def f(xs, n=n): return xs**n

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"f can be approximated with error"
                                r" less than atol.",
                        category=RuntimeWarning,
                    )
                    coeffs, err = approximation.minimax_polynomial_coefficients(
                        f, a, b, n + extra_n,
                    )

                self.assertLess(np.abs(err), 1e-15)
                self.assertTrue(np.allclose(
                    coeffs,
                    np.array([
                        1. if i == n else 0.
                        for i in range(n + extra_n + 1)
                    ]),
                ))

    @pytest.mark.level(2)
    def test_minimax_polynomial_coefficients_on_general_functions(self):
        atol = 256. * np.spacing(1.)
        for f, a, b in [
                ( np.abs,      -1.,      1.),
                (np.sqrt,       0.,      1.),
                ( np.exp,      -1.,      1.),
                ( np.sin, -2*np.pi, 2*np.pi),
        ]:
            for n in [0, 1, 2, 5, 15]:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"f can be approximated with error"
                                r" less than atol.",
                        category=RuntimeWarning,
                    )
                    coeffs, err = approximation.minimax_polynomial_coefficients(
                        f, a, b, n, atol=atol,
                    )

                def p(xs, coeffs=coeffs):
                    # Evaluate the polynomial with coeffs as coefficients.
                    ys = np.zeros_like(xs)
                    for coeff in coeffs[::-1]:
                        ys *= xs
                        ys += coeff
                    return ys

                grid = np.linspace(a, b, num=1_000)

                # Check the coefficients give the minimax polynomial.
                p_minimax, _ = approximation.minimax_polynomial_approximation(
                    f, a, b, n, atol=atol,
                )
                self.assertTrue(np.allclose(p(grid), p_minimax(grid)))
                # Check that err bounds the maximum error.
                self.assertGreater(err, 0.)
                self.assertGreater(
                    # Computing the coefficients incurs some rounding
                    # error, so use a slightly increased number instead
                    # of err directly.
                    1.01 * err + atol,
                    np.max(np.abs(f(grid) - p(grid))),
                )


class PiecewisePolynomialKnotsTestCase(unittest.TestCase):
    """Test opda.approximation.piecewise_polynomial_knots."""

    def test_piecewise_polynomial_knots(self):
        # Test argument validation.
        #   when f is not a callable
        with self.assertRaises(TypeError):
            approximation.piecewise_polynomial_knots(
                None, 0., 1., [1, 2],
            )
        with self.assertRaises(TypeError):
            approximation.piecewise_polynomial_knots(
                1., 0., 1., [1, 2],
            )
        with self.assertRaises(TypeError):
            approximation.piecewise_polynomial_knots(
                "f", 0., 1., [1, 2],
            )
        #   when a is not a scalar
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, [0.], 1., [1, 2],
            )
        #   when b is not a scalar
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., [1.], [1, 2],
            )
        #   when ns is not a 1D array
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., 1,
            )
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [[1], [2]],
            )
        #   when ns is not all integers
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [1.5],
            )
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [1, 1.5],
            )
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [1.5, 1],
            )
        #   when ns is not all non-negative.
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [-1],
            )
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [-1, 1],
            )
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [1, -1],
            )
        #   when atol is negative.
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 0., 1., [1, 2], atol=-1e-5,
            )
        #   when a > b
        with self.assertRaises(ValueError):
            approximation.piecewise_polynomial_knots(
                np.exp, 1., 0., [1, 2],
            )

        # Test piecewise_polynomial_knots.

        f, a, b, ns = lambda xs: xs**2, -1., 1., [1, 1, 1]
        # A piecewise polynomial approximation should have error at most
        # that of the minimax polynomial.
        err_bound = 0.5  # minimax polynomial approximation error for x**2

        knots, err = approximation.piecewise_polynomial_knots(f, a, b, ns)

        # Check knots.
        self.assertEqual(knots[0], a)
        self.assertEqual(knots[-1], b)
        self.assertEqual(len(knots), len(ns) + 1)
        self.assertEqual(knots.tolist(), sorted(knots.tolist()))
        # Check err.
        self.assertGreater(err, 0)
        self.assertLess(err, err_bound)

    @pytest.mark.level(3)
    def test_piecewise_polynomial_knots_on_general_functions(self):
        for f, a, b, ns in [
                (lambda xs: xs**2, -1., 1., [1, 0, 1]),
                (          np.exp, -1., 1., [0, 1, 2]),
        ]:
            knots, err = approximation.piecewise_polynomial_knots(f, a, b, ns)

            _, _, err0 = approximation.remez(f, knots[0], knots[1], ns[0])
            _, _, err1 = approximation.remez(f, knots[1], knots[2], ns[1])
            _, _, err2 = approximation.remez(f, knots[2], knots[3], ns[2])

            # Overall error should equal the error on each piece.
            self.assertAlmostEqual(err, err0, delta=0.01 * err)
            self.assertAlmostEqual(err, err1, delta=0.01 * err)
            self.assertAlmostEqual(err, err2, delta=0.01 * err)
