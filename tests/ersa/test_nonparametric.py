"""Tests for ersa.nonparametric"""

import unittest

import numpy as np
import pytest
from scipy import stats

from ersa import nonparametric, utils


class EmpiricalDistributionTestCase(unittest.TestCase):
    """Test ersa.nonparametric.EmpiricalDistribution."""

    def test_sample(self):
        # Test without weights.
        #   when len(ys) == 1
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        values = np.unique(dist.sample(10))
        self.assertEqual(values.tolist(), [42.])
        #   when len(ys) == 2
        ys = [1., 42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        _, freqs = np.unique(dist.sample(2_500), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.allclose(freqs, [0.5, 0.5], atol=0.05))
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        _, freqs = np.unique(dist.sample(2_500), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.allclose(freqs, [0.2] * 5, atol=0.05))
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        _, freqs = np.unique(dist.sample((500, 5)), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.allclose(freqs, [0.2] * 5, atol=0.05))

        # Test with weights.
        #   when len(ys) == 1
        ys = [42.]
        ws = [1.]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        values = np.unique(dist.sample(10))
        self.assertEqual(values.tolist(), [42.])
        #   when len(ys) == 2
        ys = [1., 42.]
        ws = [0.1, 0.9]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        _, freqs = np.unique(dist.sample(2_500), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.allclose(freqs, ws, atol=0.05))
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        _, freqs = np.unique(dist.sample(2_500), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.allclose(freqs, ws, atol=0.05))
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        _, freqs = np.unique(dist.sample((500, 5)), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.allclose(freqs, ws, atol=0.05))

    def test_pmf(self):
        # Test without weights.
        #   when len(ys) == 1
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(dist.pmf(42.), 1.)
        self.assertEqual(dist.pmf(0.), 0.)
        self.assertEqual(dist.pmf(42. - 1e-10), 0.)
        self.assertEqual(dist.pmf(42. + 1e-10), 0.)
        #   when len(ys) == 2
        ys = [1., 42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(dist.pmf(ys).tolist(), [1./len(ys)] * len(ys))
        self.assertEqual(
            dist.pmf([0., 1., 42., -1.]).tolist(),
            [0., 0.5, 0.5, 0.],
        )
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(dist.pmf(ys).tolist(), [1./len(ys)] * len(ys))
        self.assertEqual(
            dist.pmf([0., 1., 42., -1.]).tolist(),
            [0., 0.2, 0.2, 0.],
        )
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(
            dist.pmf([[0., 1.], [42., -1.]]).tolist(),
            [[0., 0.2], [0.2, 0.]],
        )

        # Test with weights.
        #   when len(ys) == 1
        ys = [42.]
        ws = [1.]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(dist.pmf(42.), 1.)
        self.assertEqual(dist.pmf(0.), 0.)
        self.assertEqual(dist.pmf(42. - 1e-10), 0.)
        self.assertEqual(dist.pmf(42. + 1e-10), 0.)
        #   when len(ys) == 2
        ys = [1., 42.]
        ws = [0.1, 0.9]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(dist.pmf(ys).tolist(), ws)
        self.assertEqual(
            dist.pmf([0., 1., 42., -1.]).tolist(),
            [0., 0.1, 0.9, 0.],
        )
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(dist.pmf(ys).tolist(), ws)
        self.assertEqual(
            dist.pmf([0., 1., 42., -1.]).tolist(),
            [0., 0.2, 0.3, 0.],
        )
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(
            dist.pmf([[0., 1.], [42., -1.]]).tolist(),
            [[0., 0.2], [0.3, 0.]],
        )

    def test_cdf(self):
        # Test without weights.
        #   when len(ys) == 1
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(dist.cdf(42.), 1.)
        self.assertEqual(dist.cdf(0.), 0.)
        self.assertEqual(dist.cdf(42. - 1e-10), 0.)
        self.assertEqual(dist.cdf(42. + 1e-10), 1.)
        #   when len(ys) == 2
        ys = [1., 42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(
            dist.cdf(ys).tolist(),
            [(i+1) / len(ys) for i in range(len(ys))],
        )
        self.assertEqual(
            dist.cdf([0., 1., 2., 42., -1.]).tolist(),
            [0., 0.5, 0.5, 1., 0.],
        )
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertTrue(np.allclose(
            dist.cdf(ys),
            [(i+1) / len(ys) for i in range(len(ys))],
        ))
        self.assertTrue(np.allclose(
            dist.cdf([-50, 0., 1., 42., 10_000.]),
            [0., 0.2, 0.4, 0.6, 1.],
        ))
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertTrue(np.allclose(
            dist.cdf([[0., 1.], [42., -1.]]),
            [[0.2, 0.4], [0.6, 0.2]],
        ))

        # Test without weights.
        #   when len(ys) == 1
        ys = [42.]
        ws = [1.]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(dist.cdf(42.), 1.)
        self.assertEqual(dist.cdf(0.), 0.)
        self.assertEqual(dist.cdf(42. - 1e-10), 0.)
        self.assertEqual(dist.cdf(42. + 1e-10), 1.)
        #   when len(ys) == 2
        ys = [1., 42.]
        ws = [0.1, 0.9]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(
            dist.cdf(ys).tolist(),
            [0.1, 1.],
        )
        self.assertEqual(
            dist.cdf([0., 1., 2., 42., -1.]).tolist(),
            [0., 0.1, 0.1, 1., 0.],
        )
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertTrue(np.allclose(
            dist.cdf(ys),
            [0.1, 0.3, 0.6, 0.75, 1.],
        ))
        self.assertTrue(np.allclose(
            dist.cdf([-50, 0., 1., 42., 10_000.]),
            [0., 0.1, 0.3, 0.6, 1.],
        ))
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertTrue(np.allclose(
            dist.cdf([[0., 1.], [42., -1.]]),
            [[0.1, 0.3], [0.6, 0.1]],
        ))

    def test_ppf(self):
        # Test without weights.
        #   when len(ys) == 1
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(dist.ppf(0.), -np.inf)
        self.assertEqual(dist.ppf(0.5), 42.)
        self.assertEqual(dist.ppf(1.), 42.)
        #   when len(ys) == 2
        ys = [1., 42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(
            dist.ppf([(i+1) / len(ys) for i in range(len(ys))]).tolist(),
            ys,
        )
        self.assertEqual(
            dist.ppf([0., 0.5, 1., 0.75]).tolist(),
            [-np.inf, 1., 42., 42.],
        )
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(
            dist.ppf([(i+1) / len(ys) for i in range(len(ys))]).tolist(),
            ys,
        )
        self.assertEqual(
            dist.ppf([0., 0.25, 1., 0.4]).tolist(),
            [-np.inf, 1., 1_000, 1.],
        )
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(
            dist.ppf([[0.2, 0.4], [0.6, 0.2]]).tolist(),
            [[-42, 1.], [42., -42.]],
        )

        # Test with weights.
        #   when len(ys) == 1
        ys = [42.]
        ws = [1.]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(dist.ppf(0.), -np.inf)
        self.assertEqual(dist.ppf(0.5), 42.)
        self.assertEqual(dist.ppf(1.), 42.)
        #   when len(ys) == 2
        ys = [1., 42.]
        ws = [0.1, 0.9]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(
            dist.ppf([0.1, 1.]).tolist(),
            ys,
        )
        self.assertEqual(
            dist.ppf([0., 0.05, 1., 0.25]).tolist(),
            [-np.inf, 1., 42., 42.],
        )
        #   when len(ys) > 2
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(
            dist.ppf([0.1, 0.3, 0.6, 0.75, 1.]).tolist(),
            ys,
        )
        self.assertEqual(
            dist.ppf([0., 0.1, 1., 0.4]).tolist(),
            [-np.inf, -42., 1_000, 42.],
        )
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(
            dist.ppf([[0.2, 0.4], [0.6, 0.1]]).tolist(),
            [[1., 42.], [42., -42.]],
        )

    def test_quantile_tuning_curve(self):
        for quantile in [0.25, 0.5, 0.75]:
            for use_weights in [False, True]:
                ys = [42.]
                ws = [1.] if use_weights else None
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                # Test when len(ys) == 1.
                self.assertEqual(
                    dist.quantile_tuning_curve(1, q=quantile),
                    42.,
                )
                self.assertEqual(
                    dist.quantile_tuning_curve(10, q=quantile),
                    42.,
                )
                self.assertEqual(
                    dist.quantile_tuning_curve([1, 10], q=quantile).tolist(),
                    [42., 42.],
                )
                self.assertEqual(
                    dist.quantile_tuning_curve([
                        [1, 10],
                        [10, 1],
                    ], q=quantile).tolist(),
                    [
                        [42., 42.],
                        [42., 42.],
                    ],
                )

                ys = [0., 50., 25., 100., 75.]
                ws = (
                    np.random.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                curve = np.quantile(
                    np.maximum.accumulate(
                        np.random.choice(ys, p=ws, size=(10_000, 7), replace=True),
                        axis=1,
                    ),
                    quantile,
                    method='inverted_cdf',
                    axis=0,
                )
                # Test 0 < ns <= len(ys).
                #   scalar
                self.assertAlmostEqual(
                    dist.quantile_tuning_curve(1, q=quantile),
                    curve[0],
                    delta=25.,
                )
                self.assertAlmostEqual(
                    dist.quantile_tuning_curve(3, q=quantile),
                    curve[2],
                    delta=25.,
                )
                self.assertAlmostEqual(
                    dist.quantile_tuning_curve(5, q=quantile),
                    curve[4],
                    delta=25.,
                )
                #   1D array
                self.assertTrue(np.allclose(
                    dist.quantile_tuning_curve([1, 2, 3, 4, 5], q=quantile),
                    curve[:5],
                    atol=25.,
                ))
                self.assertTrue(np.allclose(
                    dist.quantile_tuning_curve([2, 4, 1, 3, 5], q=quantile),
                    [curve[1], curve[3], curve[0], curve[2], curve[4]],
                    atol=25.,
                ))
                #   2D array
                self.assertTrue(np.allclose(
                    dist.quantile_tuning_curve([
                        [1, 2, 3, 4, 5],
                        [2, 4, 1, 3, 5],
                    ], q=quantile),
                    [
                        curve[:5],
                        [curve[1], curve[3], curve[0], curve[2], curve[4]],
                    ],
                    atol=25.,
                ))
                # Test ns > len(ys).
                #   scalar
                self.assertAlmostEqual(
                    dist.quantile_tuning_curve(6, q=quantile),
                    curve[5],
                    delta=25.,
                )
                self.assertAlmostEqual(
                    dist.quantile_tuning_curve(7, q=quantile),
                    curve[6],
                    delta=25.,
                )
                #   1D array
                self.assertTrue(np.allclose(
                    dist.quantile_tuning_curve([1, 2, 7], q=quantile),
                    [curve[0], curve[1], curve[6]],
                    atol=25.,
                ))
                #   2D array
                self.assertTrue(np.allclose(
                    dist.quantile_tuning_curve([
                        [1, 2, 7],
                        [6, 2, 1],
                    ], q=quantile),
                    [
                        [curve[0], curve[1], curve[6]],
                        [curve[5], curve[1], curve[0]],
                    ],
                    atol=25.,
                ))
                # Test ns <= 0.
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve(0, q=quantile)
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve(-1, q=quantile)
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve([0], q=quantile)
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve([-2], q=quantile)
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve([0, 1], q=quantile)
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve([-2, 1], q=quantile)
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve([[0], [1]], q=quantile)
                with self.assertRaises(ValueError):
                    dist.quantile_tuning_curve([[-2], [1]], q=quantile)

    def test_average_tuning_curve(self):
        for use_weights in [False, True]:
            ys = [42.]
            ws = [1.] if use_weights else None
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
            # Test when len(ys) == 1.
            self.assertEqual(dist.average_tuning_curve(1), 42.)
            self.assertEqual(dist.average_tuning_curve(10), 42.)
            self.assertEqual(
                dist.average_tuning_curve([1, 10]).tolist(),
                [42., 42.],
            )
            self.assertEqual(
                dist.average_tuning_curve([
                    [1, 10],
                    [10, 1],
                ]).tolist(),
                [
                    [42., 42.],
                    [42., 42.],
                ],
            )

            ys = [0., 50., 25., 100., 75.]
            ws = np.random.dirichlet(np.ones_like(ys)) if use_weights else None
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
            curve = np.mean(
                np.maximum.accumulate(
                    np.random.choice(ys, p=ws, size=(2_500, 7), replace=True),
                    axis=1,
                ),
                axis=0,
            )
            # Test 0 < ns <= len(ys).
            #   scalar
            self.assertAlmostEqual(
                dist.average_tuning_curve(1),
                curve[0],
                delta=5.,
            )
            self.assertAlmostEqual(
                dist.average_tuning_curve(3),
                curve[2],
                delta=5.,
            )
            self.assertAlmostEqual(
                dist.average_tuning_curve(5),
                curve[4],
                delta=5.,
            )
            #   1D array
            self.assertTrue(np.allclose(
                dist.average_tuning_curve([1, 2, 3, 4, 5]),
                curve[:5],
                atol=5.,
            ))
            self.assertTrue(np.allclose(
                dist.average_tuning_curve([2, 4, 1, 3, 5]),
                [curve[1], curve[3], curve[0], curve[2], curve[4]],
                atol=5.,
            ))
            #   2D array
            self.assertTrue(np.allclose(
                dist.average_tuning_curve([
                    [1, 2, 3, 4, 5],
                    [2, 4, 1, 3, 5],
                ]),
                [
                    curve[:5],
                    [curve[1], curve[3], curve[0], curve[2], curve[4]],
                ],
                atol=5.,
            ))
            # Test ns > len(ys).
            #   scalar
            self.assertAlmostEqual(
                dist.average_tuning_curve(6),
                curve[5],
                delta=5.,
            )
            self.assertAlmostEqual(
                dist.average_tuning_curve(7),
                curve[6],
                delta=5.,
            )
            #   1D array
            self.assertTrue(np.allclose(
                dist.average_tuning_curve([1, 2, 7]),
                [curve[0], curve[1], curve[6]],
                atol=5.,
            ))
            #   2D array
            self.assertTrue(np.allclose(
                dist.average_tuning_curve([
                    [1, 2, 7],
                    [6, 2, 1],
                ]),
                [
                    [curve[0], curve[1], curve[6]],
                    [curve[5], curve[1], curve[0]],
                ],
                atol=5.,
            ))
            # Test ns <= 0.
            with self.assertRaises(ValueError):
                dist.average_tuning_curve(0)
            with self.assertRaises(ValueError):
                dist.average_tuning_curve(-1)
            with self.assertRaises(ValueError):
                dist.average_tuning_curve([0])
            with self.assertRaises(ValueError):
                dist.average_tuning_curve([-2])
            with self.assertRaises(ValueError):
                dist.average_tuning_curve([0, 1])
            with self.assertRaises(ValueError):
                dist.average_tuning_curve([-2, 1])
            with self.assertRaises(ValueError):
                dist.average_tuning_curve([[0], [1]])
            with self.assertRaises(ValueError):
                dist.average_tuning_curve([[-2], [1]])

    def test_naive_tuning_curve(self):
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        # Test when len(ys) == 1.
        self.assertEqual(dist.naive_tuning_curve(1), 42.)
        self.assertEqual(dist.naive_tuning_curve(10), 42.)
        self.assertEqual(
            dist.naive_tuning_curve([1, 10]).tolist(),
            [42., 42.],
        )
        self.assertEqual(
            dist.naive_tuning_curve([
                [1, 10],
                [10, 1],
            ]).tolist(),
            [
                [42., 42.],
                [42., 42.],
            ],
        )

        ys = [0., 50., 25., 100., 75.]
        dist = nonparametric.EmpiricalDistribution(ys)
        # Test 0 < ns <= len(ys).
        #   scalar
        self.assertEqual(dist.naive_tuning_curve(1), 0.)
        self.assertEqual(dist.naive_tuning_curve(3), 50.)
        self.assertEqual(dist.naive_tuning_curve(5), 100.)
        #   1D array
        self.assertEqual(
            dist.naive_tuning_curve([1, 2, 3, 4, 5]).tolist(),
            [0., 50., 50., 100., 100.],
        )
        self.assertEqual(
            dist.naive_tuning_curve([2, 4, 1, 3, 5]).tolist(),
            [50., 100., 0., 50., 100.],
        )
        #   2D array
        self.assertEqual(
            dist.naive_tuning_curve([
                [1, 2, 3, 4, 5],
                [2, 4, 1, 3, 5],
            ]).tolist(),
            [
                [0., 50., 50., 100., 100.],
                [50., 100., 0., 50., 100.],
            ],
        )
        # Test ns > len(ys).
        #   scalar
        self.assertEqual(dist.naive_tuning_curve(6), 100.)
        self.assertEqual(dist.naive_tuning_curve(10), 100.)
        #   1D array
        self.assertEqual(
            dist.naive_tuning_curve([1, 2, 10]).tolist(),
            [0., 50., 100.],
        )
        #   2D array
        self.assertEqual(
            dist.naive_tuning_curve([
                [1, 2, 10],
                [10, 2, 1],
            ]).tolist(),
            [
                [0., 50., 100.],
                [100., 50., 0.],
            ],
        )
        # Test ns <= 0.
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve(0)
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve(-1)
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([0])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([-2])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([0, 1])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([-2, 1])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([[0], [1]])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([[-2], [1]])

        ys = [-1, 0, 1]
        ws = [0.1, 0.5, 0.4]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        # Test when ws != None.
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve(1)
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([1])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([2, 1])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([[2], [1]])

    def test_v_tuning_curve(self):
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        # Test when len(ys) == 1.
        self.assertEqual(dist.v_tuning_curve(1), 42.)
        self.assertEqual(dist.v_tuning_curve(10), 42.)
        self.assertEqual(
            dist.v_tuning_curve([1, 10]).tolist(),
            [42., 42.],
        )
        self.assertEqual(
            dist.v_tuning_curve([
                [1, 10],
                [10, 1],
            ]).tolist(),
            [
                [42., 42.],
                [42., 42.],
            ],
        )

        ys = [0., 50., 25., 100., 75.]
        dist = nonparametric.EmpiricalDistribution(ys)
        curve = np.mean(
            np.maximum.accumulate(
                np.random.choice(ys, size=(2_500, 7), replace=True),
                axis=1,
            ),
            axis=0,
        )
        # Test 0 < ns <= len(ys).
        #   scalar
        self.assertAlmostEqual(dist.v_tuning_curve(1), curve[0], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(3), curve[2], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(5), curve[4], delta=5.)
        #   1D array
        self.assertTrue(np.allclose(
            dist.v_tuning_curve([1, 2, 3, 4, 5]),
            curve[:5],
            atol=5.,
        ))
        self.assertTrue(np.allclose(
            dist.v_tuning_curve([2, 4, 1, 3, 5]),
            [curve[1], curve[3], curve[0], curve[2], curve[4]],
            atol=5.,
        ))
        #   2D array
        self.assertTrue(np.allclose(
            dist.v_tuning_curve([
                [1, 2, 3, 4, 5],
                [2, 4, 1, 3, 5],
            ]),
            [
                curve[:5],
                [curve[1], curve[3], curve[0], curve[2], curve[4]],
            ],
            atol=5.,
        ))
        # Test ns > len(ys).
        #   scalar
        self.assertAlmostEqual(dist.v_tuning_curve(6), curve[5], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(7), curve[6], delta=5.)
        #   1D array
        self.assertTrue(np.allclose(
            dist.v_tuning_curve([1, 2, 7]),
            [curve[0], curve[1], curve[6]],
            atol=5.,
        ))
        #   2D array
        self.assertTrue(np.allclose(
            dist.v_tuning_curve([
                [1, 2, 7],
                [6, 2, 1],
            ]),
            [
                [curve[0], curve[1], curve[6]],
                [curve[5], curve[1], curve[0]],
            ],
            atol=5.,
        ))
        # Test ns <= 0.
        with self.assertRaises(ValueError):
            dist.v_tuning_curve(0)
        with self.assertRaises(ValueError):
            dist.v_tuning_curve(-1)
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([0])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([-2])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([0, 1])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([-2, 1])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([[0], [1]])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([[-2], [1]])

        ys = [-1, 0, 1]
        ws = [0.1, 0.5, 0.4]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        # Test when ws != None.
        with self.assertRaises(ValueError):
            dist.v_tuning_curve(1)
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([1])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([2, 1])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([[2], [1]])

    def test_u_tuning_curve(self):
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)
        # Test when len(ys) == 1.
        self.assertEqual(dist.u_tuning_curve(1), 42.)
        self.assertEqual(dist.u_tuning_curve(10), 42.)
        self.assertEqual(
            dist.u_tuning_curve([1, 10]).tolist(),
            [42., 42.],
        )
        self.assertEqual(
            dist.u_tuning_curve([
                [1, 10],
                [10, 1],
            ]).tolist(),
            [
                [42., 42.],
                [42., 42.],
            ],
        )

        ys = [0., 50., 25., 100., 75.]
        dist = nonparametric.EmpiricalDistribution(ys)
        curve = np.mean(
            np.maximum.accumulate(
                # Sort random numbers to batch sampling without replacement.
                np.array(ys)[np.argsort(np.random.rand(2_500, 5), axis=1)],
                axis=1,
            ),
            axis=0,
        )
        # Test 0 < ns <= len(ys).
        #   scalar
        self.assertAlmostEqual(dist.u_tuning_curve(1), curve[0], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(3), curve[2], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(5), curve[4], delta=5.)
        #   1D array
        self.assertTrue(np.allclose(
            dist.u_tuning_curve([1, 2, 3, 4, 5]),
            curve,
            atol=5.,
        ))
        self.assertTrue(np.allclose(
            dist.u_tuning_curve([2, 4, 1, 3, 5]),
            [curve[1], curve[3], curve[0], curve[2], curve[4]],
            atol=5.,
        ))
        #   2D array
        self.assertTrue(np.allclose(
            dist.u_tuning_curve([
                [1, 2, 3, 4, 5],
                [2, 4, 1, 3, 5],
            ]),
            [
                curve,
                [curve[1], curve[3], curve[0], curve[2], curve[4]],
            ],
            atol=5.,
        ))
        # Test ns > len(ys).
        #   scalar
        self.assertAlmostEqual(dist.u_tuning_curve(6), curve[4], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(7), curve[4], delta=5.)
        #   1D array
        self.assertTrue(np.allclose(
            dist.u_tuning_curve([1, 2, 7]),
            [curve[0], curve[1], curve[4]],
            atol=5.,
        ))
        #   2D array
        self.assertTrue(np.allclose(
            dist.u_tuning_curve([
                [1, 2, 7],
                [6, 2, 1],
            ]),
            [
                [curve[0], curve[1], curve[4]],
                [curve[4], curve[1], curve[0]],
            ],
            atol=5.,
        ))
        # Test ns <= 0.
        with self.assertRaises(ValueError):
            dist.u_tuning_curve(0)
        with self.assertRaises(ValueError):
            dist.u_tuning_curve(-1)
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([0])
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([-2])
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([0, 1])
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([-2, 1])
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([[0], [1]])
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([[-2], [1]])

        ys = [-1, 0, 1]
        ws = [0.1, 0.5, 0.4]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        # Test when ws != None.
        with self.assertRaises(ValueError):
            dist.u_tuning_curve(1)
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([1])
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([2, 1])
        with self.assertRaises(ValueError):
            dist.u_tuning_curve([[2], [1]])

    @pytest.mark.level(2)
    def test_confidence_bands(self):
        n = 10
        for method in ['dkw', 'ks', 'beta_ppf', 'beta_hpd']:
            for confidence in [0.25, 0.5, 0.75]:
                ys = np.random.uniform(0, 1, size=n)
                lo, dist, hi =\
                    nonparametric.EmpiricalDistribution.confidence_bands(
                        ys, confidence, method=method,
                    )
                # Check that dist is the empirical distribution.
                self.assertEqual(
                    dist.cdf(ys).tolist(),
                    nonparametric.EmpiricalDistribution(ys).cdf(ys).tolist(),
                )
                # Check bounded below by 0.
                self.assertGreaterEqual(np.min(lo.cdf(ys)), 0. - 1e-15)
                self.assertGreaterEqual(np.min(dist.cdf(ys)), 0. - 1e-15)
                self.assertGreaterEqual(np.min(hi.cdf(ys)), 0. - 1e-15)
                # Check bounded above by 1.
                self.assertLessEqual(np.max(lo.cdf(ys)), 1. + 1e-15)
                self.assertLessEqual(np.max(dist.cdf(ys)), 1. + 1e-15)
                self.assertLessEqual(np.max(hi.cdf(ys)), 1. + 1e-15)
                # Check bands are proper distance from the empirical CDF.
                if method == 'dkw' or method == 'ks':
                    epsilon = (
                        utils.dkw_epsilon(n, confidence)
                        if method == 'dkw' else
                        stats.kstwo(n).ppf(confidence)
                    )
                    self.assertTrue(np.allclose(
                        (dist.cdf(ys) - lo.cdf(ys))[dist.cdf(ys) > epsilon],
                        epsilon,
                    ))
                    self.assertTrue(np.allclose(
                        (hi.cdf(ys) - dist.cdf(ys))[1. - dist.cdf(ys) > epsilon],
                        epsilon,
                    ))

    def test_ppf_is_almost_sure_left_inverse_of_cdf(self):
        # NOTE: In general, the quantile function is an almost sure left
        # inverse of the cumulative distribution function.
        dist = nonparametric.EmpiricalDistribution(
            np.random.uniform(0, 1, size=5),
            ws=np.random.dirichlet(np.ones(5)),
        )
        for _ in range(7):
            ys = dist.sample(100)
            self.assertEqual(dist.ppf(dist.cdf(ys)).tolist(), ys.tolist())

    def test_ppf_at_extreme_values(self):
        # Test when support is infinite.
        dist = nonparametric.EmpiricalDistribution([0.], ws=[1.])
        self.assertEqual(dist.ppf(0. - 1e-12), -np.inf)
        self.assertEqual(dist.ppf(0.), -np.inf)
        self.assertEqual(dist.ppf(1.), 0.)
        self.assertEqual(dist.ppf(1. + 1e-12), 0.)
        # Test when support is finite.
        dist = nonparametric.EmpiricalDistribution([0.], ws=[1.], a=-1., b=1.)
        self.assertEqual(dist.ppf(0. - 1e-12), -np.inf)
        self.assertEqual(dist.ppf(0.), -np.inf)
        self.assertEqual(dist.ppf(1.), 0.)
        self.assertEqual(dist.ppf(1. + 1e-12), 0.)

    def test_average_tuning_curve_with_probability_mass_at_infinity(self):
        for ys, expected in [
                ([-np.inf, -10., 0., 10.,   100.], -np.inf),
                ([  -100., -10., 0., 10., np.inf],  np.inf),
                ([-np.inf, -10., 0., 10., np.inf],  np.nan),
        ]:
            for use_weights in [False, True]:
                ws = (
                    np.random.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                # Test 0 < ns <= len(ys).
                #   scalar
                self.assertTrue(np.isclose(
                    dist.average_tuning_curve(1),
                    expected,
                    equal_nan=True,
                ))
                self.assertTrue(np.isclose(
                    dist.average_tuning_curve(3),
                    expected,
                    equal_nan=True,
                ))
                self.assertTrue(np.isclose(
                    dist.average_tuning_curve(5),
                    expected,
                    equal_nan=True,
                ))
                #   1D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve([1, 2, 3, 4, 5]),
                    expected,
                    equal_nan=True,
                ))
                #   2D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve([
                        [1, 2, 3, 4, 5],
                        [2, 4, 1, 3, 5],
                    ]),
                    expected,
                    equal_nan=True,
                ))
                # Test ns > len(ys).
                #   scalar
                self.assertTrue(np.isclose(
                    dist.average_tuning_curve(6),
                    expected,
                    equal_nan=True,
                ))
                self.assertTrue(np.isclose(
                    dist.average_tuning_curve(7),
                    expected,
                    equal_nan=True,
                ))
                #   1D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve([1, 2, 7]),
                    expected,
                    equal_nan=True,
                ))
                #   2D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve([
                        [1, 2, 7],
                        [6, 2, 1],
                    ]),
                    expected,
                    equal_nan=True,
                ))

    @pytest.mark.level(3)
    def test_dkw_bands_have_correct_coverage(self):
        n_trials = 2_500
        dist = stats.norm(0., 1.)
        grid = np.linspace(-5., 5., num=10_000)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [10, 100]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='dkw',
                        )
                    covered.append(
                        np.all(lo.cdf(grid) <= dist.cdf(grid))
                        & np.all(dist.cdf(grid) <= hi.cdf(grid))
                    )

                stderr = (confidence * (1. - confidence) / n_trials)**0.5
                self.assertGreater(np.mean(covered) + 6 * stderr, confidence)

    @pytest.mark.level(3)
    def test_ks_bands_have_correct_coverage(self):
        n_trials = 2_500
        dist = stats.norm(0., 1.)
        grid = np.linspace(-5., 5., num=10_000)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [10, 100]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='ks',
                        )
                    covered.append(
                        np.all(lo.cdf(grid) <= dist.cdf(grid))
                        & np.all(dist.cdf(grid) <= hi.cdf(grid))
                    )

                stderr = (confidence * (1. - confidence) / n_trials)**0.5
                self.assertAlmostEqual(
                    np.mean(covered),
                    confidence,
                    delta=6 * stderr,
                )

    @pytest.mark.level(3)
    def test_beta_ppf_bands_has_correct_coverage(self):
        n_trials = 2_500
        dist = stats.norm(0., 1.)
        grid = np.linspace(-5., 5., num=10_000)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [10, 100]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='beta_ppf',
                        )
                    covered.append(
                        np.all(lo.cdf(grid) <= dist.cdf(grid))
                        & np.all(dist.cdf(grid) <= hi.cdf(grid))
                    )

                stderr = (confidence * (1. - confidence) / n_trials)**0.5
                self.assertAlmostEqual(
                    np.mean(covered),
                    confidence,
                    delta=6 * stderr,
                )

    @pytest.mark.level(3)
    def test_beta_hpd_bands_has_correct_coverage(self):
        n_trials = 2_500
        dist = stats.norm(0., 1.)
        grid = np.linspace(-5., 5., num=10_000)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [10, 100]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='beta_hpd',
                        )
                    covered.append(
                        np.all(lo.cdf(grid) <= dist.cdf(grid))
                        & np.all(dist.cdf(grid) <= hi.cdf(grid))
                    )

                stderr = (confidence * (1. - confidence) / n_trials)**0.5
                self.assertAlmostEqual(
                    np.mean(covered),
                    confidence,
                    delta=6 * stderr,
                )
