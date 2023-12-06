"""Tests for opda.nonparametric"""

import unittest
import warnings

import numpy as np
import pytest
from scipy import stats

from opda import nonparametric, utils


class EmpiricalDistributionTestCase(unittest.TestCase):
    """Test opda.nonparametric.EmpiricalDistribution."""

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
        #   when ys has duplicates
        ys = [-42., -42., 1., 100., 1_000]
        dist = nonparametric.EmpiricalDistribution(ys)
        _, freqs = np.unique(dist.sample(2_500), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.isclose(freqs[0], 0.4, atol=0.05))
        self.assertTrue(np.allclose(freqs[1:], [0.2] * 3, atol=0.05))
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
        #   when ys has duplicates
        ys = [-42., -42., 1., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        _, freqs = np.unique(dist.sample(2_500), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.isclose(freqs[0], np.sum(ws[:2]), atol=0.05))
        self.assertTrue(np.allclose(freqs[1:], ws[2:], atol=0.05))
        #   when shape is 2D
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        _, freqs = np.unique(dist.sample((500, 5)), return_counts=True)
        freqs = freqs / 2_500.
        self.assertTrue(np.allclose(freqs, ws, atol=0.05))

    def test_pmf(self):
        for a, b in [(-1e4, 1e4), (-np.inf, np.inf), (None, None)]:
            # Test without weights.
            #   when len(ys) == 1
            ys = [42.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertEqual(dist.pmf(42.), 1.)
            self.assertEqual(dist.pmf(0.), 0.)
            self.assertEqual(dist.pmf(42. - 1e-10), 0.)
            self.assertEqual(dist.pmf(42. + 1e-10), 0.)
            #   when len(ys) == 2
            ys = [1., 42.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertEqual(dist.pmf(ys).tolist(), [1./len(ys)] * len(ys))
            self.assertEqual(
                dist.pmf([0., 1., 42., -1.]).tolist(),
                [0., 0.5, 0.5, 0.],
            )
            #   when len(ys) > 2
            ys = [-42., 1., 42., 100., 1_000.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertEqual(dist.pmf(ys).tolist(), [1./len(ys)] * len(ys))
            self.assertEqual(
                dist.pmf([0., 1., 42., -1.]).tolist(),
                [0., 0.2, 0.2, 0.],
            )
            #   when ys has duplicates
            ys = [-42., -42., 1., 100., 1_000.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertEqual(
                dist.pmf(ys).tolist(),
                [0.4, 0.4, 0.2, 0.2, 0.2],
            )
            self.assertEqual(
                dist.pmf([0., 1., -42., -1.]).tolist(),
                [0., 0.2, 0.4, 0.],
            )
            #   when shape is 2D
            ys = [-42., 1., 42., 100., 1_000.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertEqual(
                dist.pmf([[0., 1.], [42., -1.]]).tolist(),
                [[0., 0.2], [0.2, 0.]],
            )

            # Test with weights.
            #   when len(ys) == 1
            ys = [42.]
            ws = [1.]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertEqual(dist.pmf(42.), 1.)
            self.assertEqual(dist.pmf(0.), 0.)
            self.assertEqual(dist.pmf(42. - 1e-10), 0.)
            self.assertEqual(dist.pmf(42. + 1e-10), 0.)
            #   when len(ys) == 2
            ys = [1., 42.]
            ws = [0.1, 0.9]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertEqual(dist.pmf(ys).tolist(), ws)
            self.assertEqual(
                dist.pmf([0., 1., 42., -1.]).tolist(),
                [0., 0.1, 0.9, 0.],
            )
            #   when len(ys) > 2
            ys = [-42., 1., 42., 100., 1_000.]
            ws = [0.1, 0.2, 0.3, 0.15, 0.25]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertEqual(dist.pmf(ys).tolist(), ws)
            self.assertEqual(
                dist.pmf([0., 1., 42., -1.]).tolist(),
                [0., 0.2, 0.3, 0.],
            )
            #   when ys has duplicates
            ys = [-42., -42., 1., 100., 1_000.]
            ws = [0.1, 0.1, 0.4, 0.15, 0.25]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertEqual(
                dist.pmf(ys).tolist(),
                [0.2, 0.2, 0.4, 0.15, 0.25],
            )
            self.assertEqual(
                dist.pmf([0., 1., -42., -1.]).tolist(),
                [0., 0.4, 0.2, 0.],
            )
            #   when shape is 2D
            ys = [-42., 1., 42., 100., 1_000.]
            ws = [0.1, 0.2, 0.3, 0.15, 0.25]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertEqual(
                dist.pmf([[0., 1.], [42., -1.]]).tolist(),
                [[0., 0.2], [0.3, 0.]],
            )

            # Test outside of the distribution's support.
            ys = [-42., 1., 42., 100., 1_000.]
            for ws in [None, [0.1, 0.2, 0.3, 0.15, 0.25]]:
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
                if a is not None:
                    self.assertEqual(dist.pmf(a - 1e-10), np.array(0.))
                    self.assertEqual(dist.pmf(a - 10), np.array(0.))
                if b is not None:
                    self.assertEqual(dist.pmf(b + 1e-10), np.array(0.))
                    self.assertEqual(dist.pmf(b + 10), np.array(0.))

    def test_cdf(self):
        for a, b in [(-1e4, 1e4), (-np.inf, np.inf), (None, None)]:
            # Test without weights.
            #   when len(ys) == 1
            ys = [42.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertEqual(dist.cdf(42.), 1.)
            self.assertEqual(dist.cdf(0.), 0.)
            self.assertEqual(dist.cdf(42. - 1e-10), 0.)
            self.assertEqual(dist.cdf(42. + 1e-10), 1.)
            #   when len(ys) == 2
            ys = [1., 42.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
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
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertTrue(np.allclose(
                dist.cdf(ys),
                [(i+1) / len(ys) for i in range(len(ys))],
            ))
            self.assertTrue(np.allclose(
                dist.cdf([-50, 0., 1., 42., 10_000.]),
                [0., 0.2, 0.4, 0.6, 1.],
            ))
            #   when ys has duplicates
            ys = [-42., -42., 1., 100., 1_000.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertTrue(np.allclose(
                dist.cdf(ys),
                [0.4, 0.4, 0.6, 0.8, 1.0],
            ))
            self.assertTrue(np.allclose(
                dist.cdf([-50, 0., 1., 42., 10_000.]),
                [0., 0.4, 0.6, 0.6, 1.],
            ))
            #   when shape is 2D
            ys = [-42., 1., 42., 100., 1_000.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertTrue(np.allclose(
                dist.cdf([[0., 1.], [42., -1.]]),
                [[0.2, 0.4], [0.6, 0.2]],
            ))

            # Test without weights.
            #   when len(ys) == 1
            ys = [42.]
            ws = [1.]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertEqual(dist.cdf(42.), 1.)
            self.assertEqual(dist.cdf(0.), 0.)
            self.assertEqual(dist.cdf(42. - 1e-10), 0.)
            self.assertEqual(dist.cdf(42. + 1e-10), 1.)
            #   when len(ys) == 2
            ys = [1., 42.]
            ws = [0.1, 0.9]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
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
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertTrue(np.allclose(
                dist.cdf(ys),
                [0.1, 0.3, 0.6, 0.75, 1.],
            ))
            self.assertTrue(np.allclose(
                dist.cdf([-50, 0., 1., 42., 10_000.]),
                [0., 0.1, 0.3, 0.6, 1.],
            ))
            #   when ys has duplicates
            ys = [-42., -42., 1., 100., 1_000.]
            ws = [0.1, 0.1, 0.4, 0.15, 0.25]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertTrue(np.allclose(
                dist.cdf(ys),
                [0.2, 0.2, 0.6, 0.75, 1.],
            ))
            self.assertTrue(np.allclose(
                dist.cdf([-50, 0., 1., 42., 10_000.]),
                [0., 0.2, 0.6, 0.6, 1.],
            ))
            #   when shape is 2D
            ys = [-42., 1., 42., 100., 1_000.]
            ws = [0.1, 0.2, 0.3, 0.15, 0.25]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertTrue(np.allclose(
                dist.cdf([[0., 1.], [42., -1.]]),
                [[0.1, 0.3], [0.6, 0.1]],
            ))

            # Test outside of the distribution's support.
            ys = [-42., 1., 42., 100., 1_000.]
            for ws in [None, [0.1, 0.2, 0.3, 0.15, 0.25]]:
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
                if a is not None:
                    self.assertEqual(dist.cdf(a - 1e-10), 0.)
                    self.assertEqual(dist.cdf(a - 10), 0.)
                if b is not None:
                    self.assertEqual(dist.cdf(b + 1e-10), 1.)
                    self.assertEqual(dist.cdf(b + 10), 1.)

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
        #   when ys has duplicates
        ys = [-42., -42., 1., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertEqual(
            dist.ppf([0.4, 0.4, 0.6, 0.8, 1.]).tolist(),
            ys,
        )
        self.assertEqual(
            dist.ppf([0., 0.25, 1., 0.5]).tolist(),
            [-np.inf, -42., 1_000, 1.],
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
        #   when ys has duplicates
        ys = [-42., -42., 1., 100., 1_000.]
        ws = [0.1, 0.1, 0.4, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertEqual(
            dist.ppf([0.2, 0.2, 0.6, 0.75, 1.]).tolist(),
            ys,
        )
        self.assertEqual(
            dist.ppf([0., 0.2, 1., 0.6]).tolist(),
            [-np.inf, -42., 1_000, 1.],
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
        for use_weights in [False, True]:
            for quantile in [0.25, 0.5, 0.75]:
                for minimize in [False, True]:
                    # Test when len(ys) == 1.
                    ys = [42.]
                    ws = [1.] if use_weights else None
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)

                    self.assertEqual(
                        dist.quantile_tuning_curve(
                            1,
                            q=quantile,
                            minimize=minimize,
                        ),
                        42.,
                    )
                    self.assertEqual(
                        dist.quantile_tuning_curve(
                            10,
                            q=quantile,
                            minimize=minimize,
                        ),
                        42.,
                    )
                    self.assertEqual(
                        dist.quantile_tuning_curve(
                            [1, 10],
                            q=quantile,
                            minimize=minimize,
                        ).tolist(),
                        [42., 42.],
                    )
                    self.assertEqual(
                        dist.quantile_tuning_curve(
                            [
                                [1, 10],
                                [10, 1],
                            ],
                            q=quantile,
                            minimize=minimize,
                        ).tolist(),
                        [
                            [42., 42.],
                            [42., 42.],
                        ],
                    )

                    # Test when len(ys) > 1.
                    ys = [0., 50., 25., 100., 75.]
                    ws = (
                        np.random.dirichlet(np.full_like(ys, 5))
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                    curve = np.quantile(
                        np.minimum.accumulate(
                            np.random.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        )
                        if minimize else
                        np.maximum.accumulate(
                            np.random.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        ),
                        quantile,
                        method='inverted_cdf',
                        axis=0,
                    )
                    #   Test 0 < ns <= len(ys).
                    #     scalar
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            1,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[0],
                        delta=25.,
                    )
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            3,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[2],
                        delta=25.,
                    )
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            5,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[4],
                        delta=25.,
                    )
                    #     1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [1, 2, 3, 4, 5],
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[:5],
                        atol=25.,
                    ))
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [2, 4, 1, 3, 5],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [curve[1], curve[3], curve[0], curve[2], curve[4]],
                        atol=25.,
                    ))
                    #     2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [1, 2, 3, 4, 5],
                                [2, 4, 1, 3, 5],
                            ],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [
                            curve[:5],
                            [curve[1], curve[3], curve[0], curve[2], curve[4]],
                        ],
                        atol=25.,
                    ))
                    #   Test ns > len(ys).
                    #     scalar
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            6,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve
                        [5],
                        delta=25.,
                    )
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            7,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[6],
                        delta=25.,
                    )
                    #     1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [1, 2, 7],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [curve[0], curve[1], curve[6]],
                        atol=25.,
                    ))
                    #     2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [1, 2, 7],
                                [6, 2, 1],
                            ],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [
                            [curve[0], curve[1], curve[6]],
                            [curve[5], curve[1], curve[0]],
                        ],
                        atol=25.,
                    ))
                    #   Test ns <= 0.
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            0,
                            q=quantile,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            -1,
                            q=quantile,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [0],
                            q=quantile,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [-2],
                            q=quantile,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [0, 1],
                            q=quantile,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [-2, 1],
                            q=quantile,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [[0], [1]],
                            q=quantile,
                            minimize=minimize,
                        )
                    with self.assertRaises(ValueError):
                        dist.quantile_tuning_curve(
                            [[-2], [1]],
                            q=quantile,
                            minimize=minimize,
                        )

                    # Test when ys has duplicates.
                    ys = [0., 0., 50., 0., 25.]
                    ws = (
                        np.random.dirichlet(np.full_like(ys, 5))
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                    curve = np.quantile(
                        np.minimum.accumulate(
                            np.random.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        )
                        if minimize else
                        np.maximum.accumulate(
                            np.random.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        ),
                        quantile,
                        method='inverted_cdf',
                        axis=0,
                    )
                    #   Test 0 < ns <= len(ys).
                    #     scalar
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            1,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[0],
                        delta=25.,
                    )
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            3,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[2],
                        delta=25.,
                    )
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            5,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[4],
                        delta=25.,
                    )
                    #     1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [1, 2, 3, 4, 5],
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[:5],
                        atol=25.,
                    ))
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [2, 4, 1, 3, 5],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [curve[1], curve[3], curve[0], curve[2], curve[4]],
                        atol=25.,
                    ))
                    #     2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [1, 2, 3, 4, 5],
                                [2, 4, 1, 3, 5],
                            ],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [
                            curve[:5],
                            [curve[1], curve[3], curve[0], curve[2], curve[4]],
                        ],
                        atol=25.,
                    ))
                    #   Test ns > len(ys).
                    #     scalar
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            6,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[5],
                        delta=25.,
                    )
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            7,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[6],
                        delta=25.,
                    )
                    #     1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [1, 2, 7],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [curve[0], curve[1], curve[6]],
                        atol=25.,
                    ))
                    #     2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [1, 2, 7],
                                [6, 2, 1],
                            ],
                            q=quantile,
                            minimize=minimize,
                        ),
                        [
                            [curve[0], curve[1], curve[6]],
                            [curve[5], curve[1], curve[0]],
                        ],
                        atol=25.,
                    ))

                    # Test when n is non-integral.
                    ys = [0., 50., 25., 100., 75.]
                    ws = (
                        np.random.dirichlet(np.full_like(ys, 5))
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                    #   scalar
                    #     0 < ns <= len(ys)
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            0.5,
                            q=quantile,
                            minimize=minimize,
                        ),
                        dist.quantile_tuning_curve(
                            1,
                            q=1-(1-quantile)**(1/0.5)
                              if minimize else
                              quantile**(1/0.5),
                            minimize=minimize,
                        ),
                    )
                    #     ns > len(ys)
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            10.5,
                            q=quantile,
                            minimize=minimize,
                        ),
                        dist.quantile_tuning_curve(
                            1,
                            q=1-(1-quantile)**(1/10.5)
                              if minimize else
                              quantile**(1/10.5),
                            minimize=minimize,
                        ),
                    )
                    #   1D array
                    #     0 < ns <= len(ys) and ns > len(ys)
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [0.5, 10.5],
                            q=quantile,
                            minimize=minimize,
                        ),
                        dist.quantile_tuning_curve(
                            [1, 21],
                            q=1-(1-quantile)**2
                              if minimize else
                              quantile**2,
                            minimize=minimize,
                        ),
                    ))
                    #   2D array
                    #     0 < ns <= len(ys) and ns > len(ys)
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [0.5, 10.5, 2.5],
                                [2.5,  3.5, 0.5],
                            ],
                            q=quantile,
                            minimize=minimize,
                        ),
                        dist.quantile_tuning_curve(
                            [
                                [1, 21, 5],
                                [5,  7, 1],
                            ],
                            q=1-(1-quantile)**2
                              if minimize else
                              quantile**2,
                            minimize=minimize,
                        ),
                    ))

    def test_average_tuning_curve(self):
        for use_weights in [False, True]:
            for minimize in [False, True]:
                # Test when len(ys) == 1.
                ys = [42.]
                ws = [1.] if use_weights else None
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)

                self.assertEqual(
                    dist.average_tuning_curve(1, minimize=minimize),
                    42.,
                )
                self.assertEqual(
                    dist.average_tuning_curve(10, minimize=minimize),
                    42.,
                )
                self.assertEqual(
                    dist.average_tuning_curve(
                        [1, 10],
                        minimize=minimize,
                    ).tolist(),
                    [42., 42.],
                )
                self.assertEqual(
                    dist.average_tuning_curve(
                        [
                            [1, 10],
                            [10, 1],
                        ],
                        minimize=minimize,
                    ).tolist(),
                    [
                        [42., 42.],
                        [42., 42.],
                    ],
                )

                # Test when len(ys) > 1.
                ys = [0., 50., 25., 100., 75.]
                ws = (
                    np.random.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                curve = np.mean(
                    np.minimum.accumulate(
                        np.random.choice(
                            ys,
                            p=ws,
                            size=(2_500, 7),
                            replace=True,
                        ),
                        axis=1,
                    )
                    if minimize else
                    np.maximum.accumulate(
                        np.random.choice(
                            ys,
                            p=ws,
                            size=(2_500, 7),
                            replace=True,
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
                #   Test 0 < ns <= len(ys).
                #     scalar
                self.assertAlmostEqual(
                    dist.average_tuning_curve(1, minimize=minimize),
                    curve[0],
                    delta=5.,
                )
                self.assertAlmostEqual(
                    dist.average_tuning_curve(3, minimize=minimize),
                    curve[2],
                    delta=5.,
                )
                self.assertAlmostEqual(
                    dist.average_tuning_curve(5, minimize=minimize),
                    curve[4],
                    delta=5.,
                )
                #     1D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [1, 2, 3, 4, 5],
                        minimize=minimize,
                    ),
                    curve[:5],
                    atol=5.,
                ))
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [2, 4, 1, 3, 5],
                        minimize=minimize,
                    ),
                    [curve[1], curve[3], curve[0], curve[2], curve[4]],
                    atol=5.,
                ))
                #     2D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [
                            [1, 2, 3, 4, 5],
                            [2, 4, 1, 3, 5],
                        ],
                        minimize=minimize,
                    ),
                    [
                        curve[:5],
                        [curve[1], curve[3], curve[0], curve[2], curve[4]],
                    ],
                    atol=5.,
                ))
                #   Test ns > len(ys).
                #     scalar
                self.assertAlmostEqual(
                    dist.average_tuning_curve(6, minimize=minimize),
                    curve[5],
                    delta=5.,
                )
                self.assertAlmostEqual(
                    dist.average_tuning_curve(7, minimize=minimize),
                    curve[6],
                    delta=5.,
                )
                #     1D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [1, 2, 7],
                        minimize=minimize,
                    ),
                    [curve[0], curve[1], curve[6]],
                    atol=5.,
                ))
                #     2D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [
                            [1, 2, 7],
                            [6, 2, 1],
                        ],
                        minimize=minimize,
                    ),
                    [
                        [curve[0], curve[1], curve[6]],
                        [curve[5], curve[1], curve[0]],
                    ],
                    atol=5.,
                ))
                #   Test ns <= 0.
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

                # Test when ys has duplicates.
                ys = [0., 0., 50., 0., 100.]
                ws = (
                    np.random.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                curve = np.mean(
                    np.minimum.accumulate(
                        np.random.choice(
                            ys,
                            p=ws,
                            size=(2_500, 7),
                            replace=True,
                        ),
                        axis=1,
                    )
                    if minimize else
                    np.maximum.accumulate(
                        np.random.choice(
                            ys,
                            p=ws,
                            size=(2_500, 7),
                            replace=True,
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
                #   Test 0 < ns <= len(ys).
                #     scalar
                self.assertAlmostEqual(
                    dist.average_tuning_curve(1, minimize=minimize),
                    curve[0],
                    delta=5.,
                )
                self.assertAlmostEqual(
                    dist.average_tuning_curve(3, minimize=minimize),
                    curve[2],
                    delta=5.,
                )
                self.assertAlmostEqual(
                    dist.average_tuning_curve(5, minimize=minimize),
                    curve[4],
                    delta=5.,
                )
                #     1D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [1, 2, 3, 4, 5],
                        minimize=minimize,
                    ),
                    curve[:5],
                    atol=5.,
                ))
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [2, 4, 1, 3, 5],
                        minimize=minimize,
                    ),
                    [curve[1], curve[3], curve[0], curve[2], curve[4]],
                    atol=5.,
                ))
                #     2D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [
                            [1, 2, 3, 4, 5],
                            [2, 4, 1, 3, 5],
                        ],
                        minimize=minimize,
                    ),
                    [
                        curve[:5],
                        [curve[1], curve[3], curve[0], curve[2], curve[4]],
                    ],
                    atol=5.,
                ))
                #   Test ns > len(ys).
                #     scalar
                self.assertAlmostEqual(
                    dist.average_tuning_curve(
                        6,
                        minimize=minimize,
                    ),
                    curve[5],
                    delta=5.,
                )
                self.assertAlmostEqual(
                    dist.average_tuning_curve(
                        7,
                        minimize=minimize,
                    ),
                    curve[6],
                    delta=5.,
                )
                #     1D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [1, 2, 7],
                        minimize=minimize,
                    ),
                    [curve[0], curve[1], curve[6]],
                    atol=5.,
                ))
                #     2D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [
                            [1, 2, 7],
                            [6, 2, 1],
                        ],
                        minimize=minimize,
                    ),
                    [
                        [curve[0], curve[1], curve[6]],
                        [curve[5], curve[1], curve[0]],
                    ],
                    atol=5.,
                ))

                # Test when n is non-integral.
                n_trials = 10_000
                ys = [0., 50., 25., 100., 75.]
                ws = (
                    np.random.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                # Generate ground truth values along the tuning curve.
                ys_sorted, ws_sorted = utils.sort_by_first(
                    ys,
                    ws if ws is not None else np.ones_like(ys) / len(ys),
                )
                ws_sorted_cumsum = np.clip(np.cumsum(ws_sorted), 0., 1.)
                # NOTE: Make sure to include 0 < ns <= len(ys) and ns > len(ys).
                ns = np.arange(10) + 0.5
                ts = np.mean([
                    np.random.choice(
                        ys_sorted,
                        p=(
                            (1-np.concatenate([[0], ws_sorted_cumsum[:-1]]))**n
                            - (1-ws_sorted_cumsum)**n
                            if minimize else
                            ws_sorted_cumsum**n
                            - np.concatenate([[0], ws_sorted_cumsum[:-1]])**n
                        ),
                        size=n_trials,
                    )
                    for n in ns
                ], axis=1)
                # Compute a bound on the standard error for our ground truth.
                err = 6 * (
                    0.25 * (np.max(ys) - np.min(ys))**2  # Popoviciu's inequality
                    / n_trials                           # divided by sample size
                )
                #   scalar
                for n, t in zip(ns, ts):
                    self.assertAlmostEqual(
                        dist.average_tuning_curve(n, minimize=minimize),
                        t,
                        delta=err,
                    )
                #   1D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [ns[0], ns[-1]],
                        minimize=minimize,
                    ),
                    np.array([ts[0], ts[-1]]),
                    atol=err,
                ))
                #   2D array
                self.assertTrue(np.allclose(
                    dist.average_tuning_curve(
                        [
                            [ ns[0], ns[3], ns[-1]],
                            [ns[-1], ns[0],  ns[3]],
                        ],
                        minimize=minimize,
                    ),
                    np.array([
                        [ ts[0], ts[3], ts[-1]],
                        [ts[-1], ts[0],  ts[3]],
                    ]),
                    atol=err,
                ))

    def test_naive_tuning_curve(self):
        # Test when len(ys) == 1.
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)

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

        # Test when len(ys) > 1.
        ys = [0., 50., 25., 100., 75.]
        dist = nonparametric.EmpiricalDistribution(ys)
        #   Test 0 < ns <= len(ys).
        #     scalar
        self.assertEqual(dist.naive_tuning_curve(1), 0.)
        self.assertEqual(dist.naive_tuning_curve(3), 50.)
        self.assertEqual(dist.naive_tuning_curve(5), 100.)
        #     1D array
        self.assertEqual(
            dist.naive_tuning_curve([1, 2, 3, 4, 5]).tolist(),
            [0., 50., 50., 100., 100.],
        )
        self.assertEqual(
            dist.naive_tuning_curve([2, 4, 1, 3, 5]).tolist(),
            [50., 100., 0., 50., 100.],
        )
        #     2D array
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
        #   Test ns > len(ys).
        #     scalar
        self.assertEqual(dist.naive_tuning_curve(6), 100.)
        self.assertEqual(dist.naive_tuning_curve(10), 100.)
        #     1D array
        self.assertEqual(
            dist.naive_tuning_curve([1, 2, 10]).tolist(),
            [0., 50., 100.],
        )
        #     2D array
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
        #   Test non-integer ns.
        for n in range(1, 11):
            self.assertEqual(
                dist.naive_tuning_curve(int(n)),
                dist.naive_tuning_curve(float(n)),
            )
        for n in [0.5, 1.5, 10.5]:
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve(n)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([n])
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([n, 1])
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([[n], [1]])
        #   Test ns <= 0.
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

        # Test when ys has duplicates.
        ys = [0., 0., 50., 0., 100.]
        dist = nonparametric.EmpiricalDistribution(ys)
        #   Test 0 < ns <= len(ys).
        #     scalar
        self.assertEqual(dist.naive_tuning_curve(1), 0.)
        self.assertEqual(dist.naive_tuning_curve(2), 0.)
        self.assertEqual(dist.naive_tuning_curve(3), 50.)
        self.assertEqual(dist.naive_tuning_curve(4), 50.)
        self.assertEqual(dist.naive_tuning_curve(5), 100.)
        #     1D array
        self.assertEqual(
            dist.naive_tuning_curve([1, 2, 3, 4, 5]).tolist(),
            [0., 0., 50., 50., 100.],
        )
        self.assertEqual(
            dist.naive_tuning_curve([2, 4, 1, 3, 5]).tolist(),
            [0., 50., 0., 50., 100.],
        )
        #     2D array
        self.assertEqual(
            dist.naive_tuning_curve([
                [1, 2, 3, 4, 5],
                [2, 4, 1, 3, 5],
            ]).tolist(),
            [
                [0., 0., 50., 50., 100.],
                [0., 50., 0., 50., 100.],
            ],
        )
        #   Test ns > len(ys).
        #     scalar
        self.assertEqual(dist.naive_tuning_curve(6), 100.)
        self.assertEqual(dist.naive_tuning_curve(10), 100.)
        #     1D array
        self.assertEqual(
            dist.naive_tuning_curve([1, 2, 10]).tolist(),
            [0., 0., 100.],
        )
        #     2D array
        self.assertEqual(
            dist.naive_tuning_curve([
                [1, 2, 10],
                [10, 2, 1],
            ]).tolist(),
            [
                [0., 0., 100.],
                [100., 0., 0.],
            ],
        )

        # Test when ws != None.
        ys = [-1, 0, 1]
        ws = [0.1, 0.5, 0.4]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)

        with self.assertRaises(ValueError):
            dist.naive_tuning_curve(1)
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([1])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([2, 1])
        with self.assertRaises(ValueError):
            dist.naive_tuning_curve([[2], [1]])

    def test_v_tuning_curve(self):
        # Test when len(ys) == 1.
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)

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

        # Test when len(ys) > 1.
        ys = [0., 50., 25., 100., 75.]
        dist = nonparametric.EmpiricalDistribution(ys)
        curve = np.mean(
            np.maximum.accumulate(
                np.random.choice(ys, size=(2_500, 7), replace=True),
                axis=1,
            ),
            axis=0,
        )
        #   Test 0 < ns <= len(ys).
        #     scalar
        self.assertAlmostEqual(dist.v_tuning_curve(1), curve[0], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(3), curve[2], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(5), curve[4], delta=5.)
        #     1D array
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
        #     2D array
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
        #   Test ns > len(ys).
        #     scalar
        self.assertAlmostEqual(dist.v_tuning_curve(6), curve[5], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(7), curve[6], delta=5.)
        #     1D array
        self.assertTrue(np.allclose(
            dist.v_tuning_curve([1, 2, 7]),
            [curve[0], curve[1], curve[6]],
            atol=5.,
        ))
        #     2D array
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
        #   Test non-integer ns.
        for n in range(1, 11):
            self.assertEqual(
                dist.v_tuning_curve(int(n)),
                dist.v_tuning_curve(float(n)),
            )
        for n in [0.5, 1.5, 10.5]:
            with self.assertRaises(ValueError):
                dist.v_tuning_curve(n)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([n])
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([n, 1])
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([[n], [1]])
        #   Test ns <= 0.
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

        # Test when ys has duplicates.
        ys = [0., 0., 50., 0., 100.]
        dist = nonparametric.EmpiricalDistribution(ys)
        curve = np.mean(
            np.maximum.accumulate(
                np.random.choice(ys, size=(2_500, 7), replace=True),
                axis=1,
            ),
            axis=0,
        )
        #   Test 0 < ns <= len(ys).
        #     scalar
        self.assertAlmostEqual(dist.v_tuning_curve(1), curve[0], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(3), curve[2], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(5), curve[4], delta=5.)
        #     1D array
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
        #     2D array
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
        #   Test ns > len(ys).
        #     scalar
        self.assertAlmostEqual(dist.v_tuning_curve(6), curve[5], delta=5.)
        self.assertAlmostEqual(dist.v_tuning_curve(7), curve[6], delta=5.)
        #     1D array
        self.assertTrue(np.allclose(
            dist.v_tuning_curve([1, 2, 7]),
            [curve[0], curve[1], curve[6]],
            atol=5.,
        ))
        #     2D array
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

        # Test when ws != None.
        ys = [-1, 0, 1]
        ws = [0.1, 0.5, 0.4]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)

        with self.assertRaises(ValueError):
            dist.v_tuning_curve(1)
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([1])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([2, 1])
        with self.assertRaises(ValueError):
            dist.v_tuning_curve([[2], [1]])

    def test_u_tuning_curve(self):
        # Test when len(ys) == 1.
        ys = [42.]
        dist = nonparametric.EmpiricalDistribution(ys)

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

        # Test when len(ys) > 1.
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
        #   Test 0 < ns <= len(ys).
        #     scalar
        self.assertAlmostEqual(dist.u_tuning_curve(1), curve[0], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(3), curve[2], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(5), curve[4], delta=5.)
        #     1D array
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
        #     2D array
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
        #   Test ns > len(ys).
        #     scalar
        self.assertAlmostEqual(dist.u_tuning_curve(6), curve[4], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(7), curve[4], delta=5.)
        #     1D array
        self.assertTrue(np.allclose(
            dist.u_tuning_curve([1, 2, 7]),
            [curve[0], curve[1], curve[4]],
            atol=5.,
        ))
        #     2D array
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
        #   Test non-integer ns.
        for n in range(1, 11):
            self.assertEqual(
                dist.u_tuning_curve(int(n)),
                dist.u_tuning_curve(float(n)),
            )
        for n in [0.5, 1.5, 10.5]:
            with self.assertRaises(ValueError):
                dist.u_tuning_curve(n)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([n])
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([n, 1])
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([[n], [1]])
        #   Test ns <= 0.
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

        # Test when ys has duplicates.
        ys = [0., 0., 50., 0., 100.]
        dist = nonparametric.EmpiricalDistribution(ys)
        curve = np.mean(
            np.maximum.accumulate(
                # Sort random numbers to batch sampling without replacement.
                np.array(ys)[np.argsort(np.random.rand(2_500, 5), axis=1)],
                axis=1,
            ),
            axis=0,
        )
        #   Test 0 < ns <= len(ys).
        #     scalar
        self.assertAlmostEqual(dist.u_tuning_curve(1), curve[0], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(3), curve[2], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(5), curve[4], delta=5.)
        #     1D array
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
        #     2D array
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
        #   Test ns > len(ys).
        #     scalar
        self.assertAlmostEqual(dist.u_tuning_curve(6), curve[4], delta=5.)
        self.assertAlmostEqual(dist.u_tuning_curve(7), curve[4], delta=5.)
        #     1D array
        self.assertTrue(np.allclose(
            dist.u_tuning_curve([1, 2, 7]),
            [curve[0], curve[1], curve[4]],
            atol=5.,
        ))
        #     2D array
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

        # Test when ws != None.
        ys = [-1, 0, 1]
        ws = [0.1, 0.5, 0.4]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)

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
        n = 5
        methods = ['dkw', 'ks', 'ld_equal_tailed', 'ld_highest_density']
        for method in methods:
            for confidence in [0.5, 0.9]:
                for a, b in [(0., 1.), (-np.inf, np.inf), (None, None)]:
                    for has_duplicates in [False, True]:
                        if has_duplicates:
                            ys = np.random.uniform(0, 1, size=n)
                            ys = np.concatenate([ys[:n-n//3], ys[:n//3]])
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    'ignore',
                                    message=r'Duplicates detected in ys',
                                    category=RuntimeWarning,
                                )
                                lo, dist, hi =\
                                    nonparametric.EmpiricalDistribution\
                                    .confidence_bands(
                                        ys,
                                        confidence,
                                        a=a,
                                        b=b,
                                        method=method,
                                    )
                        else:
                            ys = np.random.uniform(0, 1, size=n)
                            lo, dist, hi =\
                                nonparametric.EmpiricalDistribution\
                                .confidence_bands(
                                    ys,
                                    confidence,
                                    a=a,
                                    b=b,
                                    method=method,
                                )
                        # Check that dist is the empirical distribution.
                        self.assertEqual(
                            dist.cdf(ys).tolist(),
                            nonparametric.EmpiricalDistribution(ys).cdf(ys)\
                              .tolist(),
                        )
                        # Check bounded below by 0.
                        #   on ys
                        self.assertGreaterEqual(np.min(lo.cdf(ys)), 0.)
                        self.assertGreaterEqual(np.min(dist.cdf(ys)), 0.)
                        self.assertGreaterEqual(np.min(hi.cdf(ys)), 0.)
                        #   on the support's bounds
                        #     lower bound
                        if a is not None:
                            self.assertGreaterEqual(lo.cdf(a), 0.)
                            self.assertGreaterEqual(dist.cdf(a), 0.)
                            self.assertGreaterEqual(hi.cdf(a), 0.)
                        #     upper bound
                        if b is not None:
                            self.assertGreaterEqual(lo.cdf(b), 0.)
                            self.assertGreaterEqual(dist.cdf(b), 0.)
                            self.assertGreaterEqual(hi.cdf(b), 0.)
                        #     -np.inf
                        self.assertGreaterEqual(lo.cdf(-np.inf), 0.)
                        self.assertGreaterEqual(dist.cdf(-np.inf), 0.)
                        self.assertGreaterEqual(hi.cdf(-np.inf), 0.)
                        #     np.inf
                        self.assertGreaterEqual(lo.cdf(np.inf), 0.)
                        self.assertGreaterEqual(dist.cdf(np.inf), 0.)
                        self.assertGreaterEqual(hi.cdf(np.inf), 0.)
                        # Check bounded above by 1.
                        #   on ys
                        self.assertLessEqual(np.max(lo.cdf(ys)), 1.)
                        self.assertLessEqual(np.max(dist.cdf(ys)), 1.)
                        self.assertLessEqual(np.max(hi.cdf(ys)), 1.)
                        #   on the support's bounds
                        #     lower bound
                        if a is not None:
                            self.assertLessEqual(lo.cdf(a), 1.)
                            self.assertLessEqual(dist.cdf(a), 1.)
                            self.assertLessEqual(hi.cdf(a), 1.)
                        #     upper bound
                        if b is not None:
                            self.assertLessEqual(lo.cdf(b), 1.)
                            self.assertLessEqual(dist.cdf(b), 1.)
                            self.assertLessEqual(hi.cdf(b), 1.)
                        #     -np.inf
                        self.assertLessEqual(lo.cdf(-np.inf), 1.)
                        self.assertLessEqual(dist.cdf(-np.inf), 1.)
                        self.assertLessEqual(hi.cdf(-np.inf), 1.)
                        #     np.inf
                        self.assertLessEqual(lo.cdf(np.inf), 1.)
                        self.assertLessEqual(dist.cdf(np.inf), 1.)
                        self.assertLessEqual(hi.cdf(np.inf), 1.)
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
        for has_ws in [False, True]:
            for _ in range(10):
                dist = nonparametric.EmpiricalDistribution(
                    np.random.uniform(0, 1, size=5),
                    ws=np.random.dirichlet(np.ones(5)) if has_ws else None,
                )
                ys = dist.sample(100)
                self.assertEqual(dist.ppf(dist.cdf(ys)).tolist(), ys.tolist())

    def test_ppf_at_extreme_values(self):
        for has_ws in [False, True]:
            for a, b in [[None, None], [None, 1.], [-1., None], [-1., 1.]]:
                dist = nonparametric.EmpiricalDistribution(
                    [0.],
                    ws=[1.] if has_ws else None,
                    a=a,
                    b=b,
                )
                self.assertEqual(
                    dist.ppf(0. - 1e-12),
                    a if a is not None else -np.inf,
                )
                self.assertEqual(
                    dist.ppf(0.),
                    a if a is not None else -np.inf,
                )
                self.assertEqual(dist.ppf(1.), 0.)
                self.assertEqual(dist.ppf(1. + 1e-12), 0.)

    def test_quantile_tuning_curve_minimize_is_dual_to_maximize(self):
        for _ in range(4):
            for use_weights in [False, True]:
                ys = np.random.normal(size=7)
                # NOTE: This test must use an _odd_ number of ys. When
                # the sample size is even, there is no exact
                # median. Our definition takes the order statistic to
                # the left of the middle. Thus, the median for ys and
                # -ys differs. Avoid this issue by using an odd number.
                ws = (
                    np.random.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                ns = np.arange(1, 17)

                self.assertTrue(np.allclose(
                    nonparametric
                      .EmpiricalDistribution(ys, ws=ws)
                      .quantile_tuning_curve(ns, minimize=False),
                    -nonparametric
                      .EmpiricalDistribution(-ys, ws=ws)
                      .quantile_tuning_curve(ns, minimize=True),
                ))
                self.assertTrue(np.allclose(
                    nonparametric
                      .EmpiricalDistribution(ys, ws=ws)
                      .quantile_tuning_curve(ns, minimize=True),
                    -nonparametric
                      .EmpiricalDistribution(-ys, ws=ws)
                      .quantile_tuning_curve(ns, minimize=False),
                ))

    def test_quantile_tuning_curve_with_probability_mass_at_infinity(self):
        for ys, n, expected_minimize, expected_maximize in [
                #            ys,                   n, minimize, maximize
                ([-np.inf, 100],                   1,  -np.inf,  -np.inf),
                ([-np.inf, 100],                   2,  -np.inf,      100),
                ([100, np.inf],                    1,      100,      100),
                ([100, np.inf],                    2,      100,   np.inf),
                ([-np.inf, -10., 0., 10.,   100.], 1,        0,        0),
                ([-np.inf, -10., 0., 10.,   100.], 4,  -np.inf,      100),
                ([  -100., -10., 0., 10., np.inf], 1,        0,        0),
                ([  -100., -10., 0., 10., np.inf], 4,     -100,   np.inf),
                ([-np.inf, np.inf],                1,  -np.inf,  -np.inf),
                ([-np.inf, np.inf],                2,  -np.inf,   np.inf),
                ([-np.inf, 0., np.inf],            1,        0,        0),
                ([-np.inf, 0., np.inf],            2,  -np.inf,   np.inf),
        ]:
            for use_weights in [False, True]:
                for minimize in [False, True]:
                    expected = expected_minimize if minimize else expected_maximize
                    ws = (
                        np.ones_like(ys) / len(ys)
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                    # Test 0 < ns <= len(ys).
                    #   scalar
                    self.assertTrue(np.isclose(
                        dist.quantile_tuning_curve(
                            n,
                            minimize=minimize,
                        ),
                        expected,
                        equal_nan=True,
                    ))
                    #   1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [n] * 10,
                            minimize=minimize,
                        ),
                        expected,
                        equal_nan=True,
                    ))
                    #   2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [n] * 10,
                                [n] * 10,
                            ],
                            minimize=minimize,
                        ),
                        expected,
                        equal_nan=True,
                    ))
                    # Test ns > len(ys).
                    #   scalar
                    self.assertTrue(np.isclose(
                        dist.quantile_tuning_curve(6, minimize=minimize),
                        ys[0] if minimize else ys[-1],
                        equal_nan=True,
                    ))
                    self.assertTrue(np.isclose(
                        dist.quantile_tuning_curve(7, minimize=minimize),
                        ys[0] if minimize else ys[-1],
                        equal_nan=True,
                    ))
                    #   1D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [6, 7, 8],
                            minimize=minimize,
                        ),
                        ys[0] if minimize else ys[-1],
                        equal_nan=True,
                    ))
                    #   2D array
                    self.assertTrue(np.allclose(
                        dist.quantile_tuning_curve(
                            [
                                [6, 7, 8],
                                [9, 8, 7],
                            ],
                            minimize=minimize,
                        ),
                        ys[0] if minimize else ys[-1],
                        equal_nan=True,
                    ))

    def test_average_tuning_curve_with_probability_mass_at_infinity(self):
        for ys, expected in [
                ([-np.inf, -10., 0., 10.,   100.], -np.inf),
                ([  -100., -10., 0., 10., np.inf],  np.inf),
                ([-np.inf, -10., 0., 10., np.inf],  np.nan),
        ]:
            for use_weights in [False, True]:
                for minimize in [False, True]:
                    ws = (
                        np.random.dirichlet(np.ones_like(ys))
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)

                    with warnings.catch_warnings():
                        if np.isnan(expected):
                            # If both infinity and negative infinity are
                            # present in ys then numpy will log a
                            # warning when taking a weighted sum over
                            # the samples. This warning is useful because
                            # if both infinity and negative infinity have
                            # non-zero probability then the average
                            # tuning curve is undefined. Suppress this
                            # warning in the test though, since it is
                            # expected.
                            warnings.filterwarnings(
                                'ignore',
                                message=r'invalid value encountered in reduce',
                                category=RuntimeWarning,
                            )

                        # Test 0 < ns <= len(ys).
                        #   scalar
                        self.assertTrue(np.isclose(
                            dist.average_tuning_curve(1, minimize=minimize),
                            expected,
                            equal_nan=True,
                        ))
                        self.assertTrue(np.isclose(
                            dist.average_tuning_curve(3, minimize=minimize),
                            expected,
                            equal_nan=True,
                        ))
                        self.assertTrue(np.isclose(
                            dist.average_tuning_curve(5, minimize=minimize),
                            expected,
                            equal_nan=True,
                        ))
                        #   1D array
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [1, 2, 3, 4, 5],
                                minimize=minimize,
                            ),
                            expected,
                            equal_nan=True,
                        ))
                        #   2D array
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [
                                    [1, 2, 3, 4, 5],
                                    [2, 4, 1, 3, 5],
                                ],
                                minimize=minimize,
                            ),
                            expected,
                            equal_nan=True,
                        ))
                        # Test ns > len(ys).
                        #   scalar
                        self.assertTrue(np.isclose(
                            dist.average_tuning_curve(6, minimize=minimize),
                            expected,
                            equal_nan=True,
                        ))
                        self.assertTrue(np.isclose(
                            dist.average_tuning_curve(7, minimize=minimize),
                            expected,
                            equal_nan=True,
                        ))
                        #   1D array
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [1, 2, 7],
                                minimize=minimize,
                            ),
                            expected,
                            equal_nan=True,
                        ))
                        #   2D array
                        self.assertTrue(np.allclose(
                            dist.average_tuning_curve(
                                [
                                    [1, 2, 7],
                                    [6, 2, 1],
                                ],
                                minimize=minimize,
                            ),
                            expected,
                            equal_nan=True,
                        ))

    @pytest.mark.level(3)
    def test_dkw_bands_have_correct_coverage(self):
        n_trials = 1_000
        dist = stats.norm(0., 1.)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [2, 16]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='dkw',
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15))
                    )

                _, hi = utils.binomial_confidence_interval(
                    n_successes=np.sum(covered),
                    n_total=n_trials,
                    confidence=0.999999,
                )
                self.assertGreater(hi, confidence)

    @pytest.mark.level(3)
    def test_ks_bands_have_correct_coverage(self):
        n_trials = 1_000
        dist = stats.norm(0., 1.)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [2, 16]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='ks',
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15))
                    )

                lo, hi = utils.binomial_confidence_interval(
                    n_successes=np.sum(covered),
                    n_total=n_trials,
                    confidence=0.999999,
                )
                self.assertLess(lo, confidence)
                self.assertGreater(hi, confidence)

    @pytest.mark.level(3)
    def test_ld_equal_tailed_bands_have_correct_coverage(self):
        n_trials = 1_000
        dist = stats.norm(0., 1.)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [2, 16]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='ld_equal_tailed',
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15))
                    )

                lo, hi = utils.binomial_confidence_interval(
                    n_successes=np.sum(covered),
                    n_total=n_trials,
                    confidence=0.999999,
                )
                self.assertLess(lo, confidence)
                self.assertGreater(hi, confidence)

    @pytest.mark.level(3)
    def test_ld_highest_density_bands_have_correct_coverage(self):
        n_trials = 1_000
        dist = stats.norm(0., 1.)
        for confidence in [0.5, 0.9, 0.99]:
            for n_samples in [2, 16]:
                covered = []
                for _ in range(n_trials):
                    ys = dist.rvs(size=n_samples)
                    lo, _, hi =\
                        nonparametric.EmpiricalDistribution.confidence_bands(
                            ys, confidence, method='ld_highest_density',
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15))
                    )

                lo, hi = utils.binomial_confidence_interval(
                    n_successes=np.sum(covered),
                    n_total=n_trials,
                    confidence=0.999999,
                )
                self.assertLess(lo, confidence)
                self.assertGreater(hi, confidence)
