"""Tests for opda.nonparametric."""

import warnings

import numpy as np
import pytest
from scipy import stats

from opda import nonparametric, utils
import opda.random

from tests import testcases


class EmpiricalDistributionTestCase(testcases.RandomTestCase):
    """Test opda.nonparametric.EmpiricalDistribution."""

    @pytest.mark.level(1)
    def test_attributes(self):
        n_samples = 1_000_000

        # Test the attributes.
        #   when all ys are finite
        yss = [[0.], [1.], [-1., 1.], [0., 1.], [-1., 0., 1.]]
        bounds = [(-np.inf, np.inf), (-5, 5), (-10., 10.)]
        for ys in yss:
            wss = (
                [None, self.generator.dirichlet(np.ones_like(ys))]
                if len(ys) > 1 else
                # If len(ys) == 1 then there is only one possible set of
                # weights: ws == [1.] or, equivalently, ws == None.
                [None]
            )
            for ws in wss:
                for a, b in bounds:
                    dist = nonparametric.EmpiricalDistribution(ys, ws, a, b)
                    samples = dist.sample(n_samples)
                    self.assertAlmostEqual(
                        dist.mean,
                        np.mean(samples),
                        delta=6 * np.std(samples) / n_samples**0.5,
                    )
                    self.assertAlmostEqual(
                        dist.variance,
                        np.mean((samples - dist.mean)**2),
                        delta=(
                            6 * np.std((samples - dist.mean)**2)
                            / n_samples**0.5
                        ),
                    )
        #   when some ys are infinite
        for ys, ws, (a, b), mean, variance in [
                ([-np.inf], None, (-np.inf, np.inf), -np.inf, np.nan),
                ([-np.inf], None, (-np.inf,     0.), -np.inf, np.nan),
                ([ np.inf], None, (-np.inf, np.inf),  np.inf, np.nan),
                ([ np.inf], None, (     0., np.inf),  np.inf, np.nan),
                ([-np.inf, -1.], None, (-np.inf, np.inf), -np.inf, np.nan),
                ([-np.inf, -1.], None, (-np.inf,     0.), -np.inf, np.nan),
                ([ np.inf,  1.], None, (-np.inf, np.inf),  np.inf, np.nan),
                ([ np.inf,  1.], None, (     0., np.inf),  np.inf, np.nan),
                ([-np.inf, -1.], [0., 1.], (-np.inf, np.inf),     -1.,     0.),
                ([-np.inf, -1.], [1., 0.], (-np.inf, np.inf), -np.inf, np.nan),
                ([ np.inf,  1.], [0., 1.], (-np.inf, np.inf),      1.,     0.),
                ([ np.inf,  1.], [1., 0.], (-np.inf, np.inf),  np.inf, np.nan),
                ([-np.inf, -1.], [.2, .8], (-np.inf, np.inf), -np.inf, np.nan),
                ([ np.inf,  1.], [.2, .8], (-np.inf, np.inf),  np.inf, np.nan),
                ([-np.inf, np.inf], [1, 0], (-np.inf, np.inf), -np.inf, np.nan),
                ([-np.inf, np.inf], [0, 1], (-np.inf, np.inf),  np.inf, np.nan),
                ([-np.inf, np.inf],    None, (-np.inf, np.inf), np.nan, np.nan),
                ([-np.inf, np.inf], [.2,.8], (-np.inf, np.inf), np.nan, np.nan),
                ([-np.inf, np.inf], [.8,.2], (-np.inf, np.inf), np.nan, np.nan),
        ]:
            dist = nonparametric.EmpiricalDistribution(ys, ws, a, b)
            self.assertTrue(np.allclose(
                dist.mean,
                mean,
                equal_nan=True,
            ))
            self.assertTrue(np.allclose(
                dist.variance,
                variance,
                equal_nan=True,
            ))

    def test___eq__(self):
        yss = [[0.], [1.], [-1., 1.], [0., 1.], [-1., 0., 1.]]
        bounds = [(-np.inf, np.inf), (-5, 5), (-10., 10.)]
        for ys in yss:
            wss = (
                [None, self.generator.dirichlet(np.ones_like(ys))]
                if len(ys) > 1 else
                # If len(ys) == 1 then there is only one possible set of
                # weights: ws == [1.] or, equivalently, ws == None.
                [None]
            )
            for ws in wss:
                for a, b in bounds:
                    # Test inequality with other objects.
                    self.assertNotEqual(
                        nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        None,
                    )
                    self.assertNotEqual(
                        nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        1.,
                    )
                    self.assertNotEqual(
                        nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        set(),
                    )

                    # Test (in)equality between instances of the same class.
                    #   equality
                    self.assertEqual(
                        nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        nonparametric.EmpiricalDistribution(ys, ws, a, b),
                    )
                    #   inequality
                    for ys_ in yss:
                        if np.array_equal(ys_, ys) or len(ys_) != len(ys):
                            # NOTE: ys_ must have the same length as ws
                            # (and thus ys) to use ys_ with ws as
                            # arguments to EmpiricalDistribution.
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            nonparametric.EmpiricalDistribution(ys_, ws, a, b),
                        )
                    for ws_ in wss:
                        if np.array_equal(ws_, ws):
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            nonparametric.EmpiricalDistribution(ys, ws_, a, b),
                        )
                    for a_, _ in bounds:
                        if a_ == a:
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            nonparametric.EmpiricalDistribution(ys, ws, a_, b),
                        )
                    for _, b_ in bounds:
                        if b_ == b:
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            nonparametric.EmpiricalDistribution(ys, ws, a, b_),
                        )

                    # Test (in)equality between instances of different classes.
                    class EmpiricalDistributionSubclass(
                            nonparametric.EmpiricalDistribution,
                    ):
                        pass
                    #   equality
                    self.assertEqual(
                        nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        EmpiricalDistributionSubclass(ys, ws, a, b),
                    )
                    self.assertEqual(
                        EmpiricalDistributionSubclass(ys, ws, a, b),
                        nonparametric.EmpiricalDistribution(ys, ws, a, b),
                    )
                    #   inequality
                    for ys_ in yss:
                        if np.array_equal(ys_, ys) or len(ys_) != len(ys):
                            # NOTE: ys_ must have the same length as ws
                            # (and thus ys) to use ys_ with ws as
                            # arguments to EmpiricalDistribution.
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            EmpiricalDistributionSubclass(ys_, ws, a, b),
                        )
                        self.assertNotEqual(
                            EmpiricalDistributionSubclass(ys_, ws, a, b),
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        )
                    for ws_ in wss:
                        if np.array_equal(ws_, ws):
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            EmpiricalDistributionSubclass(ys, ws_, a, b),
                        )
                        self.assertNotEqual(
                            EmpiricalDistributionSubclass(ys, ws_, a, b),
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        )
                    for a_, _ in bounds:
                        if a_ == a:
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            EmpiricalDistributionSubclass(ys, ws, a_, b),
                        )
                        self.assertNotEqual(
                            EmpiricalDistributionSubclass(ys, ws, a_, b),
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        )
                    for _, b_ in bounds:
                        if b_ == b:
                            continue
                        self.assertNotEqual(
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                            EmpiricalDistributionSubclass(ys, ws, a, b_),
                        )
                        self.assertNotEqual(
                            EmpiricalDistributionSubclass(ys, ws, a, b_),
                            nonparametric.EmpiricalDistribution(ys, ws, a, b),
                        )

        # Test equality when one instance has ws is None and the other
        # has ws is an array of equal weights.
        for ys in yss:
            for a, b in bounds:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"ws gives equal weight to each sample.",
                        category=RuntimeWarning,
                    )
                    self.assertEqual(
                        nonparametric.EmpiricalDistribution(ys, None, a, b),
                        nonparametric.EmpiricalDistribution(
                            ys,
                            np.ones_like(ys) / len(ys),
                            a,
                            b,
                        ),
                    )
                    self.assertEqual(
                        nonparametric.EmpiricalDistribution(
                            ys,
                            np.ones_like(ys) / len(ys),
                            a,
                            b,
                        ),
                        nonparametric.EmpiricalDistribution(ys, None, a, b),
                    )

    def test___str__(self):
        self.assertEqual(
            str(nonparametric.EmpiricalDistribution(
                [0., 1.], None, -1., 1.,
            )),
            f"EmpiricalDistribution("
                f"ys={np.array([0., 1.])!s},"
                f" ws={None!s},"
                f" a={np.array(-1.)[()]!s},"
                f" b={np.array(1.)[()]!s}"
            f")",
        )
        self.assertEqual(
            str(nonparametric.EmpiricalDistribution(
                [0., 1.], [0.3, 0.7], -1., 1.,
            )),
            f"EmpiricalDistribution("
                f"ys={np.array([0., 1.])!s},"
                f" ws={np.array([0.3, 0.7])!s},"
                f" a={np.array(-1.)[()]!s},"
                f" b={np.array(1.)[()]!s}"
            f")",
        )

    def test___repr__(self):
        self.assertEqual(
            repr(nonparametric.EmpiricalDistribution(
                [0., 1.], None, -1., 1.,
            )),
            f"EmpiricalDistribution("
                f"ys={np.array([0., 1.])!r},"
                f" ws={None!r},"
                f" a={np.array(-1.)[()]!r},"
                f" b={np.array(1.)[()]!r}"
            f")",
        )
        self.assertEqual(
            repr(nonparametric.EmpiricalDistribution(
                [0., 1.], [0.3, 0.7], -1., 1.,
            )),
            f"EmpiricalDistribution("
                f"ys={np.array([0., 1.])!r},"
                f" ws={np.array([0.3, 0.7])!r},"
                f" a={np.array(-1.)[()]!r},"
                f" b={np.array(1.)[()]!r}"
            f")",
        )

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
        #   when output is scalar
        #     size is left at its default value
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        y = dist.sample()
        self.assertTrue(np.isscalar(y))
        self.assertIn(y, ys)
        #     size=None
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        y = dist.sample(size=None)
        self.assertTrue(np.isscalar(y))
        self.assertIn(y, ys)

        # Test with weights.
        #   when len(ys) == 1
        ys = [42.]
        ws = [1.]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"ws gives equal weight to each sample.",
                category=RuntimeWarning,
            )
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
        #   when output is scalar
        #     size is left at its default value
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        y = dist.sample()
        self.assertTrue(np.isscalar(y))
        self.assertIn(y, ys)
        #     size=None
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        y = dist.sample(size=None)
        self.assertTrue(np.isscalar(y))
        self.assertIn(y, ys)

    def test_pmf(self):
        for a, b in [(-1e4, 1e4), (-np.inf, np.inf)]:
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
            #   when output is scalar
            ys = [-42., 1., 42., 100., 1_000.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertTrue(np.isscalar(dist.pmf(1.)))
            self.assertEqual(dist.pmf(1.), 0.2)

            # Test with weights.
            #   when len(ys) == 1
            ys = [42.]
            ws = [1.]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"ws gives equal weight to each sample.",
                    category=RuntimeWarning,
                )
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
            #   when output is scalar
            ys = [-42., 1., 42., 100., 1_000.]
            ws = [0.1, 0.2, 0.3, 0.15, 0.25]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertTrue(np.isscalar(dist.pmf(1.)))
            self.assertEqual(dist.pmf(1.), 0.2)

            # Test outside of the distribution's support.
            ys = [-42., 1., 42., 100., 1_000.]
            for ws in [None, [0.1, 0.2, 0.3, 0.15, 0.25]]:
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
                self.assertEqual(dist.pmf(a - 1e-10), 0.)
                self.assertEqual(dist.pmf(a - 10), 0.)
                self.assertEqual(dist.pmf(b + 1e-10), 0.)
                self.assertEqual(dist.pmf(b + 10), 0.)

    def test_cdf(self):
        for a, b in [(-1e4, 1e4), (-np.inf, np.inf)]:
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
            #   when output is scalar
            ys = [-42., 1., 42., 100., 1_000.]
            dist = nonparametric.EmpiricalDistribution(ys, a=a, b=b)
            self.assertTrue(np.isscalar(dist.cdf(1.)))
            self.assertAlmostEqual(dist.cdf(1.), 0.4)

            # Test without weights.
            #   when len(ys) == 1
            ys = [42.]
            ws = [1.]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"ws gives equal weight to each sample.",
                    category=RuntimeWarning,
                )
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
            #   when output is scalar
            ys = [-42., 1., 42., 100., 1_000.]
            ws = [0.1, 0.2, 0.3, 0.15, 0.25]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
            self.assertTrue(np.isscalar(dist.cdf(1.)))
            self.assertAlmostEqual(dist.cdf(1.), 0.3)

            # Test outside of the distribution's support.
            ys = [-42., 1., 42., 100., 1_000.]
            for ws in [None, [0.1, 0.2, 0.3, 0.15, 0.25]]:
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws, a=a, b=b)
                self.assertEqual(dist.cdf(a - 1e-10), 0.)
                self.assertEqual(dist.cdf(a - 10), 0.)
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
        #   when output is scalar
        ys = [-42., 1., 42., 100., 1_000.]
        dist = nonparametric.EmpiricalDistribution(ys)
        self.assertTrue(np.isscalar(dist.ppf(0.4)))
        self.assertEqual(dist.ppf(0.4), 1.)

        # Test with weights.
        #   when len(ys) == 1
        ys = [42.]
        ws = [1.]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"ws gives equal weight to each sample.",
                category=RuntimeWarning,
            )
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
        #   when output is scalar
        ys = [-42., 1., 42., 100., 1_000.]
        ws = [0.1, 0.2, 0.3, 0.15, 0.25]
        dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
        self.assertTrue(np.isscalar(dist.ppf(0.4)))
        self.assertEqual(dist.ppf(0.4), 42.)

    def test_quantile_tuning_curve(self):
        for use_weights in [False, True]:
            for quantile in [0.25, 0.5, 0.75]:
                for minimize in [False, True]:
                    # Test when len(ys) == 1.
                    ys = [42.]
                    ws = [1.] if use_weights else None
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"ws gives equal weight to each sample.",
                            category=RuntimeWarning,
                        )
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
                        self.generator.dirichlet(np.full_like(ys, 5))
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                    curve = np.quantile(
                        np.minimum.accumulate(
                            self.generator.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        )
                        if minimize else
                        np.maximum.accumulate(
                            self.generator.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        ),
                        quantile,
                        method="inverted_cdf",
                        axis=0,
                    )
                    #   Test 0 < ns <= len(ys).
                    #     scalar
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            1,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            1,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[0],
                        delta=25.,
                    )
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            3,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            3,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[2],
                        delta=25.,
                    )
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            5,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
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
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            6,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
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
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            7,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
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
                        self.generator.dirichlet(np.full_like(ys, 5))
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                    curve = np.quantile(
                        np.minimum.accumulate(
                            self.generator.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        )
                        if minimize else
                        np.maximum.accumulate(
                            self.generator.choice(
                                ys,
                                p=ws,
                                size=(1_000, 7),
                                replace=True,
                            ),
                            axis=1,
                        ),
                        quantile,
                        method="inverted_cdf",
                        axis=0,
                    )
                    #   Test 0 < ns <= len(ys).
                    #     scalar
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            1,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            1,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[0],
                        delta=25.,
                    )
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            3,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            3,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[2],
                        delta=25.,
                    )
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            5,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
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
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            6,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
                    self.assertAlmostEqual(
                        dist.quantile_tuning_curve(
                            6,
                            q=quantile,
                            minimize=minimize,
                        ),
                        curve[5],
                        delta=25.,
                    )
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            7,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
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
                        self.generator.dirichlet(np.full_like(ys, 5))
                        if use_weights else
                        None
                    )
                    dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                    #   scalar
                    #     0 < ns <= len(ys)
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            0.5,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
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
                    self.assertTrue(np.isscalar(
                        dist.quantile_tuning_curve(
                            10.5,
                            q=quantile,
                            minimize=minimize,
                        ),
                    ))
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
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"ws gives equal weight to each sample.",
                        category=RuntimeWarning,
                    )
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
                    self.generator.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                curve = np.mean(
                    np.minimum.accumulate(
                        self.generator.choice(
                            ys,
                            p=ws,
                            size=(2_500, 7),
                            replace=True,
                        ),
                        axis=1,
                    )
                    if minimize else
                    np.maximum.accumulate(
                        self.generator.choice(
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
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(1, minimize=minimize),
                ))
                self.assertAlmostEqual(
                    dist.average_tuning_curve(1, minimize=minimize),
                    curve[0],
                    delta=5.,
                )
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(3, minimize=minimize),
                ))
                self.assertAlmostEqual(
                    dist.average_tuning_curve(3, minimize=minimize),
                    curve[2],
                    delta=5.,
                )
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(5, minimize=minimize),
                ))
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
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(6, minimize=minimize),
                ))
                self.assertAlmostEqual(
                    dist.average_tuning_curve(6, minimize=minimize),
                    curve[5],
                    delta=5.,
                )
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(7, minimize=minimize),
                ))
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
                    self.generator.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                dist = nonparametric.EmpiricalDistribution(ys, ws=ws)
                curve = np.mean(
                    np.minimum.accumulate(
                        self.generator.choice(
                            ys,
                            p=ws,
                            size=(2_500, 7),
                            replace=True,
                        ),
                        axis=1,
                    )
                    if minimize else
                    np.maximum.accumulate(
                        self.generator.choice(
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
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(1, minimize=minimize),
                ))
                self.assertAlmostEqual(
                    dist.average_tuning_curve(1, minimize=minimize),
                    curve[0],
                    delta=5.,
                )
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(3, minimize=minimize),
                ))
                self.assertAlmostEqual(
                    dist.average_tuning_curve(3, minimize=minimize),
                    curve[2],
                    delta=5.,
                )
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(5, minimize=minimize),
                ))
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
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(6, minimize=minimize),
                ))
                self.assertAlmostEqual(
                    dist.average_tuning_curve(6, minimize=minimize),
                    curve[5],
                    delta=5.,
                )
                self.assertTrue(np.isscalar(
                    dist.average_tuning_curve(7, minimize=minimize),
                ))
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

                # Test when n is non-integral.
                n_trials = 10_000
                ys = [0., 50., 25., 100., 75.]
                ws = (
                    self.generator.dirichlet(np.ones_like(ys))
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
                def normalize(ps): return ps / np.sum(ps)
                ns = np.arange(10) + 0.5
                ts = np.mean([
                    self.generator.choice(
                        ys_sorted,
                        p=normalize(
                            # Normalize the probabilities to fix round errors.
                            np.diff(
                                (1-ws_sorted_cumsum[::-1])**n,
                                append=[1.],
                            )[::-1]
                            if minimize else
                            np.diff(ws_sorted_cumsum**n, prepend=[0.]),
                        ),
                        size=n_trials,
                    )
                    for n in ns
                ], axis=1)
                # Compute a bound on the standard error for our ground truth.
                err = 6 * np.sqrt(
                    # Use Popoviciu's inequality on variances to bound the
                    # variance of one sample, then divide by the sample size to
                    # get an upper bound on the variance of the average tuning
                    # curve estimate.
                    0.25 * (np.max(ys) - np.min(ys))**2 / n_trials,
                )
                #   scalar
                for n, t in zip(ns, ts):
                    self.assertTrue(np.isscalar(
                        dist.average_tuning_curve(n, minimize=minimize),
                    ))
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
        for minimize in [False, True]:
            # Test when len(ys) == 1.
            ys = [42.]
            dist = nonparametric.EmpiricalDistribution(ys)

            self.assertEqual(
                dist.naive_tuning_curve(1, minimize=minimize),
                42.,
            )
            self.assertEqual(
                dist.naive_tuning_curve(10, minimize=minimize),
                42.,
            )
            self.assertEqual(
                dist.naive_tuning_curve([1, 10], minimize=minimize).tolist(),
                [42., 42.],
            )
            self.assertEqual(
                dist.naive_tuning_curve(
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
            ys = [0., 50., -25., 100., 75.]
            dist = nonparametric.EmpiricalDistribution(ys)
            #   Test 0 < ns <= len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(1, minimize=minimize),
            ))
            self.assertEqual(dist.naive_tuning_curve(1, minimize=minimize), 0.)
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(3, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(3, minimize=minimize),
                -25. if minimize else 50.,
            )
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(5, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(5, minimize=minimize),
                -25. if minimize else 100.,
            )
            #     1D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [1, 2, 3, 4, 5],
                    minimize=minimize,
                ).tolist(),
                [0., 0., -25., -25., -25.]
                if minimize else
                [0., 50., 50., 100., 100.],
            )
            self.assertEqual(
                dist.naive_tuning_curve(
                    [2, 4, 1, 3, 5],
                    minimize=minimize,
                ).tolist(),
                [0., -25., 0., -25., -25.]
                if minimize else
                [50., 100., 0., 50., 100.],
            )
            #     2D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [
                        [1, 2, 3, 4, 5],
                        [2, 4, 1, 3, 5],
                    ],
                    minimize=minimize,
                ).tolist(),
                [
                    [0., 0., -25., -25., -25.],
                    [0., -25., 0., -25., -25.],
                ]
                if minimize else
                [
                    [0., 50., 50., 100., 100.],
                    [50., 100., 0., 50., 100.],
                ],
            )
            #   Test ns > len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(6, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(6, minimize=minimize),
                -25. if minimize else 100.,
            )
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(10, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(10, minimize=minimize),
                -25. if minimize else 100.,
            )
            #     1D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [1, 2, 10],
                    minimize=minimize,
                ).tolist(),
                [0., 0., -25.]
                if minimize else
                [0., 50., 100.],
            )
            #     2D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [
                        [1, 2, 10],
                        [10, 2, 1],
                    ],
                    minimize=minimize,
                ).tolist(),
                [
                    [0., 0., -25.],
                    [-25., 0., 0.],
                ]
                if minimize else
                [
                    [0., 50., 100.],
                    [100., 50., 0.],
                ],
            )
            #   Test non-integer ns.
            for n in range(1, 11):
                self.assertEqual(
                    dist.naive_tuning_curve(int(n), minimize=minimize),
                    dist.naive_tuning_curve(float(n), minimize=minimize),
                )
            for n in [0.5, 1.5, 10.5]:
                with self.assertRaises(ValueError):
                    dist.naive_tuning_curve(n, minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.naive_tuning_curve([n], minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.naive_tuning_curve([n, 1], minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.naive_tuning_curve([[n], [1]], minimize=minimize)
            #   Test ns <= 0.
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve(0, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve(-1, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([0], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([-2], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([0, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([-2, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([[0], [1]], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([[-2], [1]], minimize=minimize)

            # Test when ys has duplicates.
            ys = [0., 0., 50., -25., 100.]
            dist = nonparametric.EmpiricalDistribution(ys)
            #   Test 0 < ns <= len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(1, minimize=minimize),
            ))
            self.assertEqual(dist.naive_tuning_curve(1, minimize=minimize), 0.)
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(2, minimize=minimize),
            ))
            self.assertEqual(dist.naive_tuning_curve(2, minimize=minimize), 0.)
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(3, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(3, minimize=minimize),
                0. if minimize else 50.,
            )
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(4, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(4, minimize=minimize),
                -25. if minimize else 50.,
            )
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(5, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(5, minimize=minimize),
                -25. if minimize else 100.,
            )
            #     1D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [1, 2, 3, 4, 5],
                    minimize=minimize,
                ).tolist(),
                [0., 0., 0., -25., -25.]
                if minimize else
                [0., 0., 50., 50., 100.],
            )
            self.assertEqual(
                dist.naive_tuning_curve(
                    [2, 4, 1, 3, 5],
                    minimize=minimize,
                ).tolist(),
                [0., -25., 0., 0., -25.]
                if minimize else
                [0., 50., 0., 50., 100.],
            )
            #     2D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [
                        [1, 2, 3, 4, 5],
                        [2, 4, 1, 3, 5],
                    ],
                    minimize=minimize,
                ).tolist(),
                [
                    [0., 0., 0., -25., -25.],
                    [0., -25., 0., 0., -25.],
                ]
                if minimize else
                [
                    [0., 0., 50., 50., 100.],
                    [0., 50., 0., 50., 100.],
                ],
            )
            #   Test ns > len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(6, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(6, minimize=minimize),
                -25. if minimize else 100.,
            )
            self.assertTrue(np.isscalar(
                dist.naive_tuning_curve(10, minimize=minimize),
            ))
            self.assertEqual(
                dist.naive_tuning_curve(10, minimize=minimize),
                -25. if minimize else 100.,
            )
            #     1D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [1, 2, 10],
                    minimize=minimize,
                ).tolist(),
                [0., 0., -25.]
                if minimize else
                [0., 0., 100.],
            )
            #     2D array
            self.assertEqual(
                dist.naive_tuning_curve(
                    [
                        [1, 2, 10],
                        [10, 2, 1],
                    ],
                    minimize=minimize,
                ).tolist(),
                [
                    [0., 0., -25.],
                    [-25., 0., 0.],
                ]
                if minimize else
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
                dist.naive_tuning_curve(1, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([2, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.naive_tuning_curve([[2], [1]], minimize=minimize)

    def test_v_tuning_curve(self):
        for minimize in [False, True]:
            # Test when len(ys) == 1.
            ys = [42.]
            dist = nonparametric.EmpiricalDistribution(ys)

            self.assertEqual(dist.v_tuning_curve(1, minimize=minimize), 42.)
            self.assertEqual(dist.v_tuning_curve(10, minimize=minimize), 42.)
            self.assertEqual(
                dist.v_tuning_curve([1, 10], minimize=minimize).tolist(),
                [42., 42.],
            )
            self.assertEqual(
                dist.v_tuning_curve(
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
            dist = nonparametric.EmpiricalDistribution(ys)
            curve = np.mean(
                np.minimum.accumulate(
                    self.generator.choice(ys, size=(2_500, 7), replace=True),
                    axis=1,
                )
                if minimize else
                np.maximum.accumulate(
                    self.generator.choice(ys, size=(2_500, 7), replace=True),
                    axis=1,
                ),
                axis=0,
            )
            #   Test 0 < ns <= len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(1, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(1, minimize=minimize),
                curve[0],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(3, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(3, minimize=minimize),
                curve[2],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(5, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(5, minimize=minimize),
                curve[4],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve([1, 2, 3, 4, 5], minimize=minimize),
                curve[:5],
                atol=5.,
            ))
            self.assertTrue(np.allclose(
                dist.v_tuning_curve([2, 4, 1, 3, 5], minimize=minimize),
                [curve[1], curve[3], curve[0], curve[2], curve[4]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve(
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
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(6, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(6, minimize=minimize),
                curve[5],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(7, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(7, minimize=minimize),
                curve[6],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve([1, 2, 7], minimize=minimize),
                [curve[0], curve[1], curve[6]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve(
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
            #   Test non-integer ns.
            for n in range(1, 11):
                self.assertEqual(
                    dist.v_tuning_curve(int(n), minimize=minimize),
                    dist.v_tuning_curve(float(n), minimize=minimize),
                )
            for n in [0.5, 1.5, 10.5]:
                with self.assertRaises(ValueError):
                    dist.v_tuning_curve(n, minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.v_tuning_curve([n], minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.v_tuning_curve([n, 1], minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.v_tuning_curve([[n], [1]], minimize=minimize)
            #   Test ns <= 0.
            with self.assertRaises(ValueError):
                dist.v_tuning_curve(0, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve(-1, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([0], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([-2], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([0, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([-2, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([[0], [1]], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([[-2], [1]], minimize=minimize)

            # Test when ys has duplicates.
            ys = [0., 0., 50., 0., 100.]
            dist = nonparametric.EmpiricalDistribution(ys)
            curve = np.mean(
                np.minimum.accumulate(
                    self.generator.choice(ys, size=(2_500, 7), replace=True),
                    axis=1,
                )
                if minimize else
                np.maximum.accumulate(
                    self.generator.choice(ys, size=(2_500, 7), replace=True),
                    axis=1,
                ),
                axis=0,
            )
            #   Test 0 < ns <= len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(1, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(1, minimize=minimize),
                curve[0],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(3, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(3, minimize=minimize),
                curve[2],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(5, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(5, minimize=minimize),
                curve[4],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve([1, 2, 3, 4, 5], minimize=minimize),
                curve[:5],
                atol=5.,
            ))
            self.assertTrue(np.allclose(
                dist.v_tuning_curve([2, 4, 1, 3, 5], minimize=minimize),
                [curve[1], curve[3], curve[0], curve[2], curve[4]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve(
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
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(6, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(6, minimize=minimize),
                curve[5],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.v_tuning_curve(7, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.v_tuning_curve(7, minimize=minimize),
                curve[6],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve([1, 2, 7], minimize=minimize),
                [curve[0], curve[1], curve[6]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.v_tuning_curve(
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

            # Test when ws != None.
            ys = [-1, 0, 1]
            ws = [0.1, 0.5, 0.4]
            dist = nonparametric.EmpiricalDistribution(ys, ws=ws)

            with self.assertRaises(ValueError):
                dist.v_tuning_curve(1, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([2, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.v_tuning_curve([[2], [1]], minimize=minimize)

    def test_u_tuning_curve(self):
        for minimize in [False, True]:
            # Test when len(ys) == 1.
            ys = [42.]
            dist = nonparametric.EmpiricalDistribution(ys)

            self.assertEqual(dist.u_tuning_curve(1, minimize=minimize), 42.)
            self.assertEqual(dist.u_tuning_curve(10, minimize=minimize), 42.)
            self.assertEqual(
                dist.u_tuning_curve([1, 10], minimize=minimize).tolist(),
                [42., 42.],
            )
            self.assertEqual(
                dist.u_tuning_curve(
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
            dist = nonparametric.EmpiricalDistribution(ys)
            curve = np.mean(
                np.minimum.accumulate(
                    # Sort random numbers to batch sampling without replacement.
                    np.array(ys)[np.argsort(
                        self.generator.random(size=(2_500, 5)),
                        axis=1,
                    )],
                    axis=1,
                )
                if minimize else
                np.maximum.accumulate(
                    # Sort random numbers to batch sampling without replacement.
                    np.array(ys)[np.argsort(
                        self.generator.random(size=(2_500, 5)),
                        axis=1,
                    )],
                    axis=1,
                ),
                axis=0,
            )
            #   Test 0 < ns <= len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(1, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(1, minimize=minimize),
                curve[0],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(3, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(3, minimize=minimize),
                curve[2],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(5, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(5, minimize=minimize),
                curve[4],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve([1, 2, 3, 4, 5], minimize=minimize),
                curve,
                atol=5.,
            ))
            self.assertTrue(np.allclose(
                dist.u_tuning_curve([2, 4, 1, 3, 5], minimize=minimize),
                [curve[1], curve[3], curve[0], curve[2], curve[4]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve(
                    [
                        [1, 2, 3, 4, 5],
                        [2, 4, 1, 3, 5],
                    ],
                    minimize=minimize,
                ),
                [
                    curve,
                    [curve[1], curve[3], curve[0], curve[2], curve[4]],
                ],
                atol=5.,
            ))
            #   Test ns > len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(6, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(6, minimize=minimize),
                curve[4],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(7, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(7, minimize=minimize),
                curve[4],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve([1, 2, 7], minimize=minimize),
                [curve[0], curve[1], curve[4]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve(
                    [
                        [1, 2, 7],
                        [6, 2, 1],
                    ],
                    minimize=minimize,
                ),
                [
                    [curve[0], curve[1], curve[4]],
                    [curve[4], curve[1], curve[0]],
                ],
                atol=5.,
            ))
            #   Test non-integer ns.
            for n in range(1, 11):
                self.assertEqual(
                    dist.u_tuning_curve(int(n), minimize=minimize),
                    dist.u_tuning_curve(float(n), minimize=minimize),
                )
            for n in [0.5, 1.5, 10.5]:
                with self.assertRaises(ValueError):
                    dist.u_tuning_curve(n, minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.u_tuning_curve([n], minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.u_tuning_curve([n, 1], minimize=minimize)
                with self.assertRaises(ValueError):
                    dist.u_tuning_curve([[n], [1]], minimize=minimize)
            #   Test ns <= 0.
            with self.assertRaises(ValueError):
                dist.u_tuning_curve(0, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve(-1, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([0], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([-2], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([0, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([-2, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([[0], [1]], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([[-2], [1]], minimize=minimize)

            # Test when ys has duplicates.
            ys = [0., 0., 50., 0., 100.]
            dist = nonparametric.EmpiricalDistribution(ys)
            curve = np.mean(
                np.minimum.accumulate(
                    # Sort random numbers to batch sampling without replacement.
                    np.array(ys)[np.argsort(
                        self.generator.random(size=(2_500, 5)),
                        axis=1,
                    )],
                    axis=1,
                )
                if minimize else
                np.maximum.accumulate(
                    # Sort random numbers to batch sampling without replacement.
                    np.array(ys)[np.argsort(
                        self.generator.random(size=(2_500, 5)),
                        axis=1,
                    )],
                    axis=1,
                ),
                axis=0,
            )
            #   Test 0 < ns <= len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(1, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(1, minimize=minimize),
                curve[0],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(3, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(3, minimize=minimize),
                curve[2],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(5, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(5, minimize=minimize),
                curve[4],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve([1, 2, 3, 4, 5], minimize=minimize),
                curve,
                atol=5.,
            ))
            self.assertTrue(np.allclose(
                dist.u_tuning_curve([2, 4, 1, 3, 5], minimize=minimize),
                [curve[1], curve[3], curve[0], curve[2], curve[4]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve(
                    [
                        [1, 2, 3, 4, 5],
                        [2, 4, 1, 3, 5],
                    ],
                    minimize=minimize,
                ),
                [
                    curve,
                    [curve[1], curve[3], curve[0], curve[2], curve[4]],
                ],
                atol=5.,
            ))
            #   Test ns > len(ys).
            #     scalar
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(6, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(6, minimize=minimize),
                curve[4],
                delta=5.,
            )
            self.assertTrue(np.isscalar(
                dist.u_tuning_curve(7, minimize=minimize),
            ))
            self.assertAlmostEqual(
                dist.u_tuning_curve(7, minimize=minimize),
                curve[4],
                delta=5.,
            )
            #     1D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve([1, 2, 7], minimize=minimize),
                [curve[0], curve[1], curve[4]],
                atol=5.,
            ))
            #     2D array
            self.assertTrue(np.allclose(
                dist.u_tuning_curve(
                    [
                        [1, 2, 7],
                        [6, 2, 1],
                    ],
                    minimize=minimize,
                ),
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
                dist.u_tuning_curve(1, minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([2, 1], minimize=minimize)
            with self.assertRaises(ValueError):
                dist.u_tuning_curve([[2], [1]], minimize=minimize)

    @pytest.mark.level(2)
    def test_confidence_bands(self):
        n = 5
        methods = ["dkw", "ks", "ld_equal_tailed", "ld_highest_density"]
        for method in methods:
            for confidence in [0.5, 0.9]:
                for a, b in [(0., 1.), (-np.inf, np.inf)]:
                    for has_duplicates in [False, True]:
                        if has_duplicates:
                            ys = self.generator.uniform(0, 1, size=n)
                            ys = np.concatenate([ys[:n-n//3], ys[:n//3]])
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message=r"Duplicates detected in ys",
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
                            ys = self.generator.uniform(0, 1, size=n)
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
                        self.assertGreaterEqual(lo.cdf(a), 0.)
                        self.assertGreaterEqual(dist.cdf(a), 0.)
                        self.assertGreaterEqual(hi.cdf(a), 0.)
                        #     upper bound
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
                        self.assertLessEqual(lo.cdf(a), 1.)
                        self.assertLessEqual(dist.cdf(a), 1.)
                        self.assertLessEqual(hi.cdf(a), 1.)
                        #     upper bound
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
                        # Check the bands are the right distance from the eCDF.
                        if method in ("dkw", "ks"):
                            epsilon = (
                                utils.dkw_epsilon(n, confidence)
                                if method == "dkw" else
                                stats.kstwo(n).ppf(confidence)
                            )
                            self.assertTrue(np.allclose(
                                (dist.cdf(ys) - lo.cdf(ys))[
                                    dist.cdf(ys) > epsilon
                                ],
                                epsilon,
                            ))
                            self.assertTrue(np.allclose(
                                (hi.cdf(ys) - dist.cdf(ys))[
                                    1. - dist.cdf(ys) > epsilon
                                ],
                                epsilon,
                            ))

    def test_sample_defaults_to_global_random_number_generator(self):
        # sample should be deterministic if global seed is set.
        dist = nonparametric.EmpiricalDistribution(range(16))
        #   Before setting the seed, two samples should be unequal.
        self.assertNotEqual(dist.sample(16).tolist(), dist.sample(16).tolist())
        #   After setting the seed, two samples should be unequal.
        opda.random.set_seed(0)
        self.assertNotEqual(dist.sample(16).tolist(), dist.sample(16).tolist())
        #   Resetting the seed should produce the same sample.
        opda.random.set_seed(0)
        first_sample = dist.sample(16)
        opda.random.set_seed(0)
        second_sample = dist.sample(16)
        self.assertEqual(first_sample.tolist(), second_sample.tolist())

    def test_sample_is_deterministic_given_generator_argument(self):
        dist = nonparametric.EmpiricalDistribution(range(16))
        # Reusing the same generator, two samples should be unequal.
        generator = np.random.default_rng(0)
        self.assertNotEqual(
            dist.sample(16, generator=generator).tolist(),
            dist.sample(16, generator=generator).tolist(),
        )
        # Using generators in the same state should produce the same sample.
        self.assertEqual(
            dist.sample(16, generator=np.random.default_rng(0)).tolist(),
            dist.sample(16, generator=np.random.default_rng(0)).tolist(),
        )

    def test_ppf_is_almost_sure_left_inverse_of_cdf(self):
        # NOTE: In general, the quantile function is an almost sure left
        # inverse of the cumulative distribution function.
        for has_ws in [False, True]:
            for _ in range(10):
                dist = nonparametric.EmpiricalDistribution(
                    self.generator.uniform(0, 1, size=5),
                    ws=self.generator.dirichlet(np.ones(5)) if has_ws else None,
                )
                ys = dist.sample(100)
                self.assertEqual(dist.ppf(dist.cdf(ys)).tolist(), ys.tolist())

    def test_ppf_at_extreme_values(self):
        for has_ws in [False, True]:
            for a, b in [
                    [-np.inf, np.inf],
                    [-np.inf, 1.],
                    [-1., np.inf],
                    [-1., 1.],
            ]:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"ws gives equal weight to each sample.",
                        category=RuntimeWarning,
                    )
                    dist = nonparametric.EmpiricalDistribution(
                        [0.],
                        ws=[1.] if has_ws else None,
                        a=a,
                        b=b,
                    )
                self.assertEqual(dist.ppf(0. - 1e-12), a)
                self.assertEqual(dist.ppf(0.), a)
                self.assertEqual(dist.ppf(1.), 0.)
                self.assertEqual(dist.ppf(1. + 1e-12), 0.)

    def test_quantile_tuning_curve_minimize_is_dual_to_maximize(self):
        for _ in range(4):
            for use_weights in [False, True]:
                ys = self.generator.normal(size=7)
                # NOTE: This test must use an _odd_ number of ys. When
                # the sample size is even, there is no exact
                # median. Our definition takes the order statistic to
                # the left of the middle. Thus, the median for ys and
                # -ys differs. Avoid this issue by using an odd number.
                ws = (
                    self.generator.dirichlet(np.ones_like(ys))
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
                    expected = (
                        expected_minimize
                        if minimize else
                        expected_maximize
                    )
                    ws = (
                        np.ones_like(ys) / len(ys)
                        if use_weights else
                        None
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"ws gives equal weight to each sample.",
                            category=RuntimeWarning,
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

    def test_average_tuning_curve_minimize_is_dual_to_maximize(self):
        for _ in range(4):
            for use_weights in [False, True]:
                ys = self.generator.normal(size=8)
                ws = (
                    self.generator.dirichlet(np.ones_like(ys))
                    if use_weights else
                    None
                )
                ns = np.arange(1, 17)
                self.assertTrue(np.allclose(
                    nonparametric
                      .EmpiricalDistribution(ys, ws=ws)
                      .average_tuning_curve(ns, minimize=False),
                    -nonparametric
                      .EmpiricalDistribution(-ys, ws=ws)
                      .average_tuning_curve(ns, minimize=True),
                ))
                self.assertTrue(np.allclose(
                    nonparametric
                      .EmpiricalDistribution(ys, ws=ws)
                      .average_tuning_curve(ns, minimize=True),
                    -nonparametric
                      .EmpiricalDistribution(-ys, ws=ws)
                      .average_tuning_curve(ns, minimize=False),
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
                        self.generator.dirichlet(np.ones_like(ys))
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
                                "ignore",
                                message=r"invalid value encountered in reduce",
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
                            ys, confidence, method="dkw",
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15)),
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
                            ys, confidence, method="ks",
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15)),
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
                            ys, confidence, method="ld_equal_tailed",
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15)),
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
                            ys, confidence, method="ld_highest_density",
                        )
                    # NOTE: Since the confidence bands are step
                    # functions and the CDF is increasing, if there's a
                    # violation of the confidence bands then there will
                    # be one just before the discontinuity for the upper
                    # band and at the discontinuity for the lower band.
                    covered.append(
                        np.all(lo.cdf(ys) <= dist.cdf(ys))
                        & np.all(dist.cdf(ys - 1e-15) <= hi.cdf(ys - 1e-15)),
                    )

                lo, hi = utils.binomial_confidence_interval(
                    n_successes=np.sum(covered),
                    n_total=n_trials,
                    confidence=0.999999,
                )
                self.assertLess(lo, confidence)
                self.assertGreater(hi, confidence)

    @pytest.mark.level(3)
    def test_confidence_bands_defaults_to_global_random_number_generator(self):
        # confidence_bands should be deterministic if global seed is set.
        ys = np.arange(2)
        confidence = 0.50
        for method in ["ld_equal_tailed", "ld_highest_density"]:
            #   Before setting the seed, confidence_bands should be
            #   non-deterministic.
            first_lo, _, first_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                )

            nonparametric._ld_band_weights.cache_clear()  # noqa: SLF001
            second_lo, _, second_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                )

            self.assertNotEqual(first_lo, second_lo)
            self.assertNotEqual(first_hi, second_hi)

            #   After setting the seed, confidence_bands should be
            #   non-deterministic.
            opda.random.set_seed(0)
            first_lo, _, first_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                )

            nonparametric._ld_band_weights.cache_clear()  # noqa: SLF001
            second_lo, _, second_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                )

            self.assertNotEqual(first_lo, second_lo)
            self.assertNotEqual(first_hi, second_hi)

            #   Resetting the seed should make confidence_bands deterministic.
            opda.random.set_seed(0)
            first_lo, _, first_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                )

            opda.random.set_seed(0)
            nonparametric._ld_band_weights.cache_clear()  # noqa: SLF001
            second_lo, _, second_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                )

            self.assertEqual(first_lo, second_lo)
            self.assertEqual(first_hi, second_hi)

    @pytest.mark.level(3)
    def test_confidence_bands_is_deterministic_given_generator_argument(self):
        ys = np.arange(2)
        confidence = 0.50
        for method in ["ld_equal_tailed", "ld_highest_density"]:
            # Reusing the same generator, confidence_bands should be
            # non-deterministic.
            generator = np.random.default_rng(0)

            first_lo, _, first_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                    generator=generator,
                )

            nonparametric._ld_band_weights.cache_clear()  # noqa: SLF001
            second_lo, _, second_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                    generator=generator,
                )

            self.assertNotEqual(first_lo, second_lo)
            self.assertNotEqual(first_hi, second_hi)

            # Using generators in the same state, confidence_bands should be
            # deterministic.
            first_lo, _, first_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                    generator=np.random.default_rng(0),
                )

            nonparametric._ld_band_weights.cache_clear()  # noqa: SLF001
            second_lo, _, second_hi =\
                nonparametric.EmpiricalDistribution.confidence_bands(
                    ys,
                    confidence,
                    method=method,
                    generator=np.random.default_rng(0),
                )

            self.assertEqual(first_lo, second_lo)
            self.assertEqual(first_hi, second_hi)
