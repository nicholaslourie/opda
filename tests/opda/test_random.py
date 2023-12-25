"""Tests for opda.random."""

import unittest

import numpy as np

import opda.random


class SetSeedTestCase(unittest.TestCase):
    """Test opda.random.set_seed."""

    def test_set_seed(self):
        # Test set_seed uses system entropy when given None.
        opda.random.set_seed(seed=None)
        u0 = opda.random.DEFAULT_GENERATOR.random()
        opda.random.set_seed(seed=None)
        u1 = opda.random.DEFAULT_GENERATOR.random()
        self.assertNotEqual(u0, u1)

        # Test set_seed sets DEFAULT_GENERATOR's seed.
        opda.random.set_seed(seed=0)
        u0 = opda.random.DEFAULT_GENERATOR.random()
        opda.random.set_seed(seed=0)
        u1 = opda.random.DEFAULT_GENERATOR.random()
        self.assertEqual(u0, u1)

        # Test set_seed sets DEFAULT_GENERATOR to a generator if given one.
        #   np.random.RandomState
        generator = np.random.RandomState()
        opda.random.set_seed(generator)
        self.assertIs(opda.random.DEFAULT_GENERATOR, generator)
        #   np.random.Generator
        generator = np.random.default_rng()
        opda.random.set_seed(generator)
        self.assertIs(opda.random.DEFAULT_GENERATOR, generator)


class DefaultGeneratorTestCase(unittest.TestCase):
    """Test opda.random.DEFAULT_GENERATOR."""

    def test_default_generator(self):
        # Test DEFAULT_GENERATOR has the expected type.
        self.assertIsInstance(
            opda.random.DEFAULT_GENERATOR,
            np.random.Generator,
        )

        # Test DEFAULT_GENERATOR can generate random values.
        self.assertGreaterEqual(opda.random.DEFAULT_GENERATOR.random(), 0.)
        self.assertLessEqual(opda.random.DEFAULT_GENERATOR.random(), 1.)
        self.assertNotEqual(
            opda.random.DEFAULT_GENERATOR.random(),
            opda.random.DEFAULT_GENERATOR.random(),
        )
