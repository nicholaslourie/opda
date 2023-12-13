"""Base classes for test cases."""

import logging
import unittest

import numpy as np

import opda.random

logger = logging.getLogger(__name__)


class RandomTestCase(unittest.TestCase):
    """A base class for tests involving randomness."""

    def setUp(self):
        """Configure a random seed and generator for random numbers."""
        # Create the random seed and generator.
        self.seed = np.random.SeedSequence().entropy
        self.generator = np.random.default_rng(self.seed)

        # Use the random seed and generator globally.
        opda.random.set_seed(self.generator)

        # Log the random seed so that tests can be reproduced.
        logger.info(
            f"Running {self.id()} with random seed: {self.seed}.",
        )
