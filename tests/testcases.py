"""Base classes for test cases."""

import logging
import unittest

import numpy as np

logger = logging.getLogger(__name__)


class RandomTestCase(unittest.TestCase):
    """A base class for tests involving randomness."""

    def setUp(self):
        self.seed = np.random.SeedSequence().entropy
        self.generator = np.random.default_rng(self.seed)

        logger.info(
            f"Running {self.id()} with random seed: {self.seed}.",
        )
