# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

import unittest
from unittest.mock import MagicMock
import torch
from utils.utilities import EarlyStopping, padding


class TestUtilities(unittest.TestCase):
    def test_early_stopping(self):
        logger = MagicMock()
        early_stopping = EarlyStopping(
            patience=3,
            verbose=True,
            delta=0.1,
            monitor="val_loss",
            op_type="min",
            logger=logger,
        )

        # Test improvement
        early_stopping(1.0)
        self.assertFalse(early_stopping.early_stop)
        self.assertEqual(early_stopping.counter, 0)

        # Test no improvement
        early_stopping(1.05)
        self.assertFalse(early_stopping.early_stop)
        self.assertEqual(early_stopping.counter, 1)

        # Test patience exceeded
        early_stopping(1.05)
        early_stopping(1.05)
        self.assertTrue(early_stopping.early_stop)

    def test_padding(self):
        pooled_input = torch.randn(1, 1, 4, 4)
        original = torch.randn(1, 1, 6, 6)
        padded = padding(pooled_input, original)
        self.assertEqual(padded.size(), original.size())


if __name__ == "__main__":
    unittest.main()
