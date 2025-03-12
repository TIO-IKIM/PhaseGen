# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(current_dir))
sys.path.append(project_root)

import torch
import unittest
from utils.fourier import ifft, fft


class TestFourierTransforms(unittest.TestCase):
    def test_fft_2d(self):
        scan = torch.randn(4, 4)
        transformed = fft(scan)
        self.assertEqual(transformed.shape, scan.shape)

    def test_ifft_2d(self):
        scan = torch.randn(4, 4)
        transformed = ifft(scan)
        self.assertEqual(transformed.shape, scan.shape)

    def test_fft_3d(self):
        scan = torch.randn(3, 4, 4)
        transformed = fft(scan)
        self.assertEqual(transformed.shape, scan.shape)

    def test_ifft_3d(self):
        scan = torch.randn(3, 4, 4)
        transformed = ifft(scan)
        self.assertEqual(transformed.shape, scan.shape)

    def test_fft_4d(self):
        scan = torch.randn(2, 3, 4, 4)
        transformed = fft(scan)
        self.assertEqual(transformed.shape, scan.shape)

    def test_ifft_4d(self):
        scan = torch.randn(2, 3, 4, 4)
        transformed = ifft(scan)
        self.assertEqual(transformed.shape, scan.shape)

    def test_fft_5d(self):
        scan = torch.randn(2, 2, 3, 4, 4)
        transformed = fft(scan)
        self.assertEqual(transformed.shape, scan.shape)

    def test_ifft_5d(self):
        scan = torch.randn(2, 2, 3, 4, 4)
        transformed = ifft(scan)
        self.assertEqual(transformed.shape, scan.shape)


if __name__ == "__main__":
    unittest.main()
