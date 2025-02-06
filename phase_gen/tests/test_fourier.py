import torch
import unittest
from phase_gen.utils.fourier import ifft, fft

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

    def test_invalid_dimension_fft(self):
        scan = torch.randn(1)
        with self.assertRaises(ValueError):
            fft(scan)

    def test_invalid_dimension_ifft(self):
        scan = torch.randn(1)
        with self.assertRaises(ValueError):
            ifft(scan)

if __name__ == '__main__':
    unittest.main()