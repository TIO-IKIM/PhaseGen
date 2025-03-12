# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import unittest
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.utils.validation import mse, nmse, psnr, ssim, ReconstructionEvaluation


class TestValidationMetrics(unittest.TestCase):
    def setUp(self):
        self.gt = np.random.rand(5, 64, 64)
        self.pred = np.random.rand(5, 64, 64)

    def test_mse(self):
        result = mse(self.gt, self.pred)
        expected = np.mean((self.gt - self.pred) ** 2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_nmse(self):
        result = nmse(self.gt, self.pred)
        expected = (
            np.linalg.norm(self.gt - self.pred) ** 2 / np.linalg.norm(self.gt) ** 2
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_psnr(self):
        result = psnr(self.gt, self.pred)
        maxval = self.gt.max()
        expected = peak_signal_noise_ratio(self.gt, self.pred, data_range=maxval)
        self.assertAlmostEqual(result, expected, places=5)

    def test_ssim(self):
        result = ssim(self.gt, self.pred)
        maxval = self.gt.max()
        expected = np.mean(
            [
                structural_similarity(self.gt[i], self.pred[i], data_range=maxval)
                for i in range(self.gt.shape[0])
            ]
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_reconstruction_evaluation(self):
        evaluator = ReconstructionEvaluation()
        evaluator(self.gt, self.pred)
        metrics = evaluator.metrics

        self.assertAlmostEqual(metrics["mse"], mse(self.gt, self.pred), places=5)
        self.assertAlmostEqual(metrics["nmse"], nmse(self.gt, self.pred), places=5)
        self.assertAlmostEqual(metrics["psnr"], psnr(self.gt, self.pred), places=5)
        self.assertAlmostEqual(metrics["ssim"], ssim(self.gt, self.pred), places=5)


if __name__ == "__main__":
    unittest.main()
