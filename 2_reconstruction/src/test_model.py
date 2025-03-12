# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
from pathlib import Path
from utils.create_dataset import get_test_loader
import argparse
from utils.validation import ReconstructionEvaluation
from utils.IKIMLogger import IKIMLogger
from utils.fourier import ifft


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    required=False,
    default="zerofilling",
    help="Path to the model to be tested. If None is selected, zerofilling will be tested.",
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    required=True,
    help="Path to the test data.",
)


class ModelTester:
    """
    A class to test a pre-trained model or zerofilling method for MRI reconstruction.

    Attributes:
        model : torch.nn.Module or None
            The loaded model for testing. If 'zerofilling' is specified, model is None.
        test_loader : DataLoader
            DataLoader for the test dataset.
        device : torch.device
            The device (CPU or GPU) on which the model and data are loaded.
        validator : ReconstructionEvaluation
            An instance of ReconstructionEvaluation to evaluate the reconstruction quality.

    Methods:
        test():
            Tests the model or zerofilling method on the test dataset.
        test_zerofilling():
            Tests the zerofilling method on the test dataset.
        report():
            Logs the mean PSNR, SSIM, NMSE, and MSE metrics for the test results.
    """

    def __init__(self, model_path, data_path, device):
        if model_path == "zerofilling":
            self.model = None
            standardize = False
            normalize = True
        else:
            self.model = torch.load(
                f"{model_path}/best_checkpoint.pth",
                map_location=device,
                weights_only=False,
            )
            standardize = True
            normalize = False
        self.image_domain = True
        self.test_loader = get_test_loader(
            data_path,
            num_workers=4,
            batch_size=8,
            single_coil=True,
            image_domain=self.image_domain,
            standardize=standardize,
        )
        self.device = device
        self.validator = ReconstructionEvaluation(normalize=normalize, verbose=True)

    def test(self):
        """
        Tests the model or zerofilling method on the test dataset.
        If a model is specified, it evaluates the model on the test data.
        If 'zerofilling' is specified, it evaluates the zerofilling method.
        """
        logging.info(f"Will evaluate on {len(self.test_loader.dataset)} samples")
        if self.model:
            logging.info(f"Testing model {Path(args.model_path).name}")
            self.model.eval()
            with torch.no_grad():
                for data, target, kspace_undersampled, scales in self.test_loader:
                    data, target, kspace_undersampled = (
                        data.to(self.device),
                        target.to(self.device),
                        kspace_undersampled.to(self.device),
                    )
                    output = self.model(data, kspace_undersampled)
                    self.validator(output.abs().cpu().numpy(), target.cpu().numpy())
        else:
            logging.info("Testing zerofilling")
            self.test_zerofilling()

    def test_zerofilling(self):
        for data, target, _, scales in self.test_loader:
            if self.image_domain is False:
                data = ifft(data)
            self.validator(data.abs().cpu().numpy(), target.numpy())

    def report(self):
        """
        Logs the mean PSNR, SSIM, NMSE, and MSE metrics for the test results.
        """
        logging.info(f"Mean PSNR: {self.validator['psnr']}")
        logging.info(f"Mean SSIM: {self.validator['ssim']}")
        logging.info(f"Mean NRMSE: {self.validator['nrmse']}")
        logging.info(f"Mean MSE: {self.validator['mse']}")


if __name__ == "__main__":
    args = parser.parse_args()
    ikim_logger = IKIMLogger(
        level="INFO",
        log_dir=".",
        comment=f"test_{Path(args.model_path).name}",
    )
    logging = ikim_logger.create_logger()

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    Tester = ModelTester(args.model_path, args.data_path, device)

    Tester.test()
    Tester.report()
