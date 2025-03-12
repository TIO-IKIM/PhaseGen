# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import models.layer.activations as A
import models.cDiff as cDiff
from utils.create_dataset import get_loaders, Folds
import argparse
import os


class PhaseGenTester:
    """
    Class for testing the ComplexDiff model.

    Attributes:
        model_path (str): Path to the model directory.
        checkpoint_path (str): Path to the checkpoint file.
        save_path (str): Path to save the generated phase images.
        device (torch.device): Device to run the model on.
        config (dict): Configuration dictionary.
    """

    def __init__(self, model_path, checkpoint_path=None, save_path: str = None):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = self.load_config()
        self.model = self.load_model()
        self.metrics = {
            "ssim_magn": 0,
            "ssim_phase": 0,
        }
        if self.save_path:
            os.makedirs(f"{self.save_path}/orig_phase", exist_ok=True)
            os.makedirs(f"{self.save_path}/gen_phase", exist_ok=True)
        else:
            os.makedirs("./orig_phase", exist_ok=True)
            os.makedirs("./gen_phase", exist_ok=True)

    def load_config(self):
        """
        Loads the configuration file.
        """
        with open(f"{self.model_path}/train.yaml", "r") as conf:
            return yaml.safe_load(conf)

    def load_model(self):
        activation = A.ComplexPReLU
        if self.checkpoint_path:
            model = cDiff.PhaseDiffusionUNet(
                config=self.config,
                features=self.config["features"],
                device=self.device,
                activation=activation,
                in_channels=self.config["in_channel"],
                out_channels=self.config["out_channel"],
            )
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(self.device)
        else:
            model = torch.load(
                f"{self.model_path}/best_checkpoint.pth", map_location=self.device
            )
        model.eval()
        return model

    def get_data_loader(self):
        folds = Folds(csv_path=self.config["csv_path"], n_folds=5)
        fold = folds[0]
        _, val_loader, _ = get_loaders(
            csv_path=self.config["csv_path"],
            train_idx=fold[0],
            val_idx=fold[1],
            transform="standardization",
            batch_size=4,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )
        return val_loader

    def save_images(self, x, pred, k):
        for n in range(x.shape[0]):
            plt.imsave(
                f"{self.save_path}/orig_phase/{k}.png",
                x[n, 0, ...].angle().cpu().numpy(),
                cmap="gray",
            )
            plt.imsave(
                f"{self.save_path}/gen_phase/{k}.png",
                pred[n, 0, ...].angle().cpu().numpy(),
                cmap="gray",
            )
            k += 1
        return k

    def run(self):
        val_loader = self.get_data_loader()
        loop = tqdm(val_loader, position=0, leave=True)
        with torch.no_grad():
            k = 0
            for i, (images, labels) in enumerate(loop):
                x = images.to(self.device)
                pred = self.model.sample(x)
                k = self.save_images(x, pred, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhaseGenTester")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "-o", "--save_path", type=str, required=True, help="Path to the save directory"
    )

    args = parser.parse_args()

    tester = PhaseGenTester(args.m, args.c, args.o)
    tester.run()
