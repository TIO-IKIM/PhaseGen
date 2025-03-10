# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import argparse, yaml
from utils.IKIMLogger import IKIMLogger
from utils.create_dataset import get_test_loader
import numpy as np
from PIL import Image
from utils.utilities import (
    complex_save_predictions_as_imgs,
    dice,
    hausdorff,
    accuracy,
)
import models.layer.activations as A
import models.cResUNet as RESUNET
import utils.utilities as utilities
from scipy import ndimage
import logging
import os
import pickle
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(
    prog="Tester",
    description="Test a trained network.",
)
parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default=Path.home() / "k-Strip/src/configs/test_skullstrip.yaml",
)
parser.add_argument(
    "--e",
    action="store_false",
    default=True,
    help="Start inference. If selected, start inference. Defaults to True.",
)
parser.add_argument(
    "--perc",
    type=float,
    default=None,
    help="If selected choose single percentage value for binary masking in evaluation. \
        If None, iterate through range of values in case of evaluation. Defaults to None",
)
parser.add_argument(
    "--num_zero",
    type=int,
    default=5000,
    help="Choose one number of minmal amount non-zero pixels in segmentation masks to be taken into account in evaluation. \
    Use None to iterate through range of values in case of evaluation. Defaults to None",
)
parser.add_argument(
    "--eval_num",
    type=int,
    default=None,
    help="Define if not all samples are supposed to be used for evaluation. Defaults to None.",
)
parser.add_argument(
    "--c",
    type=int,
    default=None,
    help=(
        "Checkpoint number to be loaded. For this the path to the checkpoints need to be defined in the config file. "
        "Defaults to None."
    ),
)
parser.add_argument("--device", type=int, default=0, help="Cuda device to run test on.")


class Tester:
    def __init__(self, args: dict, config: dict) -> None:
        self.bootstrapping = True
        self.model_path = Path.home() / config["model_path"]
        self.base_path = Path.home() / config["base_path"]
        self.save_path = Path.home() / f"{config['save_path']}/{self.model_path.name}"
        self.test_path = Path(config["test_path"])
        self.config_path = config["config_path"]
        self.inference = args.e
        self.eval_num = args.eval_num
        self.filenum = config["filenum"]
        self.num_workers = config["num_workers"]
        self.pin_memory = config["pin_memory"]
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )

        assert args.perc != None or args.num_zero != None, logger.critical(
            "Choose at least one: --perc or --num_zero"
        )
        if args.perc == None:
            self.perc = np.linspace(
                config["perc"][0], config["perc"][1], config["perc"][2]
            )  # set different threshold percentage values
        else:
            self.perc = [args.perc]  # single threshold
        if args.num_zero == None:
            self.num_zero = np.linspace(
                config["num_zero"][0], config["num_zero"][1], config["num_zero"][2]
            )  # minimal amount of non-zero pixels in segmentation
        else:
            self.num_zero = [args.num_zero]
        self.transformation = config["transformation"]

    def load_model(self) -> None:
        self.model = torch.load(self.model_path, map_location=self.device)
        logging.info(f"Model {self.model_path.name} loaded!")

    def load_checkpoint(self, c: int) -> None:
        """Load a checkpoint of the given model.

        Args:
            c (int): Number of the checkpoint ot be loaded.
        """
        with open(self.config_path, "r") as conf:
            config = yaml.safe_load(conf)

        checkpoint_path = self.model_path / Path(f"checkpoint_{c}.pth.tar")

        Tester._init_activation(self, config)
        Tester._init_network(self, config)

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)["state_dict"],
            self.model,
        )

    def _init_activation(self, config):
        """Selects the activation function for training based on the input configuration.

        Args:
            config: A dictionary containing the configuration for the training process.

        Returns:
            None. The function sets the `activation` attribute in the current object to the selected loss.

        Raises:
            Exception: If the activation function specified in the input configuration is not a valid activation function.
        """

        activation_functions = {
            "elu": A.ComplexELU,
            "relu": A.ComplexReLU,
            "lrelu": A.ComplexLReLU,
            "palrelu": A.PhaseAmplitudeReLU,
            "selu": A.ComplexSELU,
            "cardioid": A.ComplexCardioid(),
        }
        selected_activation = config["activation"]
        assert (
            selected_activation in activation_functions
        ), f"{selected_activation} is not a valid activation function! \n Valid activation functions are: \n {list(activation_functions.keys())}"
        self.activation = activation_functions[selected_activation]

    def _init_network(self, config):
        """Selects the appropriate model for training based on the input configuration.

        Args:
            config: A dictionary containing the configuration for the training process.
            device: The device to be used for training (GPU or CPU).
            padding: The padding value to be used for padding the input images.
            dilation: The dilation rate to be used for dilation of the input images.

        Returns:
            None. The function sets the `network` attribute in the current object to the selected model.

        Raises:
            Exception: If the model specified in the input configuration is not a valid model.
        """
        network_classes = {
            "resunet": RESUNET.ResUNet,
        }
        network_class = network_classes.get(config["model"])
        if network_class is None:
            raise ValueError("Select valid model!")

        self.model = network_class(
            config,
            features=config["features"],
            device=self.device,
            activation=self.activation,
            dim=config["dim"],
            padding=int((config["kernel_size"] - 1) / 2),
            dilation=config["dilation"],
            in_channels=1,
            out_channels=1,
        )
        if type(self.device) is not list:
            self.model.to(self.device)

    def bootstrap(self, percentage, seg_threshold):
        rng = np.random.RandomState(seed=12345)
        n_iterations = 30000
        n_size = int(self.iter)
        boot_acc = []
        boot_speci = []
        boot_sensi = []
        boot_dsc = []
        boot_hd = []

        logger.info(f"Starting bootstrapping with {n_iterations} iterations!")

        for i in range(n_iterations):
            # logger.info(f"Iteration {i} ...")
            sample = rng.randint(n_size)
            single_dice_score, single_HD, acc, speci, sensi = Tester.calculate_metrics(
                self, sample, percentage, seg_threshold
            )

            boot_acc.append(acc)
            boot_speci.append(speci)
            boot_sensi.append(sensi)
            boot_dsc.append(single_dice_score)
            boot_hd.append(single_HD)

        logger.info(
            f"Acc: {statistics.mean(boot_acc)} | Speci: {statistics.mean(boot_speci)} | Sensi: {statistics.mean(boot_sensi)}"
            f"DSC (mean): {statistics.mean(boot_dsc)} | HD (mean): {statistics.mean(boot_hd)}"
            f"| DSC (std): {np.percentile(boot_dsc, [75 ,25])} | HD (std): {np.percentile(boot_hd, [75 ,25])}"
        )

    def calculate_metrics(self, epoch, percentage, seg_threshold):
        returner = False
        while returner == False:
            pred_array = np.rot90(
                np.array(Image.open(f"{self.save_path}/pred_seg_{epoch}.png"))
            )
            target_array = np.rot90(
                np.array(Image.open(f"{self.save_path}/{epoch}.png"))
            )
            orig_array = np.rot90(
                np.array(Image.open(f"{self.save_path}/orig_{epoch}.png"))
            )
            kspace_orig_array = np.rot90(
                np.array(Image.open(f"{self.save_path}/orig_seg_kspace_{epoch}.png"))
            )

            pred = torch.Tensor(pred_array.copy())
            target = torch.Tensor(target_array.copy())

            binary_pred = torch.Tensor((pred > (pred.mean() * percentage)).float())
            binary_pred = ndimage.binary_fill_holes(binary_pred[:, :, 0]).astype(int)
            binary_target = (torch.Tensor((target > 1).float()))[:, :, 0].numpy()

            if np.count_nonzero(binary_target) > seg_threshold:
                single_HD = hausdorff(binary_pred, binary_target)

                single_dice_score = dice(binary_pred, binary_target)

                acc, speci, sensi = accuracy(binary_pred, binary_target)

                returner = True

            else:
                epoch += 1

        return single_dice_score, single_HD, acc, speci, sensi

    def evaluation(self) -> None:
        """
        This code performs evaluation metrics calculation for segmentation results.
        It loops over different percentage thresholds and segmentation thresholds and calculates metrics such as dice score, Hausdorff distance, accuracy, specificity, and sensitivity.
        The segmentation results are obtained from input image files and compared to target image files.

        Args:
            perc (list): A list of float values representing percentage thresholds for binary segmentation multiplied with the mean value of the image.
            num_zero (list): A list of integer values representing segmentation thresholds of the segmentation mask. The lower the value,
                the more slices will be considered in the evaluation, as these will contain less and less brain tissue.
            iter (int): The number of iterations to run.
            save_path (str): A string representing the directory where input and target image files are stored.

        Attributes:
            dice_list (list): A list of dice scores for each iteration.
            HD_list (list): A list of Hausdorff distances for each iteration.
            dice_score (float): The average dice score across all iterations and all segmentation results that pass the threshold.
            HD (float): The average Hausdorff distance across all iterations and all segmentation results that pass the threshold.

        Methods:
            accuracy(binary_pred, binary_target): Calculates the accuracy, specificity, and sensitivity between binary predictions and binary targets.
            dice(binary_pred, binary_target): Calculates the dice score between binary predictions and binary targets.
            hausdorff(binary_pred, binary_target): Calculates the Hausdorff distance between binary predictions and binary targets.

        Returns:
            None
        """

        self.dice_list = []
        self.HD_list = []

        for percentage in self.perc:
            for seg_threshold in self.num_zero:
                dice_max, hd_max, acc_max, speci_max, sensi_max = 0, 0, 0, 0, 0
                dice_min, hd_min, acc_min, speci_min, sensi_min = 1, 1, 1, 1, 1

                dice_score, HD, acc_, speci_, sensi_ = 0, 0, 0, 0, 0
                dice_num = 0
                plot = 0

                for epoch in range(self.iter):
                    logger.debug(
                        f"Iteration {epoch} Perc {percentage} Threshold {seg_threshold} pixels"
                    )

                    (
                        single_dice_score,
                        single_HD,
                        acc,
                        speci,
                        sensi,
                    ) = Tester.calculate_metrics(self, epoch, percentage, seg_threshold)

                    dice_num += 1

                    hd_max, hd_min = max(hd_max, single_HD), min(hd_min, single_HD)
                    HD += single_HD

                    dice_max, dice_min = max(dice_max, single_dice_score), min(
                        dice_min, single_dice_score
                    )
                    dice_score += single_dice_score

                    acc_max, acc_min = max(acc_max, acc), min(acc_min, acc)
                    speci_max, speci_min = max(speci_max, speci), min(speci_min, speci)
                    sensi_max, sensi_min = max(sensi_max, sensi), min(sensi_min, sensi)
                    acc_ += acc
                    speci_ += speci
                    sensi_ += sensi

                    logger.debug(
                        (
                            f"DSC: {single_dice_score:.3f}, HD: {single_HD:.3f}, Accuracy: {acc:.3f}, Specificity: {speci:.3f}, Sensitivity: {sensi:.3f}"
                        )
                    )

                self.dice_score = dice_score / dice_num
                self.HD = HD / dice_num
                acc_ /= dice_num
                speci_ /= dice_num
                sensi_ /= dice_num
                self.dice_list.append(self.dice_score)
                self.HD_list.append(self.HD)

                logger.info(
                    f"Results for the configuration: Perc {percentage} Threshold {seg_threshold} pixels\n"
                    f"DSC: {self.dice_score:.3f}, DSC_min: {dice_min:.3f}, DSC_max: {dice_max:.3f}\n"
                    f"HD: {self.HD:.3f}, HD_min: {hd_min:.3f}, HD_max: {hd_max:.3f}\n"
                    f"Accuracy: {acc_:.3f}, Acc_min: {acc_min:.3f}, Acc_max: {acc_max:.3f}\n"
                    f"Specificity: {speci_:.3f}, Spec_min: {speci_min:.3f}, Speci_max: {speci_max:.3f}\n"
                    f"Sensitivity: {sensi_:.3f}, Sensi_min: {sensi_min:.3f}, Sensi_max: {sensi_max:.3f}\n"
                )

    def plotting(self) -> None:
        """
        Plotting method that plots the data in `self.dice_list` and `self.HD_list` against
        `self.num_zero` and `self.perc`.

        If `len(self.num_zero) > 1`, it creates two subplots (1 row and 2 columns), plots the data and saves the
        plot as "Num_zero_curve.png". The data is also saved in "tumor_threshold_lists.pickle".

        If `len(self.perc) > 1`, it creates two subplots (1 row and 2 columns), plots the data and saves the
        plot as "Perc_curve.png". The data is also saved in "perc_lists.pickle".

        If neither `len(self.num_zero) > 1` nor `len(self.perc) > 1`, it logs the total DSC and total HD.
        """

        if len(self.num_zero) > 1:
            self._plot_and_save(
                self.num_zero, "Num_zero_curve.png", "tumor_threshold_lists.pickle"
            )
        if len(self.perc) > 1:
            self._plot_and_save(self.perc, "Perc_curve.png", "perc_lists.pickle")
        else:
            logger.info(
                f"Total DSC: {(self.dice_list[0]):.3f} | Total HD: {(self.HD_list[0]):.3f}"
            )

    def _plot_and_save(self, x, filename, pickle_file):
        with open(os.path.join(self.save_path, pickle_file), "wb") as f:
            pickle.dump((self.dice_list, self.HD_list, x), f)

    @utilities.timer
    def run_test(self):
        self.iter = 3050

        if self.inference == True:
            test_loader, data_length = get_test_loader(
                transform=self.transformation,
                test_path=self.test_path,
                filenum=self.filenum,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            logger.info(f"Inference for {self.model_path.name}.")

            if args.c is not None:
                Tester.load_checkpoint(self, args.c)
            else:
                Tester.load_model(self)

            for i in range(self.iter):
                try:
                    complex_save_predictions_as_imgs(
                        epoch=i,
                        loader=test_loader,
                        model=self.model,
                        num=self.filenum,
                        folder=self.save_path,
                        kdiff=False,
                        iter=i,
                        device=self.device,
                    )
                    logger.info(f"File {i+1}/{self.iter}")

                except Exception as e:
                    logger.error(e)
                    logger.error("Continue with next file!")
                    continue

        if self.inference == False:
            logger.info(f"Evaluation for {self.model_path.name}.")

            if self.bootstrapping == True:
                Tester.bootstrap(self, percentage=2.0, seg_threshold=5000)
            else:
                Tester.evaluation(self)
                Tester.plotting(self)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parser.parse_args()

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(config["num_threads"])

    ikim_logger = IKIMLogger(
        level="INFO",
        log_dir=Path.home() / "k-Strip" / "logs",
        comment=f"test_{Path(config['model_path']).name}",
    )
    logger = ikim_logger.create_logger()

    test = Tester(
        args=args,
        config=config,
    )
    test.run_test()
