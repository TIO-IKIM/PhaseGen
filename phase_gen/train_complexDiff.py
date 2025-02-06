# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import os
import torch
import argparse
from pathlib import Path
import models.cDiff_new as cDiff
import torch.optim as optim
from tqdm import tqdm
from utils.IKIMLogger import IKIMLogger
from utils.create_dataset import get_loaders, Folds
import utils.utilities as utilities
import models.layer.activations as A
import shutil
import yaml
import numpy as np
import random

parser = argparse.ArgumentParser(
    prog="Training",
)

parser.add_argument("--e", type=int, default=50, help="Number of epochs for training")
parser.add_argument(
    "--log", type=str, default="INFO", help="Define debug level. Defaults to INFO."
)
parser.add_argument(
    "--tqdm", action="store_false", help="If set, do not log training loss via tqdm."
)
parser.add_argument("--gpu", type=int, default=0, help="GPU used for training.")
parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default="train.yaml",
)
parser.add_argument(
    "--c",
    action="store_true",
    help="Continue from best checkpoint.",
)


def set_seed(seed: int = 42) -> None:
    """Set seeds for the libraries numpy, random, torch and torch.cuda.

    Args:
        seed (int, optional): Seed to be used. Defaults to `42`.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"Random seed set as {seed}")


class TrainNetwork:

    def __init__(self, args: dict, config: dict) -> None:
        """
        Initializes the TrainNetwork class with the given arguments and configuration.

        Args:
            args (dict): Command-line arguments.
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.epochs: int = args.e  # Number of total epochs
        self.train_from_checkpoint = (
            args.c
        )  # Whether to load an existing pretrained model or start anew
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.num_threads: int = config[
            "num_threads"
        ]  # Number of threads to use on the Cluster CPUs
        self.num_workers: int = config[
            "num_workers"
        ]  # Number of workers pytorch uses for dataloading
        self.pin_memory: bool = config[
            "pin_memory"
        ]  # Whether to pin memory during training or not (leads to higher efficiency in memory consumption)
        self.csv_path = config["csv_path"]
        self.base_output: Path = (
            Path.home() / config["base_output"]
        )  # Base output path for validation etc.
        self.optim: str = config[
            "optimizer"
        ]  # Optimizer to be used for training (e.g. "Adam")
        self.lr: float = config["lr"]  # Initial learning rate
        self.dropout: float = config["dropout"]  # Dropout
        self.batch_size: int = config["batch_size"]  # Batch size
        self.val_num: int = config[
            "val_num"
        ]  # Number of samples to be used for validation (e.g. 4)
        # self.fourier_weight = config["fourier_weight"]
        if self.val_num > self.batch_size:
            self.val_num = self.batch_size

        TrainNetwork._init_activation(self, config)

        TrainNetwork._init_network(self, config, logger)

        self.model_name = (
            f"{config['loss']}_"
            f"{config['lr']}_"
            f"l{config['length']}_{config['comment']}_{config['features']}"
        )
        self.save_base_folder: Path = (
            Path(self.base_output) / f"train_{self.model_name}"
        )  # Folder to save the validation images and checkpoints in

    def _init_network(self, config):
        """
        Initializes the network with the given configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        self.network = cDiff.PhaseDiffusionUNet(
            config=config,
            features=config["features"],
            device=self.device,
            activation=self.activation,
            in_channels=config["in_channel"],
            out_channels=config["out_channel"],
            logger=logger,
        )
        if type(self.device) is not list:
            self.network.to(self.device)

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
            "prelu": A.ComplexPReLU,
            "palrelu": A.PhaseAmplitudeReLU,
            "selu": A.ComplexSELU,
            "cardioid": A.ComplexCardioid,
            "amprelu": A.AmplitudeRelu,
        }
        selected_activation = config["activation"]
        assert (
            selected_activation in activation_functions
        ), f"{selected_activation} is not a valid activation function! \n Valid activation functions are: \n {list(activation_functions.keys())}"
        self.activation = activation_functions[selected_activation]

    def __repr__(self) -> str:
        """This function just returns out an overview of some important settings.

        Returns:
            str: Comment for a logger or direct printing.
        """
        return f"batch_size = {self.batch_size} loss = {self.loss} lr = {self.lr} kernel_size = {self.kernel_size} pooling_size = {self.pooling_size} {self.model_name}"

    @utilities.timer
    def train_fn(self) -> None:
        """
        Trains the model for one epoch.
        """
        if args.tqdm == True:
            loop = tqdm(self.train_loader, miniters=100)
        else:
            loop = self.train_loader
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=self.device, non_blocking=True)

            loss = self.model(data)

            total_loss += loss.item()

            if torch.isnan(loss):
                raise ValueError("-- Loss NaN --")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            self.optimizer.step()

            if args.tqdm == True:
                loop.set_postfix(loss=loss.item())

        self.scheduler.step()
        self.total_loss = total_loss / len(self.train_loader)

    @utilities.timer
    def validation(self) -> None:
        """
        Validates the model on the validation set.
        """
        metrics = utilities.complex_diffusion_accuracy(
            self.val_loader,
            self.model,
            self.device,
        )
        logger.info(f"Validation loss: {metrics['val_loss']}")

        if self.epoch % 1 == 0 or self.epoch == 0 and self.ndim == 4:
            # print examples to folder
            utilities.get_diffusion_sample(
                epoch=self.epoch,
                loader=self.val_loader,
                model=self.model,
                device=self.device,
                num=self.val_num,
                folder=self.save_folder,
            )

    @utilities.timer
    def __call__(self) -> None:
        """
        Runs the training and validation process for each fold.
        """
        folds = Folds(csv_path=self.csv_path, n_folds=5)

        for self.fold_idx, fold in enumerate(folds):

            logger.info(f"---- Fold {self.fold_idx + 1} of {len(folds)} ----")

            self.save_folder = Path(self.save_base_folder) / f"fold_{self.fold_idx}"
            # Log save folder information
            logger.info(f"Save folder: {str(self.save_folder)}")
            # Create save folder if it does not exist
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)

            self.early_stopping = utilities.EarlyStopping(
                patience=30,
                verbose=True,
                monitor="val-loss",
                op_type="min",
                logger=logger,
            )

            self.train_idx = fold[0]
            self.val_idx = fold[1]

            # Load model
            if self.train_from_checkpoint:
                logger.info("==> load best checkpoint from previous training")
                self.model = torch.load(
                    f"{self.save_folder}/best_checkpoint.pth", map_location=self.device
                )
            else:
                self._init_network(self.config, logger)
                self.model = self.network

            # Log device information and model parameters
            logger.info(f"Device: {self.device}")
            table = utilities.count_parameters(self.model)
            logger.info(f"\n{table}")

            if self.optim in ["AdamW", "Adam", "SGD"]:
                self.optimizer = getattr(optim, self.optim)(
                    self.model.parameters(), lr=self.lr
                )
            else:
                logger.error("Select valid optimizer!")

            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=0.995
            )

            # Get data loaders
            self.train_loader, self.val_loader, self.data_length = get_loaders(
                csv_path=self.csv_path,
                train_idx=self.train_idx,
                val_idx=self.val_idx,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                log=logger,
            )

            # Copy config file to save folder
            shutil.copyfile(args.config, Path(self.save_folder, args.config.name))

            # Start epoch loop
            for self.epoch in range(self.epochs):
                logger.info(f"Now training epoch {self.epoch}!")

                TrainNetwork.train_fn(self)
                logger.info(f"Train-loss: {self.total_loss}")

                # Validate the model
                TrainNetwork.validation(self)

                if self.early_stopping.early_stop:
                    break
                if self.early_stopping.save == True:
                    torch.save(self.model, f"{self.save_folder}/best_checkpoint.pth")
                    logger.info("Save checkpoint.")

            torch.save(self.model, f"{Path(self.save_folder)}/last_checkpoint.pth")

            break


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parser.parse_args()
    args.config = Path(os.path.join("configs", args.config))

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(config["num_threads"])

    ikim_logger = IKIMLogger(
        level=args.log,
        log_dir="logs",
        comment=(
            f"train_{config['loss']}_{config['activation']}_"
            f"p{config['dropout']}_{config['features']}_{config['lr']}_"
            f"{config['optimizer']}_l{config['length']}_{config['comment']}"
        ),
    )
    logger = ikim_logger.create_logger()
    try:
        set_seed(1)
        training = TrainNetwork(
            args=args,
            config=config,
        )
        logger.info(training.__repr__())
        training()
    except Exception as e:
        logger.exception(e)
