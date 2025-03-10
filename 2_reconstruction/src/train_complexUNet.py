# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import os
import torch
import argparse
from pathlib import Path
import models.cResUNet as RESUNET
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.IKIMLogger import IKIMLogger
from utils.create_dataset import get_loaders
import utils.utilities as utilities
import utils.validation as val
import utils.transforms as T
import models.layer.activations as A
import models.layer.losses as L
import wandb
import shutil
import yaml
import numpy as np
import random
from utils.fourier import ifft

parser = argparse.ArgumentParser(
    prog="Training",
    description="Train a CVNN.",
)

parser.add_argument("--e", type=int, default=50, help="Number of epochs for training")
parser.add_argument(
    "--log", type=str, default="INFO", help="Define debug level. Defaults to INFO."
)
parser.add_argument(
    "--tqdm", action="store_false", help="If set, do not log training loss via tqdm."
)
parser.add_argument(
    "--gpu", type=int, nargs="+", default=0, help="GPU used for training."
)
parser.add_argument(
    "--w",
    action="store_true",
    help="Use Weights & Biases for logging.",
)
parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default="train_skullstrip_resunet2d.yaml",
)
parser.add_argument(
    "--c",
    action="store_true",
    help="Continue from best checkpoint.",
)

parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=None,
    help="Load checkpoint by giving path to checkpoint. Defaults to None.",
)

ID = wandb.util.generate_id()


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


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        monitor="val_loss",
        op_type="min",
        logger=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type
        self.logger = logger

        if self.op_type == "min":
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):
        score = -val_score if self.op_type == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
            self.save = False

    def print_and_update(self, val_score):
        """print_message when validation score decrease."""
        if self.verbose:
            self.logger.info(
                f"{self.monitor} optimized ({self.val_score_min:.6f} --> {val_score:.6f}).",
            )
        self.val_score_min = val_score
        self.save = True


class Trainer:
    """Train a neural network based on PyTorch architecture.

    Args:
        args (dict): Dictionary containing user-specified settings.
        config (dict): Dictionary containing settings set in a yaml-config file.
    """

    def __init__(self, args: dict, config: dict) -> None:
        self.config = config
        self.model: str = config["model"]  # Model name (e.g. "resunet")
        self.epochs: int = args.e  # Number of total epochs
        self.wandb = args.w  # Whether to use Weights & Biases for logging
        self.train_from_checkpoint = (
            args.c
        )  # Whether to load an existing pretrained model or start anew
        self.device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.image_domain = config["image_domain"]
        self.num_threads: int = config[
            "num_threads"
        ]  # Number of threads to use on the Cluster CPUs
        self.num_workers: int = config[
            "num_workers"
        ]  # Number of workers pytorch uses for dataloading
        self.pin_memory: bool = config[
            "pin_memory"
        ]  # Whether to pin memory during training or not (leads to higher efficiency in memory consumption)
        self.train_path = config["train_path"]
        self.val_path = config["val_path"]
        self.csv_path = config["csv_path"]
        self.base_output: Path = (
            Path.home() / config["base_output"]
        )  # Base output path for validation etc.
        self.optim: str = config[
            "optimizer"
        ]  # Optimizer to be used for training (e.g. "Adam")
        self.single_coil: bool = config["single_coil"]
        self.alpha: float = config["alpha"]  # Alpha value for logL1 loss (e.g. "0.8")
        self.lr: float = config["lr"]  # Initial learning rate
        self.dropout: float = config["dropout"]  # Dropout
        self.batch_size: int = config["batch_size"]  # Batch size
        self.kernel_size: int = config[
            "kernel_size"
        ]  # Convolution kernel size (e.g. 3)
        self.padding = int((self.kernel_size - 1) / 2)
        self.pooling_size: int = config[
            "pooling_size"
        ]  # Size of pooling kernel (e.g. 2)
        self.dilation: int = config["dilation"]
        self.val_num: int = config[
            "val_num"
        ]  # Number of samples to be used for validation (e.g. 4)
        # self.fourier_weight = config["fourier_weight"]
        if self.val_num > self.batch_size:
            self.val_num = self.batch_size

        Trainer._init_activation(self, config)

        Trainer._init_network(self, config, logger)

        Trainer._init_loss(self, config)

        self.transformation: str = config["transformation"]
        if (
            config["augmentation"] == "complex_augmentation"
        ):  # Wheter to apply augmentation to the data before training / validation
            self.augmentation = T.periphery_augmentation
        elif config["augmentation"] == None:
            self.augmentation = None
        else:
            logger.error("Select valid augmentation!")
        self.model_name = (
            f"{config['model']}_{config['loss']}_{config['activation']}_"
            f"{config['transformation']}_p{config['dropout']}_{config['features']}_{config['lr']}_"
            f"{config['optimizer']}_l{config['length']}_{config['comment']}"
        )
        self.save_base_folder: Path = (
            Path(self.base_output) / f"{self.model_name}"
        )  # Folder to save the validation images and checkpoints in
        self.group_name = f"{self.model_name}-{ID}"

    def _init_network(self, config, logger):
        """Selects the appropriate model for training based on the input configuration.

        Args:
            config: A dictionary containing the configuration for the training process.
            device: The device to be used for training (GPU or CPU).
            padding: The padding value to be used for padding the input images.
            dilation: The dilation rate to be used for dilation of the input images.
            logger: A logger object for logging information about the training process.

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

        self.network = network_class(
            config=config,
            features=config["features"],
            device=self.device,
            activation=self.activation,
            padding=self.padding,
            dilation=self.dilation,
            in_channels=config["in_channel"],
            out_channels=config["out_channel"],
            image_domain=self.image_domain,
        )
        if type(self.device) is not list:
            self.network.to(self.device)

    def _init_loss(self, config):
        """Selects the loss function for training based on the input configuration.

        Args:
            config: A dictionary containing the configuration for the training process.
            logger: A logger object for logging information about the training process.
            alpha: Value to define the influence of the logarithmic part in case of the l1log-loss function.

        Returns:
            None. The function sets the `loss` attribute in the current object to the selected loss.

        Raises:
            Exception: If the loss specified in the input configuration is not a valid loss.
        """
        loss_functions = {
            "l1": nn.L1Loss(),
            "ssim": L.SSIMLoss(),
            "ssiml1": L.SSIML1Loss(),
        }
        selected_loss = config["loss"]
        assert (
            selected_loss in loss_functions
        ), f"{selected_loss} is not a valid loss function! \n Valid loss functions are: \n {list(loss_functions.keys())}"
        self.loss = loss_functions[selected_loss]

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
            "tanh": A.ComplexTanh,
            "cardioid": A.ComplexCardioid,
            "amprelu": A.AmplitudeRelu,
            "prelu": A.ComplexPReLU,
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

    def first_epoch(self) -> None:
        """Performs the first epoch of training and initiates Weights & Bias connection.

        Args:
            self: Instance of `TrainNetwork` class.

        Returns:
            None.

        Builds the Weights & Bias connection with predefined settings and performs profiling if selected.
        Calls the `train_fn` method at the end.
        """
        if self.wandb:
            wandb.init(
                project="mri_recon_knee",
                group=self.group_name,
                config=self.config,
            )
            wandb.config.update = {
                "learning_rate": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "kernel_size": self.kernel_size,
                "pooling_size": self.pooling_size,
                "loss_function": self.loss,
                "activation": self.activation,
                "optimizer": self.optimizer,
            }

        Trainer.train(self)

    @utilities.timer
    def train(self) -> None:
        """Train function.

        Calculates loss per batch, performs backpropagation and optimizer step.

        Args:
            self: self object of the class.

        Returns:
            None.
        """
        if args.tqdm == True:
            loop = tqdm(self.train_loader, miniters=100)
        else:
            loop = self.train_loader
        total_loss = 0

        for batch_idx, (data, targets, kspace_undersampled, scales) in enumerate(loop):
            data = data.to(device=self.device, non_blocking=True, dtype=torch.complex64)
            kspace_undersampled = kspace_undersampled.to(
                device=self.device, non_blocking=True, dtype=torch.complex64
            )
            # forward
            predictions = self.model(data, kspace_undersampled)

            targets = targets.to(device=predictions.device, non_blocking=True)
            loss = self.loss(predictions.abs(), targets)

            total_loss += loss.item()

            if torch.isnan(loss):
                raise ValueError("-- Loss nan --")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            self.optimizer.step()

            if args.tqdm == True:
                loop.set_postfix(loss=loss.item())
                                                                                                
        self.scheduler.step()
        self.total_loss = total_loss / len(self.train_loader)

        if self.wandb:
            wandb.log({"Train Loss": self.total_loss}, step=self.epoch)

    @utilities.timer
    def validation(self) -> None:
        """Performs validation after each epoch.

        This method saves one batch of the validation set in the save-folder
        and calculates the dice score as well as the validation loss.
        The results are logged to Weights & Biases.

        Args:
            self: Instance of `TrainNetwork` class.

        Returns:
            None
        """
        validator = val.ReconstructionEvaluation(normalize=False)
        self.model.eval()
        val_loss = 0

        plot_idx = random.randint(0, len(self.val_loader))

        with torch.no_grad():
            for batch_idx, (data, targets, kspace_undersampled, scales) in enumerate(self.val_loader):
                data = data.to(device=self.device, dtype=torch.complex64)
                kspace_undersampled = kspace_undersampled.to(
                    device=self.device, dtype=torch.complex64
                )
                predictions = self.model(data, kspace_undersampled)
                targets = targets.to(device=predictions.device)
                predictions = predictions.abs()
                val_loss += self.loss(predictions.abs(), targets).item()
                if self.image_domain:
                    data = data.abs().cpu().numpy()
                    predictions = predictions.cpu().numpy()
                else:
                    data = ifft(data).abs().cpu().numpy()
                    predictions = ifft(predictions).cpu().numpy()
                targets = targets.cpu().numpy()
                    
                validator(predictions, targets)

                if batch_idx == plot_idx and self.epoch % 1 == 0:
                    if self.single_coil:
                        utilities.plot_singlecoil_example(
                            data, targets, predictions, self.epoch, self.save_folder, self.image_domain
                        )
                    else:
                        utilities.plot_multicoil_example(
                            data, targets, predictions, self.epoch, self.save_folder
                        )

        total_val_loss = val_loss / len(self.val_loader)

        logger.info(f"Validation metrics: {validator['all']}")

        if self.wandb:
            log_data = {
                "Validation Loss": total_val_loss,
                "Train Loss": self.total_loss,
                "MSE": validator["mse"],
                "NMSE": validator["nmse"],
                "PSNR": validator["psnr"],
                "SSIM": validator["ssim"],
                "LR": self.optimizer.param_groups[0]["lr"],
            }
            wandb.log(log_data, step=self.epoch)

        stop_metric = total_val_loss
        self.early_stopping(stop_metric)
        if self.early_stopping.early_stop:
            logger.info("Early stopping")

        self.model.train()

    @utilities.timer
    def __call__(self) -> None:
        """Perfoms all necessary training steps by initiating the epoch loop
        and saves the trained model at the end.

        Args:
            config (dict): Dictionary containing predefined settings used by several external functions.
        """
        self.save_folder = Path(self.save_base_folder)
        logger.info(f"Save folder: {str(self.save_folder)}")
        # Create save folder if it does not exist
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)

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

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[100, 200], gamma=0.5
        )

        # Get data loaders
        self.train_loader, self.val_loader, self.data_length = get_loaders(
            augmentation=self.augmentation,
            transform=self.transformation,
            train_path=self.train_path,
            val_path=self.val_path,
            save_path=self.save_folder,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            log=logger,
            single_coil=self.single_coil,
            image_domain=self.image_domain
        )

        # Log save folder information
        logger.info(f"Save folder: {str(self.save_folder)}")

        self.early_stopping = EarlyStopping(
            patience=30, verbose=True, monitor="val-loss", op_type="min", logger=logger
        )

        # Create save folder if it does not exist
        if os.path.exists(self.save_folder) is False:
            os.mkdir(self.save_folder)

        # Copy config file to save folder
        shutil.copyfile(args.config, Path(self.save_folder, args.config.name))

        # Start epoch loop
        for self.epoch in range(self.epochs):
            logger.info(f"Now training epoch {self.epoch}!")

            # Call first_epoch or train_fn based on current epoch
            if self.epoch == 0:
                Trainer.first_epoch(self)
            else:
                Trainer.train(self)
            checkpoint = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            logger.info(f"Train-loss: {self.total_loss}")

            # Validate the model
            Trainer.validation(self)

            if self.early_stopping.early_stop:
                break
            if self.early_stopping.save == True:
                torch.save(self.model, f"{self.save_folder}/best_checkpoint.pth")
                logger.info("Save checkpoint.")

            torch.save(self.model, f"{self.save_folder}/latest_checkpoint.pth")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parser.parse_args()
    args.config = Path(f"src/configs/{args.config}")

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(config["num_threads"])

    ikim_logger = IKIMLogger(
        level=args.log,
        log_dir="logs",
        comment=(
            f"train_{config['model']}_{config['loss']}_{config['activation']}_"
            f"{config['transformation']}_p{config['dropout']}_{config['features']}_{config['lr']}_"
            f"{config['optimizer']}_l{config['length']}_{config['comment']}"
        ),
    )
    logger = ikim_logger.create_logger()
    try:
        set_seed(1)
        training = Trainer(
            args=args,
            config=config,
        )
        logger.info(training.__repr__())
        training()
    except Exception as e:
        logger.exception(e)
