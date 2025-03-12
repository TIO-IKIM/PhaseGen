[![Python 3.11.2](https://img.shields.io/badge/python-3.11.2-blue.svg)](https://www.python.org/downloads/release/python-3106/) <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a> [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Black v23.9.0](https://img.shields.io/badge/black-23.9.0-orange)](https://black.readthedocs.io/en/stable/getting_started.html) [![DOI](https://img.shields.io/badge/DOI-j.cmpb.2023.107912-blue)](https://www.sciencedirect.com/science/article/pii/S0169260723005783)

<div align="center">

[Project content & structure](#Project-content-structure) • [Getting started](#getting-started) • [Usage](#usage) • [Citation](#Citation)

</div>

This repository contains the implementation of [_k-Strip_](https://www.sciencedirect.com/science/article/pii/S0169260723005783?via%3Dihub), a complex valued convolutional neural network (CVNN) based method for automated skull stripping directly in the k-space, the raw data domain of MRI scans.

This work is part of the **k-Radiomics** project. For more information, please visit our project-website https://k-radiomics.ikim.nrw.

## Project content & structure

![ResUNet](Figures/cResUNet_light.png#gh-light-mode-only)

This repository provides the architecture of a complex valued neural network in the style of a residual UNet.

We also provide multiple complex valued blocks and layer, as well as loss and activation functions, which can all be found in the folder ```src/models/layer```.


**Project folder structure**:
```bash  
├── logs                  # Location of logs created during training / testing
├── output                # Location of output created during training / testing
│   ├── training  
│   └── testing  
└── src  
    ├── configs           # Config files for training / tesing  
    ├── models            # Location of the network architecture
    │   └── layer         # Location of all building blocks needed for the network
    └── utils             # Utility functions, including dataloader
```

**Implemented layers & functions**

Toggle the lists below to see an overview of implemented layer, blocks and activation functions. For more details, have a look at the corresponding python files in the ```src/models/layer``` folder.

<details>
    <summary markdown="span">Complex layer & blocks</summary>

- cBatchNorm
- cConvolution
- cDoubleConvolution
- cResidualBlock
- cTransposedConvolution
- Spectral Pooling
- cDropout
- cUpsampling
</details>

<details>
    <summary markdown="span">Complex activation functions</summary>

- cReLU
- clReLU
- cELU
- cPReLU
- cSELU
- cTanh
- cSigmoid
- PhaseAmplitudeReLU
- PhaseReLU
- Cardioid
- AmplitudeReLU
- cLogReLU
</details>

## Getting started
1. Clone repository:

       git clone https://github.com/TIO-IKIM/k-Strip.git
   >Note: The training & testing scripts expect the repository to be saved directly in the home directory (```Path.home()```). If this is not the case, you just need to insert the correct path in the scripts.
2. Create a conda environment with Python version 3.11.2 and install the necessary dependencies:

       conda create --name k-strip python=3.11.2 --file requirements.txt
    In case of installation issues with conda, use pip install -r requirements.txt to install the dependecies.
    >Note: The PyTorch version used needs CuDA 11.7 installed. You can change the PyTorch version in the requirements file.
3. Activate your new environment:

       conda activate k-strip

4. Insert all necesarry paths to your training / testing data in the config files in ```src/configs``` and you should be all set to start training your own CVNN!
   > Note: The training config is already filled with the settings used for our publication, but you can change all parameters as you wish ... until you break it.


## Usage

The repository includes an exemple for preparing the [CC-359](https://www.ccdataset.com/download) for training the network by transforming it into fourier transformed 2D .pt slices. You can find the script at ```src/utils/prepare_data.py```.
Just download the dataset and adjust the paths in the script accordingly.  

**Training CLI**

```
usage: Training [-h] [--e E] [--log LOG] [--tqdm] [--gpu GPU] [--config CONFIG] [--load LOAD]
                [--load_checkpoint LOAD_CHECKPOINT]

Train a CVNN.

options:
  -h, --help            show this help message and exit
  --e E                 Number of epochs for training
  --log LOG             Define debug level. Defaults to INFO.
  --tqdm                If set, do not log training loss via tqdm.
  --gpu GPU             GPU used for training.
  --config CONFIG       Path to configuration file
  --load LOAD           Load model by giving path to saved model. Defaults to None.
  --load_checkpoint LOAD_CHECKPOINT
                        Load checkpoint by giving path to checkpoint. Defaults to None.
```
Checkpoints of the model and examples of the validation results are saved every 10 epochs. The frequency can be changed in the script in ```if self.epoch % 10 == 0```.

**Testing CLI**

```
usage: Tester [-h] [--config CONFIG] [--e] [--perc PERC] [--num_zero NUM_ZERO] [--eval_num EVAL_NUM]
              [--id ID] [--c C] [--device DEVICE]

Test a trained network.

options:
  -h, --help           show this help message and exit
  --config CONFIG      Path to configuration file
  --e                  Start inference. If selected, start inference. Defaults to True.
  --perc PERC          If selected choose single percentage value for binary masking in evaluation. If
                       None, iterate through range of values in case of evaluation. Defaults to None
  --num_zero NUM_ZERO  Choose one number of minmal amount non-zero pixels in segmentation masks to be taken
                       into account in evaluation. Use None to iterate through range of values in case of
                       evaluation. Defaults to None
  --eval_num EVAL_NUM  Define if not all samples are supposed to be used for evaluation. Defaults to None.
  --c C                Checkpoint number to be loaded. For this the path to the checkpoints need to be
                       defined in the config file. Defaults to None.
  --device DEVICE      Cuda device to run test on.
```

## Datastructure

The network expects a certain datastructure which is shown below.
```bash  
├── .../train  
│       ├── orig
│       └── label  
├── .../val  
│       ├── orig
│       └── label  
└── .../test  
        ├── orig
        └── label  
```

## Citation

If you use our code in your work, please cite us with the following BibTeX entry.
```latex
@article{rempe2023k,
  title={k-strip: A novel segmentation algorithm in k-space for the application of skull stripping},
  author={Rempe, Moritz and Mentzel, Florian and Pomykala, Kelsey L and Haubold, Johannes and Nensa, Felix and Kr{\"o}ninger, Kevin and Egger, Jan and Kleesiek, Jens},
  journal={Computer Methods and Programs in Biomedicine},
  pages={107912},
  year={2023},
  publisher={Elsevier}
}
```
