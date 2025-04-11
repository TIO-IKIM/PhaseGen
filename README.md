[![Python 3.12.4](https://img.shields.io/badge/python-3.12.4-blue.svg)](https://www.python.org/downloads/release/python-3124/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>
[![arXiv](https://img.shields.io/badge/arXiv-2501.05269-b31b1b.svg)](https://arxiv.org/abs/2504.07560)
# PhaseGen

<div align="center">

[Usage](#usage) • [Roadmap](#roadmap) • [Citation](#citation)

</div>

This repository consists of multiple work packages. The main package is `phase_gen`, which includes the complex-valued diffusion model for raw data generation.
Each package has its own documentation. You are able to run each package as a standalone.

![Figure 1: Project Overview](/Figures/overview.png)

## Usage

For more detailed instructions, please find the documentation in each workpackage.

### Installation via Git

1. Clone repository:

       git clone https://github.com/TIO-IKIM/PhaseGen.git

2. Create a conda environment with Python version 3.12.4 and install the necessary dependencies:

       cd PhaseGen
       conda create -n my_env python=3.12.4
      ```  
      conda activate my_env
      pip install -r requirements.txt
      ```

3. Torch is already included in the requirements.txt file. Nevertheless, you should install it separately to ensure that the correct version is installed. You can find the appropriate command for your system at [PyTorch](https://pytorch.org/get-started/locally/).

## Roadmap

Model weights of PhaseGen will be uploaded soon. Until then the weights are available upon request. 

## Citation

```
@misc{rempe2025phasegendiffusionbasedapproachcomplexvalued,
      title={PhaseGen: A Diffusion-Based Approach for Complex-Valued MRI Data Generation}, 
      author={Moritz Rempe and Fabian Hörst and Helmut Becker and Marco Schlimbach and Lukas Rotkopf and Kevin Kröninger and Jens Kleesiek},
      year={2025},
      eprint={2504.07560},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2504.07560},
}
```
