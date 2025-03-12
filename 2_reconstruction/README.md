## Package content & structure

This package includes the code for the reconstruction task. We use a complex valued neural network based on the k-Strip architecture. Additionally, we implemented a data consistency layer in the down-sampling path.

## Usage

If you already installed the necessary requirements in the skullstripping package, you should already be able to run the following code. Otherwise please install the requirements in the `requirements.txt` file.

The usage is very similar to the one of k-Strip.

**Training CLI**

```
usage: Training [-h] [--e E] [--log LOG] [--tqdm] [--gpu GPU] [--c]

Train a CVNN.

options:
  -h, --help            show this help message and exit
  --e E                 Number of epochs for training
  --log LOG             Define debug level. Defaults to INFO.
  --tqdm                If set, do not log training loss via tqdm.
  --gpu GPU             GPU used for training.
  --w                   Use Weights & Biases for logging.
  --config CONFIG       Path to configuration file
  --c                   Load checkpoint by giving path to checkpoint.
```

**Testing CLI**
```
usage: test_model.py [-h] [-m MODEL_PATH] -i DATA_PATH

options:
  -h, --help            show this help message and exit
  -m                    Path to the model to be tested.
                        If None is selected, zerofilling will be tested.
  -i                    Path to the test data.
```
