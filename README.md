# Assignment 3 - Mobilenet-v2 Compression

This codebase is the implementation of a compression algorithm for Mobilenet-v2 model. 
This was developed as part of the CS6886 Systems for Deep Learning course

## Installation

Use the package manager pip to install foobar.

```pip install -r requirements.txt```

## File Structure
The file structure is inspired by the template codebase provided at: 
https://github.com/DextroLaev/CS6886-Sys_for_dl-Assignment3

### Overview of files
- _Helper Files_: 
  - ```utils.py```: Contains helper methods for model evaluation, logging, traversal, etc.
  - ```dataloader.py```: Inspired by the dataloader in the template codebase, it contains functions to import the CIFAR-10 Dataset for model training
  - ```quantize.py```: Contains all logic units required for 4, 8 and 16bit quantization of mobilenet-v2 a model
  - ```mobilenet.py```: Inspired by the vgg16.py file in the template codebase, this file contains the building blocks of mobilenet-v2 model, as well as code model class for mobilenet-v2 built from scratch. It also contains pruning functions for Mobilenet-v2.
  - ```main.py```: This contains the code to train the Mobilenet-v2 model from scratch
  - ```test.py```: Contains the code to evaluate original model, pruned model and (pruned+quantised) models
  This also updates each sweep to the wandb dashboard
  - ```transfer_train.py```: Helper file to further fine-tune the model post training (Used for testing, not required in actual sweeps)
- _Folders_:
  - ```data``: Contains the CIFAR-10 dataset
  - ```logs```: Contains logs which were generated during model training on my machine. This folder is included for training credinbility

## Usage

### Setup using Virtual Environment (Recommended)
- Create a virtual environment using ```python3 -m venv <venv-name>```
- Source the activate file using ```source ${venv_path}/bin/activate```
- Install the required dependencies 
  ```pip install -r requirements.txt```
- Login into wandB account from the terminal

### Training the model
- For model training, use the command:
  ```python main.py```

### Evaluation and Sweeps
We run a single sweep with parameters for controlling activation and weight bits for quantisation:
```python test.py --weight_quant_bits <weight_bits> --activation_quant_bits <activation_bits>```
If the arguments are not supplied, the default value taken is 8

_NOTE_: This file will upload each sweep to wandB, so completing the wandB login is important