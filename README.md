# SpineDiffusion



## Introduction
This repository contains the code for the SpineDiffusion project within the greater 4D-Spine project by the Laboratory for Movement Biomechanics at ETH Zürich. The 4D-Spine project aims to develop a non-invasive, non-ionizing platform for the assessment of spinal deformities in the context of idiopathic scoliosis. The SpineDiffusion project implemented here leverages probabilistic diffusion models to generate synthetic 3D back scans in a tunable manner. This means that they should provide a way for the user to define the desired spinal line shape and output an appropriate back scan based on it. The ultimate goal is to augment the 4D-Spine dataset with synthetic data to improve the performance of machine learning algorithms for the estimation of the spinal line in patients with scoliosis.

## Getting started
The code has been developed in Python 3.8.3. If you are using Windows I recommend using [pyenv-win](https://pyenv-win.github.io/pyenv-win/docs/installation.html) to manage your python versions. In addition I recommend using venv to create a virtual environment for this project and install all the dependencies. Alternatively, you can use [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.anaconda.com/miniconda/).

To get started with the project, clone this repository, create a virtual environment and install the dependencies. The following commands should get you started:
```bash
git clone "repo url"
cd SpineDiffusion
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
This repository basically wraps the [Diffusers](https://huggingface.co/docs/diffusers/index) library from HuggingFace with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/), a high-level wrapper of Pytorch that organizes the code and automates most common deep learning routines. Specifically, I decided to use Pytorch Lightning's [CLI tool](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) because it provides a powerful CLI that allows you to train, test, and predict with your model in a streamlined and reproducible way (even if documentation is often lacking). Thus, if something in the code is unclear I recommend checking the Pytorch Lightning documentation first.

Training a model is as simple as running the following command:
```bash
python main.py fit --config path_to_config --options ...
```
where `path_to_config` is the path to a configuration file that specifies the model, the data, the optimizer, the scheduler, the callbacks, etc. The `options` are optional arguments that can be used to override the configuration file. For example, if you want to change the batch size you can run:
```bash
python main.py fit --config path_to_config --data.batch_size 32
```
For examples of configuration files check the `configs` folder.

Predicting with a model is similarly simple:
```bash
python main.py predict --config path_to_config --options ...
```

A big advantage of Pytorch lightning is its modularity. The code is organized into three main parts:
- The data module: This module is responsible for loading the data and preprocessing it. It is implemented in the `spinediffusion/data` folder.
- The lightning module: This module is responsible for defining the model architecture and the forward pass. It is implemented in the `spinediffusion/models` folder. It takes as attributes the actual backbone model and noise scheduler from the Diffusers library.
- The trainer: This module is responsible for training the model and it is provided by Pytorch Lightning.

This means that you can easily swap out the data module, the lightning module, or the trainer module with your own implementation. For example, if you want to use a different dataset you can simply implement a new data module that loads your data and preprocesses it in the way you want. Similarly, if you want to use a different backbone model or noise scheduler you simply have to modify the config file, as the lightning module is already implemented to take these as arguments.

In the `notebooks` folder you can find some Jupyter notebooks that demonstrate how to use the code. These notebooks are useful for debugging and testing new ideas, as well as visualizing the results from different training runs.
