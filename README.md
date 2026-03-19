# TransMASK: Masked State Representation through Learned Transformation

[![arXiv](https://img.shields.io/badge/arXiv-2508.01600-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2603.05670)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

![TransMASK](./docs/resources/intro.gif)

[**Installation**](#installation) | [**Prepare Dataset**](#downloading-the-dataset) | [**Train️‍**](#train) | [**Evaluate**](#evaluation) | [**Project Website**](https://collab.me.vt.edu/TransMASK/)

## Installation
### Clone this Repo
```
git clone https://github.com/VT-Collab/TransMASK.git
cd TransMASK
```

### Installing using Conda

You can install the vitual conda environment using the following command:
```
conda create -n transmask python=3.10
conda activate transmask
pip install -r requirements.txt 
```

### Installing panda-gym

The following command will install the panda-gym package in your virtual conda environment:
```
cd panda_gym
pip install -e .
cd ..
```

## Downloading the Dataset

Run the following command to download the panda-gym dataset for three tasks - Pick and Place, Push, and Rotate:
```
python download_datasets.py
```

This will create a ```datasets/``` directory in the main folder.
The details about the dataset structure can be found here: https://huggingface.co/datasets/SagarParekh/TransMASK

## Train
To train a Behavior Cloning (BC) policy on a dataset with privileged state information, run the following command:

```
python train.py --dataset_path [dataset_path] --device cuda --env_name [env_name] --policy [policy_type] --savename [save_name] --epochs 500 --num_distractors 4 --method [method]
```

To train a Behavior Cloning (BC) policy on a dataset with image observations, run the following command:

```
python train.py --dataset_path [dataset_path] --device cuda --env_name [env_name] --policy [policy_type] --savename [save_name] --epochs 500 --num_distractors 4 --method [method] --use_image_obs
```

The ```--env_name``` argument has the following options:
- PandaPickAndPlace-v3
- PandaPush-v3
- PandaFlipJoints-v3

The ```--policy``` argument has two options:
- mlp (to train a multi layer perceptron action head)
- diffusion (to train a diffusion policy)

For diffusion policy, you can also specify the length of input sequence as well as the length of predicted actions using the arguments ```--obs_horizon``` and ```--pred_horizon``` respectively.

The ```--method``` argument allows you to choose whether to train a vanilla BC, VAE BC, or TransMASK BC policy
- default
- vae
- our

The trained models are stored in ```./results/[task_name]/[save_name]```.

## Evaluation
To evaluate the trained policy in the simulated environment, run the following command:

```
python eval.py --num_evals 100 --device cuda --render_mode rgb_array --loadloc [path_to_trained_model]  --n_rollout_actions [n_rollout_actions]
```

The ```--n_rollout_actions``` argument allows you to specify the length of the predicted action sequence that will be executed.

The results are saved in the same directory as the trained models.
