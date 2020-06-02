# RL ASC

This repo contains codes that implement RL algorithms with ASC loss, which are proposed in the paper _**State Representation Learning for Effective Deep Reinforcement Learning**_
in ICME 2020. And the codes are based on a fork of OpenAI Baselines. The three new algorithms `deepq_asc`, `a2c_asc` and
`ppo_asc` are modified from the original `deepq`, `a2c` and `ppo2` respectively.

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```
    
## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other (For using **Conda**, see the next paragraph). You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/) 

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```

### Using Conda (Recommended)
- This line will do all things for you including installing a CUDA package.
    ```bash
    conda env create -f env.yml
    ```

### Using Pip
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

For more details of installation , see [the original README](https://github.com/openai/baselines/blob/master/README.md).

## Usage

To run three new algorithms, you can just run entries whose names are `run_*_asc.py` in the root directory.

For example, each of following commands will run both original algorithms and modified ones with ASC loss, each setting of
which is repeated `num_repeat_times` times:

```shell script
# DDQN (which runs 5 original DDQN experiments and 5 modified DDQN experiments with ASC loss)
python run_deepq_asc.py --env=PongNoFrameskip-v4 --num_steps=1e6 --lambda=0.1 --margin=0.3 --alpha=1 --num_repeat_times=5 --double_q

# A2C
python run_a2c_asc.py --env=PongNoFrameskip-v4 --num_steps=2e7 --lambda=0.1 --margin=0.3 --alpha=1 --num_repeat_times=5

# PPO
python run_ppo_asc.py --env=PongNoFrameskip-v4 --num_steps=1e7 --lambda=0.1 --margin=0.3 --alpha=1 --num_repeat_times=5
```
