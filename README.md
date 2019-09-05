# RL ASC

This repo contains codes that implement paper _**State Representation Learning for Effective Deep Reinforcement Learning**_
in AAAI 2020, which is based on a fork of OpenAI Baselines. The three new algorithms `deepq_asc`, `a2c_asc` and
`ppo_asc` are modified from original `deepq`, `a2c` and `ppo2` respectively.

## Usage

For installation of some prerequisites, see [the original README](README_baselines.md).

To run three new algorithms, you can just run entries whose names are `run_*_asc.py` in the root directory.
For example:

```shell script
# DQN ASC
python run_deepq_asc.py --env=PongNoFrameskip-v4 --num_steps=1e6 --lambda=0.1 --margin=0.3 --alpha=1 --num_repeat_times=5

# PPO ASC
python run_ppo_asc.py --env PongNoFrameskip-v4 --num_steps=1e7 --lambda=0.1 --margin=0.3 --alpha=1 --num_repeat_times=5

# A2C ASC
python run_a2c_asc.py --env PongNoFrameskip-v4 --num_steps=2e7 --lambda=0.1 --margin=0.3 --alpha=1 --num_repeat_times=5
```
