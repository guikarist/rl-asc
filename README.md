# RL ASC

This repo contains codes that implement Paper **State Representation Learning for Effective Deep Reinforcement Learning**
in AAAI 2020, which is based on a fork of OpenAI Baselines. The three new algorithms `deepq_asc`, `a2c_asc` and
`ppo_asc` are modified from original `deepq`, `a2c` and `ppo2` respectively. Only some settings of the three original
algorithms are modified. 

## Usage

For installation of some prerequisites, see [the original README](README_baselines.md).

To run three new algorithms, you can just run entries whose names are `run_*_asc.py` in the root directory.
For example:

```shell script
# DQN ASC
python run_deepq_asc.py --env=PongNoFrameskip-v4 --num_steps=1e6 --lambda=0.1 --margin=0.3 --i=1 --network=modified_conv_only_v3 --num_epochs=5

# PPO ASC
python run_ppo_asc.py --env PongNoFrameskip-v4 --num_steps=1e6 --lambda=0.1 --margin=0.3 --i=1 --network=modified_mlp_v2 --num_epochs=5

# A2C ASC
python run_a2c_asc.py --env PongNoFrameskip-v4 --num_steps=2e7 --lambda=0.1 --margin=0.3 --i=1 --network=modified_mlp_v2 --num_epochs=5
```
