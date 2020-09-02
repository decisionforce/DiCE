# Diversity-regularized Collaborative Exploration

[[Blog]](https://decisionforce.github.io/DiCE/) [[Paper]](https://arxiv.org/pdf/2006.07781.pdf)

This repository contains code for paper *"Non-local Policy Optimization via Diversity-regularized Collaborative Exploration"*. 


## Abstract

Conventional Reinforcement Learning (RL) algorithms usually have one single agent learning to solve the task independently. As a result, the agent can only explore a limited part of the state-action space while the learned behavior is highly correlated to the agent's previous experience, which makes the training prone to a local minimum. We propose a novel non-local policy optimization framework called Diversity-regularized Collaborative Exploration (DiCE). DiCE utilizes a group of heterogeneous agents to explore the environment simultaneously and share the collected experiences (implemented in `dice_ppo/dice_postprocess.py` and `dice_sac/dice_sac.py`). A regularization mechanism is further designed to maintain the diversity of the team and modulate the exploration (implemented in `dice_ppo/dice_loss.py` and `dice_sac/dice_sac_gradient.py`). We implement the framework in both on-policy (`dice_ppo/`) and off-policy (`dice_sac/`) settings and the experimental results show that DiCE can achieve substantial improvements over the baselines in the MuJoCo locomotion tasks.

## Quickstart

We use the following packages: `ray==0.8.5`, `gym==0.17.2`, `tensorflow==2.1.0`, `numpy==1.16.0`, `mujoco-py==1.50.1.68`.

For a quick start, you can run the following scripts to setup the environment:

```bash
git clone https://github.com/decisionforce/DiCE.git
cd DiCE

conda env create -f environment.yml
conda activate dice
pip install gym[mujoco]
pip install gym[box2d]
pip install ray[rllib]==0.8.5
pip install ray[tune]==0.8.5
pip install -e .
```

（You may encounter troubles when installing mujoco, please make sure that at the end you have successfully install mujoco 150 version.)

To test whether you can run the codes without bugs, run:

```bash
cd dice/dice_ppo
python test_dice.py
python train_dice.py --test
```

or 

```bash
python dice_sac/test_dice_sac.py
python dice_sac/train_dice.py --test --num-gpus 0 --exp-name TEST
```


To train DiCE in different environments, you simply need to run:

```bash
# Train DiCE-PPO in 5 environments
cd dice/dice_ppo
python train_dice.py --exp-name dice-ppo --stop 5e7 --env-name Walker2d-v3 --num-gpus 2
python train_dice.py --exp-name dice-ppo --stop 5e7 --env-name Hopper-v3 --num-gpus 2
python train_dice.py --exp-name dice-ppo --stop 2e7 --env-name Humanoid-v3 --num-gpus 2
python train_dice.py --exp-name dice-ppo --stop 5e6 --env-name Ant-v3 --num-gpus 2
python train_dice.py --exp-name dice-ppo --stop 5e6 --env-name HalfCheetah-v3 --num-gpus 2

# Train DiCE-SAC in all 5 environments
cd dice/dice_sac
python train_dice_sac.py --exp-name dice-sac
```

To train DiCE-PPO varying the number of agents, run the following script with different `X=1,3,...`:

```bash
cd dice_ppo
python train_dice.py --exp-name dice-ppo-num --stop 5e7 --num-agents X --num-gpus 2
```


## Code Structure

* `dice_ppo/dice_ppo.py` - Implements the on-policy DiCE trainer and also the function to initialize and update the target networks in each policies.
* `dice_ppo/dice_policy.py` - Implements the on-policy DiCE policy. Builds the target network; provides the API for computing diversity and update target network.
* `dice_ppo/dice_postprocess` - Implements the Collaborative Exploration module in on-policy setting. Fuses the batches collected by different agents, and computes the diversity of each agent against others.
* `dice_ppo/dice_loss.py` - Implements the Diversity Regularization module in on-policy setting. Fuses the task gradients and the diversity gradients into the final gradients.
* `dice_ppo/dice_model.py` - Modifies the original Fully Connected Network for PPO agent. Adds the diversity reward and diversity advantage to the training batch.
* `dice_ppo/train_dice.py` - Trains DiCE. 
* `dice_sac/dice_sac.py` - Implements the Collaborative Exploration in off-policy setting.
* `dice_sac/dice_sac_gradient.py` - Implements the Diversity Regularization in off-policy setting.
* `dice_sac/dice_sac_model.py` - Implements the Diversity critic in off-policy setting.
* `dice_sac/dice_sac_policy.py` - Some helper functions for statistics.
* `dice_sac/dice_sac_config.py` - The default hyper-parameters for DiCE-SAC.
* `dice_sac/train_dice_sac.py` - Train all 5 environments together.
