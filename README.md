# Diversity-regularized Collaborative Exploration

This repository contains codes for paper "Diversity-regularized Collaborative Exploration". 

You need to install the following packages:

1. ray==0.8.0
2. gym
3. tensorflow==1.15.2
4. numpy
5. mujoco-py

To test whether you can run the codes without bugs, run:

```bash
python test_dice.py
python train_dice.py --test
```

To train DiCE in different environments, you simply need to run:

```bash
python train_dice.py --exp-name dice-change-env --stop 50000000 --env-name Walker2d-v3 --num-gpus 2
python train_dice.py --exp-name dice-change-env --stop 50000000 --env-name Hopper-v3 --num-gpus 2
python train_dice.py --exp-name dice-change-env --stop 20000000 --env-name Humanoid-v3 --num-gpus 2
python train_dice.py --exp-name dice-change-env --stop 5000000 --env-name Ant-v3 --num-gpus 2
python train_dice.py --exp-name dice-change-env --stop 5000000 --env-name HalfCheetah-v3 --num-gpus 2
```

To train DiCE varying the number of agents, run:

```bash
python train_dice.py --exp-name dice-change-num --stop 50000000 --num-agents 1 --num-gpus 2
python train_dice.py --exp-name dice-change-num --stop 50000000 --num-agents 3 --num-gpus 2
python train_dice.py --exp-name dice-change-num --stop 50000000 --num-agents 7 --num-gpus 2
python train_dice.py --exp-name dice-change-num --stop 50000000 --num-agents 10 --num-gpus 2
```
