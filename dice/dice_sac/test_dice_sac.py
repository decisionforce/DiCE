import shutil
import tempfile

import gym
import numpy as np
import pytest

from dice.dice_sac import DiCESACTrainer, DiCESACPolicy
from dice.utils import get_marl_env_config, MultiAgentEnvWrapper, train, \
    initialize_ray


@pytest.fixture()
def dice_sac_policy():
    env_name = "BipedalWalker-v3"
    env = gym.make(env_name)
    policy = DiCESACPolicy(
        env.observation_space, env.action_space, {"env": env_name}
    )
    return env, policy


def test_policy(dice_sac_policy):
    env, policy = dice_sac_policy

    act, _, info = policy.compute_actions(np.random.random([400, 24]))
    assert act.shape == (400, 4)
    assert info["action_prob"].shape[0] == 400
    assert np.all(info["action_prob"] >= 0.0)
    assert info["action_logp"].shape[0] == 400
    assert info["behaviour_logits"].shape == (
        400, env.action_space.shape[0] * 2
    )

    policy._lazy_initialize({"test_my_self": policy}, None)


@pytest.fixture()
def dice_sac_trainer():
    initialize_ray(test_mode=True, local_mode=True)
    env_name = "BipedalWalker-v3"
    num_agents = 3
    env = gym.make(env_name)
    trainer = DiCESACTrainer(
        get_marl_env_config(env_name, num_agents, normalize_actions=False),
        env=MultiAgentEnvWrapper
    )
    return env, trainer


def test_trainer(dice_sac_trainer):
    env, trainer = dice_sac_trainer
    train_result = trainer.train()


def regression_test(local_mode=False):
    num_agents = 3
    local_dir = tempfile.mkdtemp()
    initialize_ray(test_mode=True, local_mode=local_mode)
    train(
        DiCESACTrainer, {
            "gamma": 0.95,
            "target_network_update_freq": 32,
            "tau": 1.0,
            "train_batch_size": 200,
            "rollout_fragment_length": 50,
            "optimization": {
                "actor_learning_rate": 0.005,
                "critic_learning_rate": 0.005,
                "entropy_learning_rate": 0.0001
            },
            **get_marl_env_config(
                "CartPole-v0", num_agents, normalize_actions=False
            )
        }, {"episode_reward_mean": 150 * num_agents},
        exp_name="DELETEME",
        local_dir=local_dir,
        test_mode=True
    )
    shutil.rmtree(local_dir, ignore_errors=True)


def regression_test2(local_mode=False):
    from ray import tune
    num_agents = 3
    local_dir = tempfile.mkdtemp()
    initialize_ray(test_mode=True, local_mode=local_mode)
    return train(
        DiCESACTrainer,
        {
            # "soft_horizon": True,
            "clip_actions": False,
            "normalize_actions": False,  # <<== Handle in MARL env

            # Evaluation
            "evaluation_interval": 1,
            "metrics_smoothing_episodes": 5,
            "evaluation_config": {
                "explore": False,
            },

            # "no_done_at_end": True,
            "train_batch_size": 256,
            "rollout_fragment_length": 1,
            "target_network_update_freq": 1,
            "timesteps_per_iteration": 100,
            # "learning_starts": 1500,
            "learning_starts": 500,
            "diversity_twin_q": tune.grid_search([True, False]),
            **get_marl_env_config(
                # "Pendulum-v0", num_agents, normalize_actions=True
                "Pendulum-v0",
                num_agents,
                normalize_actions=True
            )
        },
        {"timesteps_total": 10000},
        exp_name="DELETEME-dicesac",
        local_dir=local_dir,
        test_mode=True
    )


def regression_test_sac(local_mode=False):
    local_dir = tempfile.mkdtemp()
    initialize_ray(test_mode=True, local_mode=local_mode)
    train(
        "SAC",
        {
            "train_batch_size": 256,
            "rollout_fragment_length": 1,
            "target_network_update_freq": 1,
            "timesteps_per_iteration": 100,
            # "learning_starts": 1500,
            "learning_starts": 500,
            "env": "Pendulum-v0",
        },
        {
            # "episode_reward_mean": -300 * num_agents,
            "timesteps_total": 3000
        },
        exp_name="DELETEME",
        local_dir=local_dir,
        test_mode=True
    )
    shutil.rmtree(local_dir, ignore_errors=True)


if __name__ == "__main__":
    # pytest.main(["-v"])
    # regression_test(local_mode=False)
    anal = regression_test2(local_mode=False)
    # regression_test_sac(local_mode=False)
