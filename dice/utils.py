"""
This files defines some constants and provides some useful utilities.
"""
import argparse
import copy
import logging
import os
import pickle

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_DEFAULT
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.normalize_actions import NormalizeActionWrapper
from ray.rllib.utils import try_import_tf
from ray.tune.utils.util import merge_dicts

tf = try_import_tf()
logger = logging.getLogger(__name__)


def register_minigrid():
    if "MiniGrid-Empty-16x16-v0" in [s.id for s in gym.envs.registry.all()]:
        return
    try:
        import gym_minigrid.envs
    except ImportError as e:
        print("Failed to import minigrid environment!")
    else:
        assert "MiniGrid-Empty-16x16-v0" in [
            s.id for s in gym.envs.registry.all()
        ]
        print("Successfully imported minigrid environments!")


def get_marl_env_config(env_name, num_agents, normalize_actions=False):
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "num_agents": num_agents,
            "normalize_actions": normalize_actions
        },
        "multiagent": {}
    }
    return config


class MultiAgentEnvWrapper(MultiAgentEnv, gym.Env):
    """This is a brief wrapper to create a mock multi-agent environment"""

    def __init__(self, env_config):
        assert "num_agents" in env_config
        assert "env_name" in env_config
        num_agents = env_config['num_agents']
        agent_ids = ["agent{}".format(i) for i in range(num_agents)]
        self._render_policy = env_config.get('render_policy')
        self.num_agents = num_agents
        self.agent_ids = agent_ids
        self.env_name = env_config['env_name']
        self.env_maker = get_env_maker(
            env_config['env_name'], require_render=bool(self._render_policy)
        )

        if env_config.get("normalize_action", False):
            self.env_maker = lambda: NormalizeActionWrapper(self.env_maker())

        self.envs = {}
        if not isinstance(agent_ids, list):
            agent_ids = [agent_ids]
        for aid in agent_ids:
            if aid not in self.envs:
                self.envs[aid] = self.env_maker()
        self.dones = set()
        tmp_env = next(iter(self.envs.values()))
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        self.reward_range = tmp_env.reward_range
        self.metadata = tmp_env.metadata
        self.spec = tmp_env.spec

    def reset(self):
        self.dones = set()
        return {aid: env.reset() for aid, env in self.envs.items()}

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        for aid, act in action_dict.items():
            o, r, d, i = self.envs[aid].step(act)
            if d:
                if d in self.dones:
                    print(
                        "WARNING! current {} is already in"
                        "self.dones: {}. Given reward {}.".format(
                            aid, self.dones, r
                        )
                    )
                self.dones.add(aid)
            obs[aid], rewards[aid], dones[aid], infos[aid] = o, r, d, i
        dones["__all__"] = len(self.dones) == len(self.agent_ids)
        return obs, rewards, dones, infos

    def seed(self, s):
        for env_id, env in enumerate(self.envs.values()):
            env.seed(s + env_id * 10)

    def render(self, *args, **kwargs):
        assert self._render_policy
        assert self._render_policy in self.envs, (
            self._render_policy, self.envs.keys()
        )
        return self.envs[self._render_policy].render(*args, **kwargs)

    def __repr__(self):
        return "MultiAgentEnvWrapper({})".format(self.env_name)


class DiCECallbacks(DefaultCallbacks):
    def on_postprocess_trajectory(self, *args, **kwargs):
        on_postprocess_trajectory(*args, **kwargs)

    def on_train_result(self, *args, **kwargs):
        on_train_result(*args, **kwargs)


def on_train_result(trainer, result, **_):
    if result['custom_metrics']:
        item_list = set()
        for key, val in result['custom_metrics'].items():
            policy_id, item_full_name = key.split('-')
            # item_full_name: action_kl_mean

            item_name = \
                "".join([s + "_" for s in item_full_name.split("_")[:-1]])[:-1]
            # item_name: action_kl
            item_list.add(item_name)

            if item_name not in result:
                result[item_name] = {}

            if policy_id not in result[item_name]:
                result[item_name][policy_id] = {}

            result[item_name][policy_id][item_full_name] = val

        for item_name in item_list:
            result[item_name]['overall_mean'] = np.mean(
                [
                    a[item_name + "_mean"] for a in result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
            result[item_name]['overall_min'] = np.min(
                [
                    a[item_name + "_min"] for a in result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
            result[item_name]['overall_max'] = np.max(
                [
                    a[item_name + "_max"] for a in result[item_name].values()
                    if isinstance(a, dict)
                ]
            )
        result['custom_metrics'].clear()


def on_postprocess_trajectory(
        worker, episode, agent_id, policy_id, policies, postprocessed_batch,
        original_batches, **_
):
    """Originally, the sampled steps is accumulated by the count of the
    batch
    return by the postprocess function in dice_postprocess.py. Since we
    merge the batches of other polices into the batch of one policy,
    the count of the batch should changed to the number of sampled of all
    policies. However it's not changed. In this function, we correct the
    count of the MultiAgentBatch in order to make a fair comparison between
    DiCE and baseline.
    """
    if episode._policies[agent_id].config[ONLY_TNB]:
        # ONLY_TNB modes mean we using purely DR, without CE. So in that
        # case
        # we don't need to correct the count since no other batches are
        # merged.
        return

    if agent_id != next(iter(original_batches.keys())):
        return

    increment_count = int(
        np.mean([b.count for _, b in original_batches.values()])
    )
    current_count = episode.batch_builder.count
    corrected_count = current_count - increment_count + \
                      postprocessed_batch.count
    episode.batch_builder.count = corrected_count


USE_BISECTOR = "use_bisector"  # If false, the the DR is disabled.
USE_DIVERSITY_VALUE_NETWORK = "use_diversity_value_network"
DELAY_UPDATE = "delay_update"
TWO_SIDE_CLIP_LOSS = "two_side_clip_loss"
ONLY_TNB = "only_tnb"  # If true, then the CE is disabled.

CLIP_DIVERSITY_GRADIENT = "clip_diversity_gradient"
DIVERSITY_REWARD_TYPE = "diversity_reward_type"
DIVERSITY_REWARDS = "diversity_rewards"
DIVERSITY_VALUES = "diversity_values"
DIVERSITY_ADVANTAGES = "diversity_advantages"
DIVERSITY_VALUE_TARGETS = "diversity_value_targets"
PURE_OFF_POLICY = "pure_off_policy"
NORMALIZE_ADVANTAGE = "normalize_advantage"

dice_default_config = merge_dicts(
    PPO_DEFAULT,
    {
        USE_BISECTOR: True,
        USE_DIVERSITY_VALUE_NETWORK: False,
        DELAY_UPDATE: True,
        TWO_SIDE_CLIP_LOSS: True,
        ONLY_TNB: False,
        NORMALIZE_ADVANTAGE: False,
        CLIP_DIVERSITY_GRADIENT: True,
        DIVERSITY_REWARD_TYPE: "mse",
        PURE_OFF_POLICY: False,
        "tau": 5e-3,
        "vf_ratio_clip_param": 0.05,  # Not pass to PPOLossTwoSideClip
        "callbacks": DiCECallbacks,
        "grad_clip": 10.0
    }
)


def get_kl_divergence(source, target, mean=True):
    assert source.ndim == 2
    assert target.ndim == 2

    source_mean, source_log_std = np.split(source, 2, axis=1)
    target_mean, target_log_std = np.split(target, 2, axis=1)

    kl_divergence = np.sum(
        target_log_std - source_log_std + (
                np.square(np.exp(source_log_std)) +
                np.square(source_mean - target_mean)
        ) / (2.0 * np.square(np.exp(target_log_std)) + 1e-10) - 0.5,
        axis=1
    )
    kl_divergence = np.clip(kl_divergence, 1e-12, 1e38)  # to avoid inf
    if mean:
        averaged_kl_divergence = np.mean(kl_divergence)
        return averaged_kl_divergence
    else:
        return kl_divergence


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def _get_env_name(config):
    if isinstance(config["env"], str):
        env_name = config["env"]
    elif isinstance(config["env"], dict):
        assert "grid_search" in config["env"]
        assert isinstance(config["env"]["grid_search"], list)
        assert len(config["env"]) == 1
        env_name = config["env"]["grid_search"]
    else:
        assert config["env"] is MultiAgentEnvWrapper
        env_name = config["env_config"]["env_name"]
        if isinstance(env_name, dict):
            assert "grid_search" in env_name
            assert isinstance(env_name["grid_search"], list)
            assert len(env_name) == 1
            env_name = env_name["grid_search"]
    assert isinstance(env_name, str) or isinstance(env_name, list)
    return env_name


def train(
        trainer,
        config,
        stop,
        exp_name,
        num_seeds=1,
        num_gpus=0,
        test_mode=False,
        suffix="",
        checkpoint_freq=10,
        keep_checkpoints_num=None,
        start_seed=0,
        **kwargs
):
    # initialize ray
    if not os.environ.get("redis_password"):
        initialize_ray(
            test_mode=test_mode, local_mode=False, num_gpus=num_gpus
        )
    else:
        password = os.environ.get("redis_password")
        assert os.environ.get("ip_head")
        print(
            "We detect redis_password ({}) exists in environment! So "
            "we will start a ray cluster!".format(password)
        )
        if num_gpus:
            print(
                "We are in cluster mode! So GPU specification is disable and"
                " should be done when submitting task to cluster! You are "
                "requiring {} GPU for each machine!".format(num_gpus)
            )
        initialize_ray(
            address=os.environ["ip_head"],
            test_mode=test_mode,
            redis_password=password
        )

    # prepare config
    used_config = {
        "seed": tune.grid_search(
            [i * 100 + start_seed for i in range(num_seeds)]
        ),
        "log_level": "DEBUG" if test_mode else "INFO"
    }
    if config:
        used_config.update(config)
    config = copy.deepcopy(used_config)

    env_name = _get_env_name(config)

    trainer_name = trainer if isinstance(trainer, str) else trainer._name

    assert isinstance(env_name, str) or isinstance(env_name, list)

    if not isinstance(stop, dict):
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None and not test_mode:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq if not test_mode else None,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        max_failures=20 if not test_mode else 1,
        reuse_actors=False,
        **kwargs
    )

    # save training progress as insurance
    pkl_path = "{}-{}-{}{}.pkl".format(
        exp_name, trainer_name, env_name, "" if not suffix else "-" + suffix
    )
    with open(pkl_path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(pkl_path))
    return analysis


def get_env_maker(name, require_render=False):
    if callable(name):
        return lambda: name()
    if isinstance(name, str) and name.startswith("MiniGrid"):
        print(
            "Return the mini grid environment {} with MiniGridWrapper("
            "FlatObsWrapper)!".format(name)
        )
        return lambda: MiniGridWrapper(gym.make(name))
    else:
        assert name in [s.id for s in gym.envs.registry.all()], \
            "name of env {} not in {}".format(
                name, [s.id for s in gym.envs.registry.all()])
        return lambda: gym.make(name)


class MiniGridWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MiniGridWrapper, self).__init__(env)
        space = self.env.observation_space.spaces["image"]
        length = np.prod(space.shape)
        shape = [
            length,
        ]
        self.observation_space = gym.spaces.Box(
            low=space.low.reshape(-1)[0],
            high=space.high.reshape(-1)[0],
            shape=shape
        )

    def observation(self, obs):
        return obs["image"].ravel()


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--num-cpus-for-driver", type=float, default=1.0)
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v3")
    parser.add_argument("--test", action="store_true")
    # parser.add_argument("--redis-password", type=str, default="")
    return parser
