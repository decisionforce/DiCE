import logging

import gym
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.util import merge_dicts

logger = logging.getLogger(__name__)


class MultiAgentEnvWrapper(MultiAgentEnv):
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
        self.env_maker = lambda: gym.make(env_config['env_name'])
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

    def reset(self):
        self.dones = set()
        return {aid: env.reset() for aid, env in self.envs.items()}

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        for aid, act in action_dict.items():
            act = np.nan_to_num(act, copy=False)
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


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.ERROR if not test_mode else logging.DEBUG,
            log_to_driver=test_mode,
            local_mode=local_mode,
            num_gpus=num_gpus,
            **kwargs
        )
        print("Successfully initialize Ray!")
    if not local_mode:
        print("Available resources: ", ray.available_resources())


def on_postprocess_traj(info):
    """We correct the count of the MultiAgentBatch"""
    episode = info['episode']

    if episode._policies[info['agent_id']].config[ONLY_TNB]:
        return

    post_batch = info['post_batch']
    all_pre_batches = info['all_pre_batches']
    agent_id = next(iter(all_pre_batches.keys()))

    if agent_id != info['agent_id']:
        return

    increment_count = int(
        np.mean([b.count for _, b in all_pre_batches.values()])
    )
    current_count = episode.batch_builder.count
    corrected_count = current_count - increment_count + post_batch.count
    episode.batch_builder.count = corrected_count


USE_BISECTOR = "use_bisector"
USE_DIVERSITY_VALUE_NETWORK = "use_diversity_value_network"
CLIP_DIVERSITY_GRADIENT = "clip_diversity_gradient"
DELAY_UPDATE = "delay_update"
DIVERSITY_REWARD_TYPE = "diversity_reward_type"
TWO_SIDE_CLIP_LOSS = "two_side_clip_loss"
ONLY_TNB = "only_tnb"

DIVERSITY_REWARDS = "diversity_rewards"
DIVERSITY_VALUES = "diversity_values"
DIVERSITY_ADVANTAGES = "diversity_advantages"
DIVERSITY_VALUE_TARGETS = "diversity_value_targets"
PURE_OFF_POLICY = "pure_off_policy"
NORMALIZE_ADVANTAGE = "normalize_advantage"

dice_default_config = merge_dicts(
    DEFAULT_CONFIG, {
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
        "callbacks": {
            "on_postprocess_traj": on_postprocess_traj
        }
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
