import time

import numpy as np
from ray.rllib.agents.sac.sac import SACTrainer, \
    validate_config as validate_config_sac
from ray.rllib.optimizers.sync_replay_optimizer import SyncReplayOptimizer, \
    PrioritizedReplayBuffer, SampleBatch, MultiAgentBatch
from ray.tune.registry import _global_registry, ENV_CREATOR

from dice.dice_ppo.dice_postprocess import MY_LOGIT
from dice.dice_sac.dice_sac_config import dice_sac_default_config, SHARE_BUFFER
from dice.dice_sac.dice_sac_policy import DiCESACPolicy


def validate_config(config):
    validate_config_sac(config)

    # Hard-coded this setting
    assert not config["normalize_actions"]
    assert config["env_config"]["normalize_actions"]

    # create multi-agent environment
    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        i: (None, tmp_env.observation_space, tmp_env.action_space, {})
        for i in tmp_env.agent_ids
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x
    config['model']['custom_model'] = None
    config['model']['custom_options'] = None


def _before_learn_on_batch(samples, policy_map, train_batch_size):
    for agent_id, my_batch in samples.policy_batches.items():
        assert agent_id in policy_map, (agent_id, policy_map.keys())
        my_batch[MY_LOGIT] = \
            policy_map[agent_id]._compute_my_deterministic_action(
                my_batch["obs"])
        samples.policy_batches[agent_id]["diversity_rewards"] = \
            policy_map[agent_id].compute_diversity(
                my_batch,
                {pid: p for pid, p in policy_map.items() if pid != agent_id}
            )
    return samples


class SyncReplayOptimizerModified(SyncReplayOptimizer):
    def __init__(self, *args, **kwargs):
        assert "num_agents" in kwargs
        self.num_agents = kwargs.pop("num_agents")
        super().__init__(*args, **kwargs)
        self.config = self.workers._remote_config

    def step(self):
        super().step()

        # Workaround to make sure the data is log into result.
        now = time.time()
        while self.num_steps_sampled < self.replay_starts + \
                self.config["timesteps_per_iteration"]:
            if time.time() - now > 10:
                print(
                    "Current samples are not sufficient for learning. Launch"
                    " another sample! Current steps sampled {} and replay "
                    "will starts at {}".format(
                        self.num_steps_sampled, self.replay_starts
                    )
                )
                now = time.time()
            super().step()

    def _sample_a_mini_batch(self, idxes, policy_id, batch_size):
        replay_buffer = self.replay_buffers[policy_id]
        if self.synchronize_sampling:
            if idxes is None:
                idxes = replay_buffer.sample_idxes(batch_size)
        else:
            idxes = replay_buffer.sample_idxes(batch_size)

        if isinstance(replay_buffer, PrioritizedReplayBuffer):
            raise NotImplementedError()
        else:
            (obses_t, actions, rewards, obses_tp1,
             dones) = replay_buffer.sample_with_idxes(idxes)
            weights = np.ones_like(rewards)
            batch_indexes = -np.ones_like(rewards)
        return SampleBatch(
            {
                "obs": obses_t,
                "actions": actions,
                "rewards": rewards,
                "new_obs": obses_tp1,
                "dones": dones,
                "weights": weights,
                "batch_indexes": batch_indexes
            }
        )

    def _replay(self):
        # Here use small batch to get the data
        per_agent_train_batch_size = int(
            self.train_batch_size / self.num_agents
        )

        if self.config[SHARE_BUFFER]:
            samples = {}
            with self.replay_timer:
                for policy_id in self.replay_buffers.keys():
                    mini_batches = [
                        self._sample_a_mini_batch(
                            None, oppo, per_agent_train_batch_size
                        ) for oppo in self.replay_buffers.keys()
                    ]
                    samples[policy_id] = SampleBatch.concat_samples(
                        mini_batches
                    )
            return MultiAgentBatch(
                samples, sum(s.count for s in samples.values())
            )

        samples = {}
        idxes = None
        with self.replay_timer:
            for policy_id, replay_buffer in self.replay_buffers.items():
                mini_batch = self._sample_a_mini_batch(
                    idxes, policy_id, per_agent_train_batch_size
                )
                samples[policy_id] = mini_batch

        # Merge the things here
        shared_batch = SampleBatch.concat_samples(list(samples.values()))
        return_batch = MultiAgentBatch(
            {k: shared_batch
             for k in samples.keys()}, shared_batch.count
        )
        return return_batch


def make_policy_optimizer(workers, config):
    """Create the single process DQN policy optimizer.

    Returns:
        SyncReplayOptimizer: Used for generic off-policy Trainers.
    """
    # SimpleQ does not use a PR buffer.
    kwargs = {"prioritized_replay": config.get("prioritized_replay", False)}
    kwargs.update(**config["optimizer"])
    if "prioritized_replay" in config:
        kwargs.update(
            {
                "prioritized_replay_alpha": config["prioritized_replay_alpha"],
                "prioritized_replay_beta": config["prioritized_replay_beta"],
                "prioritized_replay_beta_annealing_timesteps": config[
                    "prioritized_replay_beta_annealing_timesteps"],
                "final_prioritized_replay_beta": config[
                    "final_prioritized_replay_beta"],
                "prioritized_replay_eps": config["prioritized_replay_eps"],
            }
        )

    return SyncReplayOptimizerModified(
        workers,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        train_batch_size=config["train_batch_size"],
        before_learn_on_batch=_before_learn_on_batch,  # <<== Add extra callback
        num_agents=config["env_config"]["num_agents"],
        **kwargs
    )


DiCESACTrainer = SACTrainer.with_updates(
    name="DiCESACTrainer",
    default_config=dice_sac_default_config,
    validate_config=validate_config,
    default_policy=DiCESACPolicy,
    get_policy_class=lambda _: DiCESACPolicy,

    # Rewrite this to add the term _before_learn_on_batch
    make_policy_optimizer=make_policy_optimizer
)
