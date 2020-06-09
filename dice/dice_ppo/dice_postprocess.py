"""
Implement the Collaborative Exploration here.

In postprocess_dice, which is called for each policy with input other
policies' batches, we fuse the batches collected by this policy and other
policies. We also compute the diversity reward and diversity advantage of this
policy.
"""
from ray.rllib.agents.ppo.ppo_tf_policy import postprocess_ppo_gae, \
    SampleBatch
from ray.rllib.evaluation.postprocessing import discount

from dice.utils import *

MY_LOGIT = "my_logits"


def postprocess_dice(policy, sample_batch, others_batches, episode):
    if not policy.loss_initialized():
        batch = postprocess_ppo_gae(policy, sample_batch)
        batch[DIVERSITY_REWARDS] = batch["advantages"].copy()
        batch[DIVERSITY_VALUE_TARGETS] = batch["advantages"].copy()
        batch[DIVERSITY_ADVANTAGES] = batch["advantages"].copy()
        batch['other_action_logp'] = batch[SampleBatch.ACTION_LOGP].copy()
        return batch

    if not others_batches:
        batch = sample_batch.copy()
        batch = postprocess_ppo_gae(policy, batch)
        batch[MY_LOGIT] = batch[SampleBatch.ACTION_DIST_INPUTS]
        batch = postprocess_diversity(policy, batch, others_batches)
        batches = [batch]
    else:
        batches = []

    for pid, (other_policy, other_batch_raw) in others_batches.items():
        # other_batch_raw is the data collected by other polices.
        if policy.config[ONLY_TNB]:
            break
        if other_batch_raw is None:
            continue
        other_batch_raw = other_batch_raw.copy()

        # Replay this policy to get the action distribution of this policy.
        replay_result = policy.compute_actions(
            other_batch_raw[SampleBatch.CUR_OBS]
        )[2]
        other_batch_raw[MY_LOGIT] = replay_result[
            SampleBatch.ACTION_DIST_INPUTS]

        # Compute the diversity reward and diversity advantage of this batch.
        other_batch_raw = postprocess_diversity(
            policy, other_batch_raw, others_batches
        )

        # Compute the task advantage of this batch.
        batches.append(postprocess_ppo_gae(policy, other_batch_raw))

    # Merge all batches.
    batch = SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]

    del batch.data['new_obs']  # save memory
    del batch.data['action_prob']
    if policy.config[ONLY_TNB]:
        assert np.unique(batch["agent_index"]).size == 1
    return batch


def postprocess_diversity(policy, batch, others_batches):
    """Compute the diversity for this policy against other policies using this
    batch."""

    # Compute diversity and add a new entry of batch: diversity_reward
    batch[DIVERSITY_REWARDS] = policy.compute_diversity(batch, others_batches)

    # Compute the diversity advantage. We mock the computing of task advantage
    # but simply replace the task reward with the diversity reward.
    completed = batch["dones"][-1]
    if completed:
        last_r_diversity = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([batch["state_out_{}".format(i)][-1]])
        last_r_diversity = policy._diversity_value(
            batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
            batch[DIVERSITY_REWARDS][-1], *next_state
        )
    diversity_advantages, diversity_value_target = \
        _compute_advantages_for_diversity(
            rewards=batch[DIVERSITY_REWARDS],
            last_r=last_r_diversity,
            gamma=policy.config["gamma"],
            lambda_=policy.config["lambda"],
            values=batch[DIVERSITY_VALUES]
            if policy.config[USE_DIVERSITY_VALUE_NETWORK] else None,
            use_gae=policy.config[USE_DIVERSITY_VALUE_NETWORK]
        )
    batch[DIVERSITY_ADVANTAGES] = diversity_advantages
    batch[DIVERSITY_VALUE_TARGETS] = diversity_value_target
    return batch


def _compute_advantages_for_diversity(
        rewards, last_r, gamma, lambda_, values, use_gae=True
):
    """Compute the diversity advantage."""
    if use_gae:
        vpred_t = np.concatenate([values, np.array([last_r])])
        delta_t = (rewards + gamma * vpred_t[1:] - vpred_t[:-1])
        advantage = discount(delta_t, gamma * lambda_)
        value_target = (advantage + values).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rewards.reshape(-1),
             np.array([last_r]).reshape(-1)]
        )
        if rewards_plus_v.size <= 2:
            logger.warning(
                "********** Current reward is empty: {}. last r {}. values {}"
                ".".format(rewards, last_r, values)
            )
            # rewards_plus_v = np.array([last_r])
        advantage = discount(rewards_plus_v, gamma)[:-1]
        value_target = np.zeros_like(advantage, dtype=np.float32)
    advantage = advantage.copy().astype(np.float32)
    return advantage, value_target
