"""
The Two-side Clip Loss and the diversity-regularized learning is implemented
in this file.

First we compute the task loss and the diversity loss in dice_loss. Then we
implement the Diversity Regularization module in dice_gradient.
"""
from ray.rllib.agents.ppo.ppo_tf_policy import PPOLoss
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch

from dice.utils import *

tf = try_import_tf()
logger = logging.getLogger(__name__)
BEHAVIOUR_LOGITS = SampleBatch.ACTION_DIST_INPUTS


class PPOLossTwoSideDiversity(object):
    """Compute the PPO loss for diversity without diversity value network"""

    def __init__(
            self,
            dist_class,
            model,
            advantages,
            actions,
            prev_logits,
            prev_actions_logp,
            curr_action_dist,
            cur_kl_coeff,
            valid_mask,
            entropy_coeff=0,
            clip_param=0.1
    ):
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        # prev_dist = dist_class(prev_logits, model)
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        self.debug_ratio = logp_ratio
        # action_kl = prev_dist.kl(curr_action_dist)
        # self.mean_kl = reduce_mean_valid(action_kl)
        # curr_entropy = curr_action_dist.entropy()
        # self.mean_entropy = reduce_mean_valid(curr_entropy)
        new_surrogate_loss = advantages * tf.minimum(
            logp_ratio, 1 + clip_param
        )
        self.mean_policy_loss = reduce_mean_valid(-new_surrogate_loss)
        self.mean_vf_loss = tf.constant(0.0)
        loss = reduce_mean_valid(
            -new_surrogate_loss
            # Update(20200417): Remove KL and Entropy in Diversity
            # + cur_kl_coeff * action_kl -
            # entropy_coeff * curr_entropy
        )
        self.loss = loss


class PPOLossTwoSideClip:
    def __init__(
            self,
            # _useless,
            dist_class,
            model,
            value_targets,
            advantages,
            actions,
            prev_logits,
            prev_actions_logp,
            vf_preds,
            curr_action_dist,
            value_fn,
            cur_kl_coeff,
            valid_mask,
            entropy_coeff=0,
            clip_param=0.1,
            vf_clip_param=0.1,
            vf_loss_coeff=1.0,
            use_gae=True,
            vf_ratio_clip_param=0.05
    ):
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)

        new_vf_mask = tf.logical_and(
            logp_ratio > 1 - vf_ratio_clip_param,
            logp_ratio < 1 + vf_ratio_clip_param
        )
        self.debug_ratio = logp_ratio
        self.vf_debug_ratio = tf.cast(new_vf_mask, tf.float32)

        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)
        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        new_surrogate_loss = advantages * tf.minimum(
            logp_ratio, 1 + clip_param
        )
        self.mean_policy_loss = reduce_mean_valid(-new_surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param
            )
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)

            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -new_surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy
            )
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(
                -new_surrogate_loss + cur_kl_coeff * action_kl -
                entropy_coeff * curr_entropy
            )
        self.loss = loss


def dice_loss(policy, model, dist_class, train_batch):
    """Compute the task loss and the diversity loss for gradients computing."""
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool
        )

    loss_cls = PPOLossTwoSideClip \
        if policy.config[TWO_SIDE_CLIP_LOSS] else PPOLoss

    policy.loss_obj = loss_cls(
        # None,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch["action_logp"],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        # vf_ratio_clip_param=policy.config["vf_ratio_clip_param"]
    )

    # Build the loss for diversity
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        # if we don't use DVN, we don't have diversity values, so the
        # entries of loss object is also changed.
        policy.diversity_loss_obj = loss_cls(
            policy.action_space,
            dist_class,
            model,
            train_batch[DIVERSITY_VALUE_TARGETS],
            train_batch[DIVERSITY_ADVANTAGES],
            train_batch[SampleBatch.ACTIONS],
            train_batch[BEHAVIOUR_LOGITS],
            train_batch["action_logp"],
            train_batch[DIVERSITY_VALUES],
            action_dist,
            model.diversity_value_function(),
            policy.kl_coeff,
            mask,
            entropy_coeff=policy.entropy_coeff,
            clip_param=policy.config["clip_param"],
            vf_clip_param=policy.config["vf_clip_param"],
            vf_loss_coeff=policy.config["vf_loss_coeff"],
            use_gae=policy.config["use_gae"]
        )
    else:
        policy.diversity_loss_obj = PPOLossTwoSideDiversity(
            dist_class,
            model,
            train_batch[DIVERSITY_ADVANTAGES],
            train_batch[SampleBatch.ACTIONS],
            train_batch[BEHAVIOUR_LOGITS],
            train_batch["action_logp"],
            action_dist,
            policy.kl_coeff,
            mask,
            entropy_coeff=policy.entropy_coeff,
            clip_param=policy.config["clip_param"]
        )

    # Add the diversity reward as a stat
    policy.diversity_reward_mean = tf.reduce_mean(
        train_batch[DIVERSITY_REWARDS]
    )
    return [policy.loss_obj.loss, policy.diversity_loss_obj.loss]


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def dice_gradient(policy, optimizer, loss):
    """Implement the idea of gradients bisector to fuse the task gradients
    with the diversity gradient.
    """
    if not policy.config[USE_BISECTOR]:
        # For ablation study. If don't use bisector, we simply return the
        # task gradient.

        # FIXING BUG (20200416) What happen if I remove dependency?
        variables = policy.model.trainable_variables()
        with tf.control_dependencies([tf.stop_gradient(loss[1])]):
            policy_grad = optimizer.compute_gradients(loss[0], variables)
        if policy.config["grad_clip"] is not None:
            clipped_grads, _ = tf.clip_by_global_norm(
                [g for g, _ in policy_grad], policy.config["grad_clip"]
            )
            return list(zip(clipped_grads, variables))
        else:
            return policy_grad

    policy_grad = optimizer.compute_gradients(loss[0])
    diversity_grad = optimizer.compute_gradients(loss[1])

    return_gradients = {}
    policy_grad_flatten = []
    policy_grad_info = []
    diversity_grad_flatten = []
    diversity_grad_info = []

    # First, flatten task gradient and diversity gradient into two vector.
    for (pg, var), (ng, var2) in zip(policy_grad, diversity_grad):
        assert var == var2
        if pg is None:
            return_gradients[var] = (ng, var2)
            continue
        if ng is None:
            return_gradients[var] = (pg, var)
            continue

        pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
        policy_grad_flatten.append(pg_flat)
        policy_grad_info.append((pg_flat_shape, pg_shape, var))

        ng_flat, ng_shape, ng_flat_shape = _flatten(ng)
        diversity_grad_flatten.append(ng_flat)
        diversity_grad_info.append((ng_flat_shape, ng_shape))

    policy_grad_flatten = tf.concat(policy_grad_flatten, 0)
    diversity_grad_flatten = tf.concat(diversity_grad_flatten, 0)

    # Second, compute the norm of two gradient.
    policy_grad_norm = tf.linalg.l2_normalize(policy_grad_flatten)
    diversity_grad_norm = tf.linalg.l2_normalize(diversity_grad_flatten)

    # Third, compute the bisector.
    final_grad = tf.linalg.l2_normalize(policy_grad_norm + diversity_grad_norm)

    # Fourth, compute the length of the final gradient.
    pg_length = tf.norm(tf.multiply(policy_grad_flatten, final_grad))
    ng_length = tf.norm(tf.multiply(diversity_grad_flatten, final_grad))
    if policy.config[CLIP_DIVERSITY_GRADIENT]:
        ng_length = tf.minimum(pg_length, ng_length)
    tg_lenth = (pg_length + ng_length) / 2

    final_grad = final_grad * tg_lenth

    # add some stats.
    policy.gradient_cosine_similarity = tf.reduce_sum(
        tf.multiply(policy_grad_norm, diversity_grad_norm)
    )
    policy.policy_grad_norm = tf.norm(policy_grad_flatten)
    policy.diversity_grad_norm = tf.norm(diversity_grad_flatten)

    # Fifth, split the flatten vector into the original form as the final
    # gradients.
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        assert flat_shape is not None
        size = flat_shape.as_list()[0]
        grad = final_grad[count:count + size]
        return_gradients[var] = (tf.reshape(grad, org_shape), var)
        count += size

    if policy.config["grad_clip"] is not None:
        ret_grads = [return_gradients[var][0] for _, var in policy_grad]
        clipped_grads, _ = tf.clip_by_global_norm(
            ret_grads, policy.config["grad_clip"]
        )
        return [
            (g, return_gradients[var][1])
            for g, (_, var) in zip(clipped_grads, policy_grad)
        ]
    else:
        return [return_gradients[var] for _, var in policy_grad]
