import tensorflow as tf
from ray.rllib.agents.impala.vtrace_policy import BEHAVIOUR_LOGITS
from ray.rllib.agents.ppo.ppo_policy import BEHAVIOUR_LOGITS, \
    PPOLoss as original_PPOLoss
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch

from utils import *

logger = logging.getLogger(__name__)


def PPOLoss(*args, **kwargs):
    """A workaround"""
    if "is_ratio" in kwargs:
        kwargs.pop("is_ratio")
    return original_PPOLoss(*args, **kwargs)


class PPOLossTwoSideNovelty(object):
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

        prev_dist = dist_class(prev_logits, model)
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)
        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)
        new_surrogate_loss = advantages * tf.minimum(
            logp_ratio, 1 + clip_param
        )
        self.mean_policy_loss = reduce_mean_valid(-new_surrogate_loss)
        self.mean_vf_loss = tf.constant(0.0)
        loss = reduce_mean_valid(
            -new_surrogate_loss + cur_kl_coeff * action_kl -
            entropy_coeff * curr_entropy
        )
        self.loss = loss


class PPOLossTwoSideClip(object):
    def __init__(
            self,
            action_space,
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
            model_config=None,
            is_ratio=None
    ):
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
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
    """Add diversity loss with original ppo loss using TNB method"""
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
        policy.action_space,
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
        model_config=policy.config["model"],
    )
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
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
            use_gae=policy.config["use_gae"],
            model_config=policy.config["model"],
            is_ratio=train_batch['is_ratio']
        )
    else:
        policy.diversity_loss_obj = PPOLossTwoSideNovelty(
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
    policy.diversity_reward_mean = tf.reduce_mean(
        train_batch[DIVERSITY_REWARDS]
    )
    # policy.debug_ratio = train_batch["debug_ratio"]
    # policy.abs_advantage = train_batch["abs_advantage"]
    return [policy.loss_obj.loss, policy.diversity_loss_obj.loss]


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def dice_gradient(policy, optimizer, loss):
    if not policy.config[USE_BISECTOR]:
        with tf.control_dependencies([loss[1]]):
            policy_grad = optimizer.compute_gradients(loss[0])
        return policy_grad

    policy_grad = optimizer.compute_gradients(loss[0])
    diversity_grad = optimizer.compute_gradients(loss[1])

    return_gradients = {}
    policy_grad_flatten = []
    policy_grad_info = []
    diversity_grad_flatten = []
    diversity_grad_info = []

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

    # implement the logic of TNB
    policy_grad_norm = tf.linalg.l2_normalize(policy_grad_flatten)
    diversity_grad_norm = tf.linalg.l2_normalize(diversity_grad_flatten)
    cos_similarity = tf.reduce_sum(
        tf.multiply(policy_grad_norm, diversity_grad_norm)
    )

    tg = tf.linalg.l2_normalize(policy_grad_norm + diversity_grad_norm)

    pg_length = tf.norm(tf.multiply(policy_grad_flatten, tg))
    ng_length = tf.norm(tf.multiply(diversity_grad_flatten, tg))

    if policy.config[CLIP_DIVERSITY_GRADIENT]:
        ng_length = tf.minimum(pg_length, ng_length)

    tg_lenth = (pg_length + ng_length) / 2
    tg = tg * tg_lenth
    total_grad = tg

    policy.gradient_cosine_similarity = cos_similarity
    policy.policy_grad_norm = tf.norm(policy_grad_flatten)
    policy.diversity_grad_norm = tf.norm(diversity_grad_flatten)

    # reshape back the gradients
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        assert flat_shape is not None
        # return_gradients.append((None, var))
        #     continue
        size = flat_shape.as_list()[0]
        grad = total_grad[count:count + size]
        return_gradients[var] = (tf.reshape(grad, org_shape), var)
        count += size

    return [return_gradients[var] for _, var in policy_grad]
