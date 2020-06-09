from gym.spaces import Box, Discrete
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin, \
    TargetNetworkMixin
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy, \
    postprocess_trajectory
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.tf_ops import make_tf_callable

from dice.dice_ppo.dice_postprocess import MY_LOGIT
from dice.dice_sac.dice_sac_config import dice_sac_default_config, \
    USE_MY_TARGET_DIVERSITY
from dice.dice_sac.dice_sac_gradient import dice_sac_gradient, dice_sac_loss, \
    apply_gradients, get_dist_class
from dice.dice_sac.dice_sac_model import DiCESACModel
from dice.utils import *


def postprocess_dice_sac(policy, sample_batch, others_batches, episode):
    # The CE part is move to optimizer
    batch = postprocess_trajectory(policy, sample_batch)
    if policy.config[ONLY_TNB] or not policy.loss_initialized():
        batch["diversity_rewards"] = np.zeros_like(
            batch[SampleBatch.REWARDS], dtype=np.float32
        )
    return batch


def stats_fn(policy, train_batch):
    return {
        "td_error": tf.reduce_mean(policy.td_error),
        "diversity_td_error": tf.reduce_mean(policy.diversity_td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        "alpha_loss": tf.reduce_mean(policy.alpha_loss),
        "target_entropy": tf.reduce_mean(policy.target_entropy),
        "entropy": tf.reduce_mean(policy.entropy),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
        "diversity_actor_loss": tf.reduce_mean(policy.diversity_actor_loss),
        "diversity_critic_loss": tf.reduce_mean(policy.diversity_critic_loss),
        "diversity_reward_mean": tf.reduce_mean(policy.diversity_reward_mean),
        "diversity_reward_max": tf.reduce_max(policy.diversity_reward_mean),
        "diversity_reward_min": tf.reduce_min(policy.diversity_reward_mean)
    }


class DiCETargetNetworkMixin:
    """SAC already have target network, so we do not need to maintain
    another set of target network.

    This mixin provide the same API of DiCE:
    policy._compute_clone_network_logits
    """

    def __init__(self):
        action_dist_class = get_dist_class(self.config, self.action_space)

        if self.config[DELAY_UPDATE]:
            used_model = self.target_model
        else:
            used_model = self.model

        @make_tf_callable(self.get_session(), True)
        def compute_clone_network_action(ob):
            feed_dict = {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                "is_training": tf.convert_to_tensor(False)
            }
            model_out, _ = used_model(feed_dict)
            distribution_inputs = used_model.get_policy_output(model_out)
            dist = action_dist_class(distribution_inputs, used_model)
            action = dist.deterministic_sample()
            return action

        if self.config[USE_MY_TARGET_DIVERSITY]:
            my_used_model = self.target_model
        else:
            my_used_model = self.model

        @make_tf_callable(self.get_session(), True)
        def compute_my_deterministic_action(ob):
            feed_dict = {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                "is_training": tf.convert_to_tensor(False)
            }
            model_out, _ = my_used_model(feed_dict)
            distribution_inputs = my_used_model.get_policy_output(model_out)
            dist = action_dist_class(distribution_inputs, my_used_model)
            action = dist.deterministic_sample()
            return action

        self._compute_clone_network_action = compute_clone_network_action
        self._compute_my_deterministic_action = compute_my_deterministic_action


def after_init(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)
    DiCETargetNetworkMixin.__init__(policy)


def before_loss_init(policy, obs_space, action_space, config):
    ComputeDiversityMixinModified.__init__(policy)
    ComputeTDErrorMixin.__init__(policy, dice_sac_loss)


def setup_early_mixins(policy, obs_space, action_space, config):
    ActorCriticOptimizerMixinModified.__init__(policy, config)


class ComputeDiversityMixinModified:
    def compute_diversity(self, my_batch, policy_map):
        """Compute the diversity of this agent."""
        replays = []
        assert policy_map
        for other_name, other_policy in policy_map.items():
            replays.append(
                other_policy._compute_clone_network_action(
                    my_batch[SampleBatch.CUR_OBS],
                )
            )
        # Compute the diversity loss based on the action distribution of
        # this policy and other polices.
        return np.mean(
            [
                (np.square(my_batch[MY_LOGIT] - other_act)).mean(1)
                for other_act in replays
            ],
            axis=0
        )


def build_sac_model(policy, obs_space, action_space, config):
    if config["model"].get("custom_model"):
        logger.warning(
            "Setting use_state_preprocessor=True since a custom model "
            "was specified."
        )
        config["use_state_preprocessor"] = True
    if not isinstance(action_space, (Box, Discrete)):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for SAC.".format(action_space)
        )
    if isinstance(action_space, Box) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space has multiple dimensions "
            "{}. ".format(action_space.shape) +
            "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API."
        )

    # 2 cases:
    # 1) with separate state-preprocessor (before obs+action concat).
    # 2) no separate state-preprocessor: concat obs+actions right away.
    if config["use_state_preprocessor"]:
        num_outputs = 256  # Flatten last Conv2D to this many nodes.
    else:
        num_outputs = 0
        # No state preprocessor: fcnet_hiddens should be empty.
        if config["model"]["fcnet_hiddens"]:
            logger.warning(
                "When not using a state-preprocessor with SAC, `fcnet_hiddens`"
                " will be set to an empty list! Any hidden layer sizes are "
                "defined via `policy_model.hidden_layer_sizes` and "
                "`Q_model.hidden_layer_sizes`."
            )
            config["model"]["fcnet_hiddens"] = []

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's "Q_model" and "policy_model"
    # settings.
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch" if config["use_pytorch"] else "tf",
        model_interface=DiCESACModel,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        diversity_twin_q=config["diversity_twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"]
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch" if config["use_pytorch"] else "tf",
        model_interface=DiCESACModel,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        diversity_twin_q=config["diversity_twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"]
    )

    return policy.model


def extra_learn_fetches_fn(policy):
    return {
        "td_error": policy.td_error,
        "diversity_td_error": policy.diversity_td_error,
    }


class ActorCriticOptimizerMixinModified:
    def __init__(self, config):
        # create global step for counting the number of update operations
        self.global_step = tf.train.get_or_create_global_step()

        # use separate optimizers for actor & critic
        self._actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["actor_learning_rate"]
        )
        self._critic_optimizer = [
            tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"]
            )
        ]
        self._diversity_critic_optimizer = [
            tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"]
            )
        ]
        if config["twin_q"]:
            self._critic_optimizer.append(
                tf.train.AdamOptimizer(
                    learning_rate=config["optimization"]
                    ["critic_learning_rate"]
                )
            )
        if config["diversity_twin_q"]:
            self._diversity_critic_optimizer.append(
                tf.train.AdamOptimizer(
                    learning_rate=config["optimization"]
                    ["critic_learning_rate"]
                )
            )
        self._alpha_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["entropy_learning_rate"]
        )


def get_distribution_inputs_and_class_modified(
        policy, model, obs_batch, *, explore=True, **kwargs
):
    # Get base-model output.
    model_out, state_out = model(
        {
            "obs": obs_batch,
            "is_training": policy._get_is_training_placeholder(),
        }, [], None
    )
    # Get action model output from base-model output.
    distribution_inputs = model.get_policy_output(model_out)
    action_dist_class = get_dist_class(policy.config, policy.action_space)
    return distribution_inputs, action_dist_class, state_out


def grad_stats_fn(policy, batch, grads):
    return {
        "cos_similarity": policy.gradient_cosine_similarity,
        "policy_grad_norm": policy.policy_grad_norm,
        "diversity_grad_norm": policy.diversity_grad_norm
    }


DiCESACPolicy = SACTFPolicy.with_updates(
    name="DiCESACPolicy",
    get_default_config=lambda: dice_sac_default_config,
    make_model=build_sac_model,
    postprocess_fn=postprocess_dice_sac,
    action_distribution_fn=get_distribution_inputs_and_class_modified,
    loss_fn=dice_sac_loss,
    stats_fn=stats_fn,
    gradients_fn=dice_sac_gradient,
    apply_gradients_fn=apply_gradients,
    extra_learn_fetches_fn=extra_learn_fetches_fn,
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixinModified,
        ComputeTDErrorMixin, DiCETargetNetworkMixin,
        ComputeDiversityMixinModified
    ],
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    after_init=after_init,

    # extra_action_fetches_fn=extra_action_fetches_fn,
    grad_stats_fn=grad_stats_fn,
)
