"""
This file implement a DiCE policy. Note that in one DiCE trainer, there are
many DiCE policies, each serves as a member in the team. We implement the
following functions for each policy:
1. Compute the diversity of one policy against others.
2. Maintain the target network for each policy if in DELAY_UPDATE mode.
3. Update the target network for each training iteration.
"""
from ray.rllib.agents.ppo.ppo_tf_policy import setup_mixins, \
    ValueNetworkMixin, KLCoeffMixin, \
    EntropyCoeffSchedule, PPOTFPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable

from dice.dice_ppo.dice_loss import dice_loss, dice_gradient
from dice.dice_ppo.dice_postprocess import postprocess_dice, MY_LOGIT
from dice.utils import *

logger = logging.getLogger(__name__)
BEHAVIOUR_LOGITS = SampleBatch.ACTION_DIST_INPUTS


def grad_stats_fn(policy, batch, grads):
    if policy.config[USE_BISECTOR]:
        ret = {
            "cos_similarity": policy.gradient_cosine_similarity,
            "policy_grad_norm": policy.policy_grad_norm,
            "diversity_grad_norm": policy.diversity_grad_norm
        }
        return ret
    else:
        return {}


class DiversityValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config.get("use_gae") and config[USE_DIVERSITY_VALUE_NETWORK]:

            @make_tf_callable(self.get_session())
            def diversity_value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model(
                    {
                        SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                        SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                            [prev_action]
                        ),
                        SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                            [prev_reward]
                        ),
                        "is_training": tf.convert_to_tensor(False),
                    }, [tf.convert_to_tensor([s]) for s in state],
                    tf.convert_to_tensor([1])
                )
                return self.model.diversity_value_function()[0]
        else:

            @make_tf_callable(self.get_session())
            def diversity_value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._diversity_value = diversity_value


def additional_fetches(policy):
    """Fetch diversity values if using diversity value network."""
    ret = {
        BEHAVIOUR_LOGITS: policy.model.last_output(),
        SampleBatch.VF_PREDS: policy.model.value_function()
    }
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret[DIVERSITY_VALUES] = policy.model.diversity_value_function()
    return ret


def kl_and_loss_stats_modified(policy, train_batch):
    """Add the diversity-related stats here."""
    ret = {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()
        ),
        "diversity_total_loss": policy.diversity_loss_obj.loss,
        "diversity_policy_loss": policy.diversity_loss_obj.mean_policy_loss,
        "diversity_vf_loss": policy.diversity_loss_obj.mean_vf_loss,
        # "diversity_kl": policy.diversity_loss_obj.mean_kl,
        "debug_ratio": policy.diversity_loss_obj.debug_ratio,
        # "diversity_entropy": policy.diversity_loss_obj.mean_entropy,
        "diversity_reward_mean": policy.diversity_reward_mean,
    }
    if hasattr(policy.loss_obj, "vf_debug_ratio"):
        ret["vf_debug_ratio"] = policy.loss_obj.vf_debug_ratio
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret['diversity_vf_explained_var'] = explained_variance(
            train_batch[DIVERSITY_VALUE_TARGETS],
            policy.model.diversity_value_function()
        )
    return ret


class ComputeDiversityMixin:
    """This class initialize a reference of the policies pool within each
    policy, and provide the function to compute the diversity of each policy.

    The _lazy_initialize is only called in DELAY_UPDATE mode. This is because
    if we compute diversity of this policy against other latest policies,
    we can simply access other policies via other_batches, the input to the
    compute_diversity function.
    """

    def __init__(self, discrete):
        self.initialized_policies_pool = False
        self.policies_pool = {}
        self.discrete = discrete

    def _lazy_initialize(self, policies_pool, my_name=None):
        """Initialize the reference of policies pool within this policy."""
        assert self.config.get(DELAY_UPDATE) or DELAY_UPDATE not in self.config
        self.policies_pool = {
            agent_name: other_policy
            for agent_name, other_policy in policies_pool.items()
            # if agent_name != my_name
        }  # Since it must in DELAY_UPDATE mode, we allow reuse all polices.
        self.num_of_policies = len(self.policies_pool)
        self.initialized_policies_pool = True

    def compute_diversity(self, my_batch, others_batches):
        """Compute the diversity of this agent."""
        replays = {}
        if self.config[DELAY_UPDATE]:
            # If in DELAY_UPDATE mode, compute diversity against the target
            # network of each policies.
            for other_name, other_policy in self.policies_pool.items():
                logits = other_policy._compute_clone_network_logits(
                    my_batch[SampleBatch.CUR_OBS]
                )
                replays[other_name] = logits
        else:
            # Otherwise compute the diversity against other latest policies
            # contained in other_batches.
            if not others_batches:
                return np.zeros_like(
                    my_batch[SampleBatch.REWARDS], dtype=np.float32
                )
            for other_name, (other_policy, _) in others_batches.items():
                _, _, info = other_policy.compute_actions(
                    my_batch[SampleBatch.CUR_OBS]
                )
                replays[other_name] = info[BEHAVIOUR_LOGITS]

        # Compute the diversity loss based on the action distribution of
        # this policy and other polices.
        if self.discrete:  # discrete
            replays = list(replays.values())
            my_act = my_batch[MY_LOGIT]
        else:
            replays = [
                np.split(logit, 2, axis=1)[0] for logit in replays.values()
            ]
            my_act = np.split(my_batch[MY_LOGIT], 2, axis=1)[0]
        return np.mean(
            [(np.square(my_act - other_act)).mean(1) for other_act in replays],
            axis=0
        )


class TargetNetworkMixin:
    """This class implement the DELAY_UPDATE mechanism. Allowing:
    1. delayed update the targets networks of each policy.
    2. allowed fetches of action distribution of the target network of each
    policy.

    Note that this Mixin is with policy. That is to say, the target network
    of each policy is maintain by their own. After each training iteration, all
    policy will update their own target network.
    """

    def __init__(self, obs_space, action_space, config):
        assert config[DELAY_UPDATE]

        # Build the target network of this policy.
        _, logit_dim = ModelCatalog.get_action_dist(
            action_space, config["model"]
        )
        self.target_model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            logit_dim,
            config["model"],
            name="target_func",
            framework="tf"
        )
        self.model_vars = self.model.variables()
        self.target_model_vars = self.target_model.variables()

        self.get_session().run(
            tf.variables_initializer(self.target_model_vars)
        )

        # Here is the delayed update mechanism.
        self.tau_value = config.get("tau")
        self.tau = tf.placeholder(tf.float32, (), name="tau")
        assign_ops = []
        assert len(self.model_vars) == len(self.target_model_vars)
        for var, var_target in zip(self.model_vars, self.target_model_vars):
            assign_ops.append(
                var_target.
                assign(self.tau * var + (1.0 - self.tau) * var_target)
            )
        self.update_target_expr = tf.group(*assign_ops)

        @make_tf_callable(self.get_session(), True)
        def compute_clone_network_logits(ob):
            feed_dict = {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                "is_training": tf.convert_to_tensor(False)
            }
            model_out, _ = self.target_model(feed_dict)
            return model_out

        self._compute_clone_network_logits = compute_clone_network_logits

    def update_target(self, tau=None):
        """Delayed update the target network."""
        tau = tau or self.tau_value
        return self.get_session().run(
            self.update_target_expr, feed_dict={self.tau: tau}
        )


def setup_mixins_dice(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    DiversityValueNetworkMixin.__init__(
        policy, obs_space, action_space, config
    )
    discrete = isinstance(action_space, gym.spaces.Discrete)
    ComputeDiversityMixin.__init__(policy, discrete)


def setup_late_mixins(policy, obs_space, action_space, config):
    if config[DELAY_UPDATE]:
        TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


DiCEPolicy = PPOTFPolicy.with_updates(
    name="DiCEPolicy",
    get_default_config=lambda: dice_default_config,
    postprocess_fn=postprocess_dice,
    loss_fn=dice_loss,
    stats_fn=kl_and_loss_stats_modified,
    gradients_fn=dice_gradient,
    grad_stats_fn=grad_stats_fn,
    extra_action_fetches_fn=additional_fetches,
    before_loss_init=setup_mixins_dice,
    after_init=setup_late_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, DiversityValueNetworkMixin, ComputeDiversityMixin,
        TargetNetworkMixin
    ]
)
