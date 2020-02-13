import tensorflow as tf
from ray.rllib.agents.ppo.ppo_policy import setup_mixins, ValueNetworkMixin, \
    EntropyCoeffSchedule, BEHAVIOUR_LOGITS, PPOTFPolicy, \
    KLCoeffMixin
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable

from dice_loss import dice_loss, dice_gradient
from dice_postprocess import postprocess_dice, MY_LOGIT
from utils import *

logger = logging.getLogger(__name__)


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


class NoveltyValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"] and config[USE_DIVERSITY_VALUE_NETWORK]:

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
    """Adds value function and logits outputs to experience train_batches."""
    ret = {
        BEHAVIOUR_LOGITS: policy.model.last_output(),
        SampleBatch.VF_PREDS: policy.model.value_function()
    }
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret[DIVERSITY_VALUES] = policy.model.diversity_value_function()
    return ret


def kl_and_loss_stats_modified(policy, train_batch):
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
        "diversity_kl": policy.diversity_loss_obj.mean_kl,
        "diversity_entropy": policy.diversity_loss_obj.mean_entropy,
        "diversity_reward_mean": policy.diversity_reward_mean,
        # "debug_ratio": policy.debug_ratio,
        # "abs_advantage": policy.abs_advantage
    }
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret['diversity_vf_explained_var'] = explained_variance(
            train_batch[DIVERSITY_VALUE_TARGETS],
            policy.model.diversity_value_function()
        )
    return ret


class ComputeNoveltyMixin:
    def __init__(self):
        self.initialized_policies_pool = False
        self.policies_pool = {}

    def _lazy_initialize(self, policies_pool, my_name):
        assert self.config[DELAY_UPDATE]
        self.policies_pool = {
            agent_name: other_policy
            for agent_name, other_policy in policies_pool.items()
            # if agent_name != my_name
        }  # Since it must in DELAY_UPDATE mode, we allow reuse all polices.
        self.num_of_policies = len(self.policies_pool)
        self.initialized_policies_pool = True

    def compute_diversity(self, my_batch, others_batches, use_my_logit):
        """It should be noted that in Cooperative Exploration setting,
        This implementation is inefficient. Since the 'observation' of each
        agent are identical, though may different in order, so we call the
        compute_actions for num_agents * num_agents * batch_size times overall.
        """
        replays = {}
        if self.config[DELAY_UPDATE]:
            for other_name, other_policy in self.policies_pool.items():
                logits = other_policy._compute_clone_network_logits(
                    my_batch[SampleBatch.CUR_OBS]
                )
                replays[other_name] = logits
        else:
            if not others_batches:
                return np.zeros_like(
                    my_batch[SampleBatch.REWARDS], dtype=np.float32
                )
            for other_name, (other_policy, _) in others_batches.items():
                _, _, info = other_policy.compute_actions(
                    my_batch[SampleBatch.CUR_OBS]
                )
                replays[other_name] = info[BEHAVIOUR_LOGITS]

        logit_key = MY_LOGIT if use_my_logit else BEHAVIOUR_LOGITS

        if self.config[DIVERSITY_REWARD_TYPE] == "kl":
            return np.mean(
                [
                    get_kl_divergence(my_batch[logit_key], logit, mean=False)
                    for logit in replays.values()
                ],
                axis=0
            )

        elif self.config[DIVERSITY_REWARD_TYPE] == "mse":
            replays = [
                np.split(logit, 2, axis=1)[0] for logit in replays.values()
            ]
            my_act = np.split(my_batch[logit_key], 2, axis=1)[0]
            return np.mean(
                [
                    (np.square(my_act - other_act)).mean(1)
                    for other_act in replays
                ],
                axis=0
            )
        else:
            raise NotImplementedError()


def setup_mixins_dece(policy, action_space, obs_space, config):
    setup_mixins(policy, action_space, obs_space, config)
    NoveltyValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    ComputeNoveltyMixin.__init__(policy)


class TargetNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        """Target Network is updated by the master learner every
        trainer.update_target_frequency steps. All worker batches
        are importance sampled w.r. to the target network to ensure
        a more stable pi_old in PPO.
        """
        assert config[DELAY_UPDATE]
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

        self.get_session().run(tf.initialize_variables(self.target_model_vars))

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

    def update_target_network(self, tau=None):
        tau = tau or self.tau_value
        return self.get_session().run(
            self.update_target_expr, feed_dict={self.tau: tau}
        )


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
    before_loss_init=setup_mixins_dece,
    after_init=setup_late_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, NoveltyValueNetworkMixin, ComputeNoveltyMixin,
        TargetNetworkMixin
    ]
)
