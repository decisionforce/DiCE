"""
We build a DiCE trainer in this file.

In each DiCE trainer, we have multiple polices. We maintain a reference of
the whole team of polices in each policy. So that for each policy it can
query other policies' responses on a given observation.

The reference of the whole team of policies is called policy map, and it is
initialized in the setup_policies_pool function below. After each iteration,
the after_optimizer_iteration is called to update the policies map in each
policy if necessary.

We also validate the config of the DiCE trainer in this file.
"""
from ray.rllib.agents.ppo.ppo import PPOTrainer, update_kl, \
    validate_config as validate_config_original
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.optimizers import LocalMultiGPUOptimizer
from ray.tune.registry import _global_registry, ENV_CREATOR

from dice.dice_ppo.dice_model import ActorDoubleCriticNetwork
from dice.dice_ppo.dice_policy import DiCEPolicy
from dice.utils import *

logger = logging.getLogger(__name__)


def validate_config(config):
    """Validate the config"""

    # create multi-agent environment
    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        i: (None, tmp_env.observation_space, tmp_env.action_space, {})
        for i in tmp_env.agent_ids
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x

    # check the model
    if config[USE_DIVERSITY_VALUE_NETWORK]:
        ModelCatalog.register_custom_model(
            "ActorDoubleCriticNetwork", ActorDoubleCriticNetwork
        )
        config['model']['custom_model'] = "ActorDoubleCriticNetwork"
        config['model']['custom_options'] = {
            "use_diversity_value_network": config[USE_DIVERSITY_VALUE_NETWORK]
        }
    else:
        config['model']['custom_model'] = None
        config['model']['custom_options'] = None

    # validate other elements of PPO config
    validate_config_original(config)


def make_policy_optimizer_tnbes(workers, config):
    """We implement the knob of NORMALIZE_ADVANTAGE here."""
    if config["simple_optimizer"]:
        raise NotImplementedError()

    if config[NORMALIZE_ADVANTAGE]:
        normalized_fields = ["advantages", DIVERSITY_ADVANTAGES]
    else:
        normalized_fields = []

    return LocalMultiGPUOptimizer(
        workers,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        rollout_fragment_length=config["rollout_fragment_length"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=normalized_fields,
        shuffle_sequences=config["shuffle_sequences"]
    )


def setup_policies_pool(trainer):
    """Initialize the team of agents by calling the function in each policy"""
    if not trainer.config[DELAY_UPDATE]:
        return
    assert not trainer.get_policy('agent0').initialized_policies_pool
    # First step, broadcast local weights to remote worker.
    if trainer.workers.remote_workers():
        weights = ray.put(trainer.workers.local_worker().get_weights())
        for e in trainer.workers.remote_workers():
            e.set_weights.remote(weights)

    # Second step, call the _lazy_initialize function of each policy, feeding
    # with the policies map in the trainer.
    def _init_pool(worker, worker_index):
        def _init_diversity_policy(policy, my_policy_name):
            # policy.update_target_network(tau=1.0)
            policy.update_target(tau=1.0)
            policy._lazy_initialize(worker.policy_map, my_policy_name)

        worker.foreach_policy(_init_diversity_policy)

    trainer.workers.foreach_worker_with_index(_init_pool)


def after_optimizer_iteration(trainer, fetches):
    """Update the policies pool in each policy."""
    update_kl(trainer, fetches)  # original PPO procedure

    # only update the policies pool if used DELAY_UPDATE, otherwise
    # the policies_pool in each policy is simply not used, so we don't
    # need to update it.
    if trainer.config[DELAY_UPDATE]:
        if trainer.workers.remote_workers():
            weights = ray.put(trainer.workers.local_worker().get_weights())
            for e in trainer.workers.remote_workers():
                e.set_weights.remote(weights)

            def _delay_update_for_worker(worker, worker_index):
                worker.foreach_policy(lambda p, _: p.update_target())

            trainer.workers.foreach_worker_with_index(_delay_update_for_worker)


def get_policy_class(config):
    return DiCEPolicy


DiCETrainer = PPOTrainer.with_updates(
    name="DiCETrainer",
    default_config=dice_default_config,
    default_policy=DiCEPolicy,
    get_policy_class=get_policy_class,
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_tnbes,
    after_init=setup_policies_pool,
    after_optimizer_step=after_optimizer_iteration,
)
