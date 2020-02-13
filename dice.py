from ray.rllib.agents.ppo.ppo import PPOTrainer, update_kl, \
    validate_config as validate_config_original
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.optimizers import LocalMultiGPUOptimizer
from ray.tune.registry import _global_registry, ENV_CREATOR

from dice_model import ActorDoubleCriticNetwork
from dice_policy import DiCEPolicy
from utils import *

logger = logging.getLogger(__name__)


def validate_config(config):
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

    validate_config_original(config)


def make_policy_optimizer_tnbes(workers, config):
    """The original optimizer has wrong number of trained samples stats.
    So we make little modification and use the corrected optimizer.
    This function is only made for PPO.
    """
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
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=normalized_fields,
        shuffle_sequences=config["shuffle_sequences"]
    )


def setup_policies_pool(trainer):
    """
    Three steps to initialize policies pool.
    First, sync weights of policies in local_worker.policy_map to remote
    workers.
    Second, build polices in each worker, based on the policy_map in them.
    Third, build the target model in each policy in policy_map, by syncing
    the weights of policy.model.
    """
    if not trainer.config[DELAY_UPDATE]:
        return
    assert not trainer.get_policy('agent0').initialized_policies_pool
    # first step, broadcast local weights to remote worker.
    if trainer.workers.remote_workers():
        weights = ray.put(trainer.workers.local_worker().get_weights())
        for e in trainer.workers.remote_workers():
            e.set_weights.remote(weights)
        # by doing these, we sync the worker.polices for all workers.

    def _init_pool(worker, worker_index):
        def _init_diversity_policy(policy, my_policy_name):
            policy.update_target_network(tau=1.0)
            policy._lazy_initialize(worker.policy_map, my_policy_name)

        worker.foreach_policy(_init_diversity_policy)

    trainer.workers.foreach_worker_with_index(_init_pool)


def after_optimizer_iteration(trainer, fetches):
    """Update the policies pool in each policy."""
    update_kl(trainer, fetches)

    if trainer.config[DELAY_UPDATE]:
        if trainer.workers.remote_workers():
            weights = ray.put(trainer.workers.local_worker().get_weights())
            for e in trainer.workers.remote_workers():
                e.set_weights.remote(weights)

            def _delay_update_for_worker(worker, worker_index):
                worker.foreach_policy(lambda p, _: p.update_target_network())

            trainer.workers.foreach_worker_with_index(_delay_update_for_worker)


DiCETrainer = PPOTrainer.with_updates(
    name="DiCETrainer",
    default_config=dice_default_config,
    default_policy=DiCEPolicy,
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_tnbes,
    after_init=setup_policies_pool,
    after_optimizer_step=after_optimizer_iteration,
)
