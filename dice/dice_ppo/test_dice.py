"""
Test DiCE trainer
"""
import shutil
import tempfile
import unittest

from dice.dice_ppo.dice_ppo import DiCETrainer
from dice.utils import *

num_agents_pair = tune.grid_search([1, 3])

true_false_pair = tune.grid_search([True, False])


def _test_dice(
        extra_config={},
        local_mode=False,
        num_agents=3,
        env_name="BipedalWalker-v3",
        t=2000
):
    num_gpus = 0
    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=num_gpus)

    # default config
    env_config = {"env_name": env_name, "num_agents": num_agents}
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
        "rollout_fragment_length": 20,
        "train_batch_size": 100,
        "sgd_minibatch_size": 60,
        "num_sgd_iter": 3,
        "num_workers": 1
    }

    if extra_config:
        config.update(extra_config)
    stop = {"timesteps_total": t} if not isinstance(t, dict) else t
    dir_path = tempfile.mkdtemp()
    ret = tune.run(
        DiCETrainer,
        local_dir=dir_path,
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop=stop,
        config=config,
        verbose=2,
        max_failures=0
    )
    shutil.rmtree(dir_path, ignore_errors=True)
    return ret


class DiCETest(unittest.TestCase):
    def test_use_bisector(self):
        _test_dice({USE_BISECTOR: False}, num_agents=2, local_mode=True)

    def test_use_dnv(self):
        _test_dice({DELAY_UPDATE: True}, num_agents=num_agents_pair)

    def test_delay_update(self):
        _test_dice({DELAY_UPDATE: False}, num_agents=3)

    def test_tsc_loss(self):
        _test_dice({TWO_SIDE_CLIP_LOSS: False}, num_agents=num_agents_pair)

    def test_only_tnb(self):
        _test_dice(
            {ONLY_TNB: True}, num_agents=num_agents_pair, local_mode=True
        )

    def test_normalize_adv(self):
        _test_dice({NORMALIZE_ADVANTAGE: True}, num_agents=num_agents_pair)

    def test_default(self):
        _test_dice(num_agents=tune.grid_search([1, 3, 5]))


if __name__ == "__main__":
    # unittest.main(verbosity=2)
    _test_dice(num_agents=5, local_mode=False, t=100000)
