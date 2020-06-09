from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.sac.sac import DEFAULT_CONFIG as sac_default_config
from ray.tune.utils.util import merge_dicts

from dice import utils as constants


class DiCESACCallbacks(DefaultCallbacks):
    def on_postprocess_trajectory(self, *args, **kwargs):
        constants.on_postprocess_trajectory(*args, **kwargs)


USE_MY_TARGET_DIVERSITY = "use_my_target_diversity"
SHARE_BUFFER = "share_buffer"

dice_sac_default_config = merge_dicts(
    sac_default_config,
    {
        # New version of SAC require this
        "postprocess_inputs": True,
        "use_exec_api": False,
        "diversity_twin_q": False,
        "grad_clip": 40.0,
        constants.USE_BISECTOR: True,
        constants.DELAY_UPDATE: True,
        USE_MY_TARGET_DIVERSITY: False,
        SHARE_BUFFER: False,
        constants.ONLY_TNB: False,
        "normalize_actions": False,
        "env_config": {
            "normalize_actions": False
        },

        # "tau": 5e-3,  # <<== SAC already have this
        "callbacks": DiCESACCallbacks
    }
)
