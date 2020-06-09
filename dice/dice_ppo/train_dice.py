from dice.dice_ppo.dice_ppo import DiCETrainer
from dice.utils import *

os.environ['OMP_NUM_THREADS'] = '1'


def train(
        extra_config,
        env_name,
        stop,
        exp_name,
        num_agents,
        num_seeds,
        num_gpus,
        test_mode=False,
        **kwargs
):
    initialize_ray(test_mode=test_mode, local_mode=False, num_gpus=num_gpus)
    config = {
        "seed": tune.grid_search([i * 100 for i in range(num_seeds)]),
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "num_agents": num_agents
        },
        "log_level": "DEBUG" if test_mode else "INFO"
    }
    if extra_config:
        config.update(extra_config)

    analysis = tune.run(
        DiCETrainer,
        name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=10,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_at_end=True,
        stop={"timesteps_total": stop} if isinstance(stop, int) else stop,
        config=config,
        max_failures=20,
        reuse_actors=False,
        **kwargs
    )

    path = "{}-{}-{}ts-{}agents.pkl".format(
        exp_name, env_name, stop, num_agents
    )
    with open(path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(path))

    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--stop", type=float, default=5e6)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    env_name = args.env_name
    exp_name = "{}-{}".format(args.exp_name, env_name)
    stop = int(args.stop)
    large = env_name in ["Walker2d-v3", "Hopper-v3"]
    if args.test:
        stop = 5000
        exp_name = "DELETE-ME-TEST-EXP"

    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "num_agents": tune.grid_search([args.num_agents])
        },
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0001,
        'sample_batch_size': 200 if large else 50,
        'sgd_minibatch_size': 100 if large else 64,
        'train_batch_size': 10000 if large else 2048,
        "num_gpus": 0.4 if args.num_gpus != 0 else 0,
        "num_cpus_per_worker": 0.4,
        "num_cpus_for_driver": 0.45,
        "num_envs_per_worker": 8 if large else 5,
        'num_workers': 8 if large else 1,
    }

    train(
        extra_config=config,
        env_name=config['env_config']['env_name'],
        stop=stop,
        exp_name=exp_name,
        num_agents=config['env_config']['num_agents'],
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        verbose=1
    )
