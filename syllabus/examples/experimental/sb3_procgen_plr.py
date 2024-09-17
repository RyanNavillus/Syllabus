import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Callable

import gym as openai_gym
import gymnasium as gym
import numpy as np
import procgen  # noqa: F401
import torch
import wandb
from procgen import ProcgenEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from syllabus.core import MultiProcessingSyncWrapper, make_multiprocessing_curriculum
from syllabus.curricula import CentralizedPrioritizedLevelReplay, DomainRandomization
from syllabus.examples.models import SB3ResNetBase, Sb3ProcgenAgent
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.examples.utils.vecenv import VecMonitor as SyllabusVecMonitor, VecNormalize as SyllabusVecNormalize, VecExtractDictObs as SyllabusVecExtractDictObs

from torch.utils.tensorboard import SummaryWriter
from wandb.integration.sb3 import WandbCallback


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="syllabus",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--logging-dir", type=str, default=".",
                        help="the base directory for logging and wandb storage.")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="starpilot",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(25e6),
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    # Procgen arguments
    parser.add_argument("--full-dist", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Train on full distribution of levels.")

    # Curriculum arguments
    parser.add_argument("--curriculum", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will use curriculum learning")
    parser.add_argument("--curriculum-method", type=str, default="plr",
                        help="curriculum method to use")
    parser.add_argument("--num-eval-episodes", type=int, default=10,
                        help="the number of episodes to evaluate the agent on after each policy update.")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


PROCGEN_RETURN_BOUNDS = {
    "coinrun": (5, 10),
    "starpilot": (2.5, 64),
    "caveflyer": (3.5, 12),
    "dodgeball": (1.5, 19),
    "fruitbot": (-1.5, 32.4),
    "chaser": (0.5, 13),
    "miner": (1.5, 13),
    "jumper": (3, 10),
    "leaper": (3, 10),
    "maze": (5, 10),
    "bigfish": (1, 40),
    "heist": (3.5, 10),
    "climber": (2, 12.6),
    "plunder": (4.5, 30),
    "ninja": (3.5, 10),
    "bossfight": (0.5, 13),
}

def make_env(env_id, seed, curriculum=None, start_level=0, num_levels=1):
    def thunk():
        env = openai_gym.make(f"procgen-{env_id}-v0", distribution_mode="easy", start_level=start_level, num_levels=num_levels)
        env = GymV21CompatibilityV0(env=env)
        if curriculum is not None:
            components = curriculum.get_components()  # This must be safe to call here
            env = ProcgenTaskWrapper(env, env_id, seed=seed)
            env = MultiProcessingSyncWrapper(
                env=env,
                components=components,
                update_on_step=False,
                task_space=env.task_space,
            )
        return env
    return thunk


def level_replay_evaluate_sb3(env_name, model, num_episodes, num_levels=0):
    eval_envs = ProcgenEnv(
        num_envs=args.num_eval_episodes,
        env_name=env_name,
        num_levels=num_levels,
        start_level=0,
        distribution_mode="easy",
        paint_vel_info=False
    )
    model.policy.set_training_mode(False)
    eval_envs = SyllabusVecExtractDictObs(eval_envs, "rgb")
    eval_envs = SyllabusVecMonitor(venv=eval_envs, filename=None, keep_buf=100)
    eval_envs = SyllabusVecNormalize(venv=eval_envs, ob=False, ret=True)
    eval_obs, _ = eval_envs.reset()
    eval_episode_rewards = [-1] * num_episodes

    while -1 in eval_episode_rewards:
        with torch.no_grad():
            eval_action, _states = model.predict(eval_obs, deterministic=False)
        eval_obs, _, _, _, infos = eval_envs.step(eval_action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys() and eval_episode_rewards[i] == -1:
                eval_episode_rewards[i] = info['episode']['r']

    model.policy.set_training_mode(True)
    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    return mean_returns, stddev_returns, normalized_mean_returns


def wrap_vecenv(vecenv):
    vecenv = VecMonitor(venv=vecenv, filename=None)
    vecenv = VecNormalize(venv=vecenv, norm_obs=False, norm_reward=True, training=True)
    return vecenv


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, curriculum, model, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.model = model

    def _on_step(self) -> bool:
        if self.curriculum is not None and type(self.curriculum.curriculum) is CentralizedPrioritizedLevelReplay:
            self.update_locals(self.locals)
            tasks = [i["task"] for i in self.locals["infos"]]
            obs = self.locals['new_obs']
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)
            with torch.no_grad():
                new_value = self.model.policy.predict_values(obs_tensor)
            update = {
                "update_type": "on_demand",
                "metrics": {
                    "value": self.locals["values"],
                    "next_value": new_value,
                    "rew": self.locals["rewards"],
                    "dones": self.locals["dones"],
                    "tasks": tasks,
                },
            }
            self.curriculum.update(update)
            del obs_tensor
            del obs
        return True

    def _on_rollout_end(self) -> None:
        mean_eval_returns, stddev_eval_returns, normalized_mean_eval_returns = level_replay_evaluate_sb3(args.env_id, self.model, args.num_eval_episodes, num_levels=0)
        mean_train_returns, stddev_train_returns, normalized_mean_train_returns = level_replay_evaluate_sb3(args.env_id, self.model, args.num_eval_episodes, num_levels=200)
        writer.add_scalar("test_eval/mean_episode_return", mean_eval_returns, self.num_timesteps)
        writer.add_scalar("test_eval/normalized_mean_eval_return", normalized_mean_eval_returns, self.num_timesteps)
        writer.add_scalar("test_eval/stddev_eval_return", stddev_eval_returns, self.num_timesteps)
        writer.add_scalar("train_eval/mean_episode_return", mean_train_returns, self.num_timesteps)
        writer.add_scalar("train_eval/normalized_mean_train_return", normalized_mean_train_returns, self.num_timesteps)
        writer.add_scalar("train_eval/stddev_train_return", stddev_train_returns, self.num_timesteps)
        if self.curriculum is not None:
            self.curriculum.log_metrics(writer, step=self.num_timesteps)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=args.logging_dir,
        )

    writer = SummaryWriter(os.path.join(args.logging_dir, "./runs/{run_name}"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device:", device)

    # Curriculum setup
    curriculum = None
    if args.curriculum:
        sample_env = openai_gym.make(f"procgen-{args.env_id}-v0")
        sample_env = GymV21CompatibilityV0(env=sample_env)
        sample_env = ProcgenTaskWrapper(sample_env, args.env_id, seed=args.seed)

        # Intialize Curriculum Method
        if args.curriculum_method == "plr":
            print("Using prioritized level replay.")
            curriculum = CentralizedPrioritizedLevelReplay(
                sample_env.task_space,
                num_steps=args.num_steps,
                num_processes=args.num_envs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                task_sampler_kwargs_dict={"strategy": "value_l1"}
            )
        elif args.curriculum_method == "dr":
            print("Using domain randomization.")
            curriculum = DomainRandomization(sample_env.task_space)
        else:
            raise ValueError(f"Unknown curriculum method {args.curriculum_method}")
        curriculum = make_multiprocessing_curriculum(curriculum)
        del sample_env


    # env setup
    print("Creating env")
    venv_fn = [
        make_env(
            args.env_id,
            args.seed + i,
            curriculum=curriculum if args.curriculum else None,
            num_levels=1 if args.curriculum else 0
        )
        for i in range(args.num_envs)
    ]
    venv = DummyVecEnv(venv_fn)
    venv = wrap_vecenv(venv)
    assert isinstance(venv.action_space, gym.spaces.discrete.Discrete), "only discrete action space is supported"

    print("Creating model")
    model = PPO(
        Sb3ProcgenAgent,
        venv,
        verbose=1,
        n_steps=256,
        learning_rate=0.0005,
        gamma=0.999,
        gae_lambda=0.95,
        n_epochs=3,
        clip_range_vf=0.2,
        ent_coef=0.01,
        batch_size=2048,
        policy_kwargs={
            "hidden_size": 256,
            "features_extractor_class": SB3ResNetBase,
            "features_extractor_kwargs": {'num_inputs': 3, "features_dim": 256},
            "normalize_images": False
        }
    )

    plr_callback = CustomCallback(curriculum, model, venv)

    if args.track:
        wandb_callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
        callback = CallbackList([wandb_callback, plr_callback])
    else:
        callback = plr_callback

    model.learn(
        25000000,
        callback=callback,
    )
