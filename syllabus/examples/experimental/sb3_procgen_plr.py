from typing import Callable

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gym as openai_gym
import gymnasium as gym
from gym import spaces
import numpy as np
import procgen  # noqa: F401
from procgen import ProcgenEnv
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import (VecExtractDictObs, DummyVecEnv, VecMonitor,
                                              VecNormalize)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from torch.utils.tensorboard import SummaryWriter

from syllabus.core import (MultiProcessingSyncWrapper,
                           make_multiprocessing_curriculum)
from syllabus.curricula import CentralizedPrioritizedLevelReplay, DomainRandomization
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from syllabus.examples.models import ProcgenAgent
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

# Copy to avoid using the wrong space in the origional class for sb3 verison VecExtractDictObs
class VecExtractDictObs(VecEnvWrapper):

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        assert isinstance(
            venv.observation_space, spaces.Dict
        ), f"VecExtractDictObs can only be used with Dict obs space, not {venv.observation_space}"
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        assert isinstance(obs, dict)
        return obs[self.key]

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, infos = self.venv.step_wait()
        assert isinstance(obs, dict)
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"] = info["terminal_observation"][self.key]
        return obs[self.key], reward, done, infos


def make_env(env_id, seed, curriculum=None, start_level=0, num_levels=1):
    def thunk():
        env = openai_gym.make(f"procgen-{env_id}-v0", distribution_mode="easy", start_level=start_level, num_levels=num_levels)
        env = GymV21CompatibilityV0(env=env)
        components = curriculum.get_components()
        if curriculum is not None:
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
    num_envs=args.num_eval_episodes, env_name=env_name, num_levels=num_levels, start_level=0, distribution_mode="easy", paint_vel_info=False
    )

    eval_envs = VecExtractDictObs(eval_envs, "rgb") 
    eval_envs = wrap_vecenv(eval_envs)

    eval_obs = eval_envs.reset()
    eval_episode_rewards = [-1] * num_episodes

    while -1 in eval_episode_rewards:
        eval_action, _states = model.predict(eval_obs, deterministic=False) 

        eval_obs, rewards, dones, infos = eval_envs.step(eval_action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys() and eval_episode_rewards[i] == -1:
                eval_episode_rewards[i] = info['episode']['r']

    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    return mean_returns, stddev_returns, normalized_mean_returns

def wrap_vecenv(vecenv):
    vecenv = VecMonitor(venv=vecenv, filename=None)
    vecenv = VecNormalize(venv=vecenv, norm_obs=False, norm_reward=True, training=False)
    return vecenv


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, curriculum, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum

    def _on_step(self) -> bool:
        if args.curriculum and args.curriculum_method == "plr":
            tasks = self.training_env.venv.venv.venv.get_attr("task")
            update = {
                "update_type": "on_demand",
                "metrics": {
                    "value": self.locals["values"],
                    "next_value": self.locals["values"],
                    "rew": self.locals["rewards"],
                    "dones": self.locals["dones"],
                    "tasks": tasks,
                },
            }
            self.curriculum.update_curriculum(update)
        return True
    
class Level_Replay_EvalCallback(BaseCallback):

    def __init__(self, env_name, model, eval_freq, num_episodes, num_levels=0, verbose=0):
        super().__init__(verbose)
        self.env_name = env_name
        self.model = model
        self.num_episodes = num_episodes
        self.num_levels = num_levels
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        You must return True to continue training, or False to stop.
        """
        return True  

    def _on_rollout_end(self) -> None:
        if self.num_timesteps % self.eval_freq == 0:
            mean_eval_returns, _, _ = level_replay_evaluate_sb3(self.env_name, self.model, self.num_episodes, self.num_levels)
            writer.add_scalar("test_eval/mean_episode_return", mean_eval_returns, self.num_timesteps)
            

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


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
        if args.curriculum_method == "dr":
            print("Using domain randomization.")
            curriculum = DomainRandomization(sample_env.task_space)
        else:
            raise ValueError(f"Unknown curriculum method {args.curriculum_method}")
        curriculum = make_multiprocessing_curriculum(curriculum)
        del sample_env

    # env setup
    print("Creating env")
    venv = DummyVecEnv(
    [
        make_env(args.env_id,
                args.seed + i,
                curriculum=curriculum if args.curriculum else None,
                num_levels=1 if args.curriculum else 0)
        for i in range(args.num_envs)
    ]
    )
    venv = wrap_vecenv(venv)
    assert isinstance(venv.action_space, gym.spaces.discrete.Discrete), "only discrete action space is supported"

    print("Creating agent")
    agent = ProcgenAgent(
        venv.single_observation_space.shape,
        venv.single_action_space.n,
        arch="large",
        base_kwargs={'recurrent': False, 'hidden_size': 256}
    ).to(device)

    print("Creating model")
    model = PPO(
        "CnnPolicy",
        venv,
        verbose=1,
        n_steps=256,
        learning_rate=linear_schedule(0.0005),
        gamma=0.999,
        gae_lambda=0.95,
        n_epochs=3,
        clip_range_vf=0.2,
        ent_coef=0.01,
        batch_size=256 * 64,
        tensorboard_log="runs/testing"
    )

    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    
    plr_callback = CustomCallback(curriculum)
    level_replay_evaluate_callback = Level_Replay_EvalCallback(args.env_id, model, int(args.num_envs * args.num_steps), args.num_eval_episodes, num_levels=0)

    callback = CallbackList([wandb_callback, plr_callback, level_replay_evaluate_callback])
    model.learn(
        25000000,
        callback=callback,
    )

    

