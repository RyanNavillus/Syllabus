import argparse
import os, sys
import random
import time
from collections import deque
from distutils.util import strtobool

import gym as openai_gym
import gymnasium as gym
import numpy as np
import procgen  # noqa: F401
from procgen import ProcgenEnv
import torch
import torch.nn as nn
import torch.optim as optim
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from torch.utils.tensorboard import SummaryWriter

from syllabus.core import MultiProcessingSyncWrapper, make_multiprocessing_curriculum
from syllabus.curricula import PrioritizedLevelReplay, DomainRandomization, LearningProgressCurriculum, SequentialCurriculum
from syllabus.examples.models import ProcgenAgent
from syllabus.examples.task_wrappers import ProcgenTaskWrapper, MinigridTaskWrapper 
from syllabus.examples.utils.vecenv import VecMonitor, VecNormalize, VecExtractDictObs
sys.path.append("/data/averma/MARL/Syllabus/syllabus/examples/task_wrappers")
from minigrid_task_wrapper_verma import *


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
            env = ProcgenTaskWrapper(env, env_id, seed=seed)
            env = MultiProcessingSyncWrapper(
                env,
                curriculum.get_components(),
                update_on_step=False,
                task_space=env.task_space,
            )
        return env
    return thunk


def wrap_vecenv(vecenv):
    vecenv.is_vector_env = True
    vecenv = VecMonitor(venv=vecenv, filename=None, keep_buf=100)
    vecenv = VecNormalize(venv=vecenv, ob=False, ret=True)
    return vecenv


def slow_level_replay_evaluate(
    env_name,
    policy,
    num_episodes,
    device,
    num_levels=0
):
    policy.eval()

    eval_envs = ProcgenEnv(
        num_envs=1, env_name=env_name, num_levels=num_levels, start_level=0, distribution_mode="easy", paint_vel_info=False
    )
    eval_envs = VecExtractDictObs(eval_envs, "rgb")
    eval_envs = wrap_vecenv(eval_envs)
    eval_obs, _ = eval_envs.reset()
    eval_episode_rewards = []

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            eval_action, _, _, _ = policy.get_action_and_value(torch.Tensor(eval_obs).to(device), deterministic=False)

        eval_obs, _, truncs, terms, infos = eval_envs.step(eval_action.cpu().numpy())
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    policy.train()
    return mean_returns, stddev_returns, normalized_mean_returns


def level_replay_evaluate(
    env_name,
    policy,
    num_episodes,
    device,
    num_levels=0
):
    policy.eval()

    eval_envs = ProcgenEnv(
        num_envs=args.num_eval_episodes, env_name=env_name, num_levels=num_levels, start_level=0, distribution_mode="easy", paint_vel_info=False
    )
    eval_envs = VecExtractDictObs(eval_envs, "rgb")
    eval_envs = wrap_vecenv(eval_envs)
    eval_obs, _ = eval_envs.reset()
    eval_episode_rewards = [-1] * num_episodes

    while -1 in eval_episode_rewards:
        with torch.no_grad():
            eval_action, _, _, _ = policy.get_action_and_value(torch.Tensor(eval_obs).to(device), deterministic=False)

        eval_obs, _, truncs, terms, infos = eval_envs.step(eval_action.cpu().numpy())
        for i, info in enumerate(infos):
            if 'episode' in info.keys() and eval_episode_rewards[i] == -1:
                eval_episode_rewards[i] = info['episode']['r']

    # print(eval_episode_rewards)
    mean_returns = np.mean(eval_episode_rewards)
    stddev_returns = np.std(eval_episode_rewards)
    env_min, env_max = PROCGEN_RETURN_BOUNDS[args.env_id]
    normalized_mean_returns = (mean_returns - env_min) / (env_max - env_min)
    policy.train()
    return mean_returns, stddev_returns, normalized_mean_returns


def make_value_fn():
    def get_value(obs):
        obs = np.array(obs)
        with torch.no_grad():
            return agent.get_value(torch.Tensor(obs).to(device))
    return get_value

def print_values(obj):
    describer = obj.__dict__
    for key in describer.keys():
        print(f"{key}: {describer[key]}")
    print()
    

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device:", device)

    # Curriculum setup
    curriculum = None
    if args.curriculum:
        print("args:\n--------------")
        print(f"{args}\n-------------\n")
        sample_env = openai_gym.make(f"procgen-{args.env_id}-v0")
        sample_env = GymV21CompatibilityV0(env=sample_env)
        procgen_env = ProcgenTaskWrapper(sample_env, args.env_id, seed=args.seed)
        minigrid_env = MinigridTaskWrapperVerma(sample_env, args.env_id, seed=args.seed)
        # print()
        # print("procgen_env attr")
        print_values(procgen_env.env)
        
        # seeds = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(args.num_envs)]
        seeds = [int(s) for s in np.random.choice(10, args.num_envs)]
        print(seeds)

        # print("procgen_env.env attr:")
        # print_values(procgen_env.env)
        # 
        # print("procgen_env.env.gym_env attr:")
        # print_values(procgen_env.env.gym_env)
        
        
