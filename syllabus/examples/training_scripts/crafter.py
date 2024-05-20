import argparse
import os
import random
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO  # Import PPO from stable-baselines3


from syllabus.core import MultiProcessingSyncWrapper, make_multiprocessing_curriculum
from syllabus.curricula import PrioritizedLevelReplay, DomainRandomization, LearningProgressCurriculum, SequentialCurriculum
from syllabus.examples.task_wrappers.crafter_task_wrapper import CrafterTaskWrapper


def parse_args():   
    # Add parser arguments for PPO and LP
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of PPO epochs")
    parser.add_argument("--ppo_clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--lp_alpha", type=float, default=0.1, help="LP alpha parameter")
    # ... (add more arguments as needed)

    args = parser.parse_args()

    return args

# Function to create the environment
def make_env(env_id, seed, curriculum=None):
    def thunk():
        env = gym.make(env_id)
        env.seed(seed)
        if curriculum is not None:
            env = CrafterTaskWrapper(env, seed=seed)
            env = MultiProcessingSyncWrapper(
                env,
                curriculum.get_components(),
                update_on_step=False,
                task_space=env.task_space,
            )
        return env
    return thunk


if __name__ == "__main__":
    args = parse_args()

    env_id = 'Crafter-v0'  # Replace with the actual environment ID
    env = make_env(env_id, args.seed)()

    # Create PPO agent
    ppo_agent = PPO("MlpPolicy", env, verbose=1, n_epochs=args.ppo_epochs, clip_range=args.ppo_clip_range)


    # Training loop
    for i_episode in range(args.num_episodes):
        observation = env.reset()
        for t in range(args.max_timesteps):
            action, _states = ppo_agent.predict(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
