import time
import warnings
from multiprocessing import Process

import gym as openai_gym
import gymnasium as gym
import numpy as np
import ray
import torch
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from syllabus.core import MultiProcessingSyncWrapper, RaySyncWrapper, ReinitTaskWrapper
from syllabus.examples.task_wrappers.cartpole_task_wrapper import CartPoleTaskWrapper
from syllabus.task_space import TaskSpace
from syllabus.tests import SyncTestEnv


def evaluate_random_policy(make_env, num_episodes=100, seeds=None):
    env = make_env(seed=seeds[0] if seeds else None)

    # Seed environment
    env.action_space.seed(0)
    env.observation_space.seed(0)

    episode_returns = []

    for i in range(num_episodes):
        episode_return = 0
        if seeds:
            _ = env.reset(new_task=seeds[i])
            env.action_space.seed(0)
            env.observation_space.seed(0)
        else:
            _ = env.reset()
        term = trunc = False
        while not (term or trunc):
            action = env.action_space.sample()
            _, rew, term, trunc, _ = env.step(action)
            episode_return += rew
        episode_returns.append(episode_return)

    avg_return = sum(episode_returns) / len(episode_returns)
    # print(f"Average Episodic Return: {avg_return}")
    return avg_return, episode_returns


def run_episode(env, new_task=None, curriculum=None, env_id=0):
    """Run a single episode of the environment."""
    if new_task is not None:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    term = trunc = False
    ep_rew = 0
    ep_len = 0
    while not (term or trunc):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        if curriculum and curriculum.requires_step_updates:
            curriculum.update_on_step(obs, rew, term, trunc, info, env_id=env_id)
            curriculum.update_task_progress(env.task_space.encode(env.task), info["task_completion"], env_id=env_id)
        ep_rew += rew
        ep_len += 1
    if curriculum and curriculum.requires_episode_updates:
        curriculum.update_on_episode(ep_rew, ep_len, env.task_space.encode(env.task), env_id=env_id)
    return ep_rew


def run_set_length(env, curriculum=None, episodes=None, steps=None, env_id=0, env_outputs=None):
    """Run environment for a set number of episodes or steps."""
    assert episodes is not None or steps is not None, "Must specify either episodes or steps."
    assert episodes is None or steps is None, "Cannot specify both episodes and steps."
    total_episodes = episodes if episodes is not None else 2**16 - 1
    total_steps = steps if steps is not None else 2**16 - 1
    n_steps = 0
    n_episodes = 0

    # Resume stepping from the last observation.
    if env_outputs is None:
        obs = env.reset(new_task=curriculum.sample()[0] if curriculum else None)

    while n_episodes < total_episodes and n_steps < total_steps:
        term = trunc = False
        ep_rew = 0
        ep_len = 0
        while not (term or trunc) and n_steps < total_steps:
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            if curriculum and curriculum.requires_step_updates:
                curriculum.update_on_step(obs, rew, term, trunc, info, env_id=env_id)
                curriculum.update_task_progress(env.task_space.encode(env.task), info["task_completion"], env_id=env_id)
            ep_rew += rew
            ep_len += 1
            n_steps += 1
        if (term or trunc) and curriculum and curriculum.requires_episode_updates:
            curriculum.update_on_episode(ep_rew, ep_len, env.task_space.encode(env.task), env_id=env_id)
        n_episodes += 1
        obs = env.reset(new_task=curriculum.sample()[0] if curriculum else None)

    return (obs, rew, term, trunc, info)


def run_episodes(env_fn, env_args, env_kwargs, curriculum=None, num_episodes=10, env_id=0):
    """Run multiple episodes of the environment."""
    env = env_fn(env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        if curriculum:
            task = env.task_space.decode(curriculum.sample()[0])
            ep_rews.append(run_episode(env, new_task=task, curriculum=curriculum, env_id=env_id))
        else:
            ep_rews.append(run_episode(env))
    env.close()


def run_episodes_queue(env_fn, env_args, env_kwargs, curriculum_components, sync=True, num_episodes=10, update_on_step=True, buffer_size=2, env_id=0):
    env = env_fn(curriculum_components, env_args=env_args, env_kwargs=env_kwargs, type="queue", update_on_step=update_on_step, buffer_size=buffer_size) if sync else env_fn(env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env, env_id=env_id))
    env.close()


@ray.remote
def run_episodes_ray(env_fn, env_args, env_kwargs, sync=True, num_episodes=10, update_on_step=True):
    env = env_fn(env_args=env_args, env_kwargs=env_kwargs, type="ray", update_on_step=update_on_step) if sync else env_fn(env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env))
    env.close()


def run_single_process(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10):
    start = time.time()
    for _ in range(num_envs):
        run_episodes(env_fn, env_args, env_kwargs, curriculum=curriculum, num_episodes=num_episodes)
    end = time.time()
    native_speed = end - start
    return native_speed


def run_native_multiprocess(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10, update_on_step=True, buffer_size=2):
    start = time.time()
    # Choose multiprocessing and curriculum methods
    if curriculum:
        target = run_episodes_queue
        args = (env_fn, env_args, env_kwargs, curriculum.get_components(), True, num_episodes, update_on_step and curriculum.curriculum.requires_step_updates, buffer_size)
    else:
        target = run_episodes
        args = (env_fn, env_args, env_kwargs, (), num_episodes)

    # Run episodes
    actors = []
    for i in range(num_envs):
        nargs = args + (i,)
        actors.append(Process(target=target, args=nargs))
    for actor in actors:
        actor.start()
    for actor in actors:
        actor.join()
    end = time.time()
    native_speed = end - start

    # Stop curriculum to prevent it from slowing down the next test
    if curriculum:
        curriculum.stop()
    return native_speed


def run_ray_multiprocess(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10, update_on_step=True):
    if curriculum:
        target = run_episodes_ray
        args = (env_fn, env_args, env_kwargs, True, num_episodes, update_on_step)
    else:
        target = run_episodes_ray
        args = (env_fn, env_args, env_kwargs, False, num_episodes, update_on_step)

    start = time.time()
    remotes = []
    for _ in range(num_envs):
        remotes.append(target.remote(*args))
    ray.get(remotes)
    end = time.time()
    ray_speed = end - start
    if curriculum:
        ray.kill(curriculum.curriculum)
    return ray_speed

def get_test_values(x):
    return torch.unsqueeze(torch.Tensor(np.array([0] * len(x))), -1)


# Sync Test Environment
def create_synctest_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    env = SyncTestEnv(*env_args, **env_kwargs)
    if type == "queue":
        env = MultiProcessingSyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    elif type == "ray":
        env = RaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    return env


# Cartpole Tests
def create_cartpole_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    env = gym.make("CartPole-v1", **env_kwargs)
    env = CartPoleTaskWrapper(env)

    if type == "queue":
        env = MultiProcessingSyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    elif type == "ray":
        env = RaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    return env


# Nethack Tests
def create_nethack_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    try:
        from nle.env.tasks import NetHackScore

        from syllabus.examples.task_wrappers.nethack_wrappers import \
            NethackTaskWrapper
    except ImportError:
        warnings.warn("Unable to import nle.")

    env = NetHackScore(*env_args, **env_kwargs)
    env = NethackTaskWrapper(env)

    if type == "queue":
        env = MultiProcessingSyncWrapper(
            env, *args, task_space=env.task_space, **kwargs
        )
    elif type == "ray":
        env = RaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    return env


# Procgen Tests
def create_procgen_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    try:
        import procgen

        from syllabus.examples.task_wrappers.procgen_task_wrapper import \
            ProcgenTaskWrapper
    except ImportError:
        warnings.warn("Unable to import procgen.")

    env = openai_gym.make("procgen-bigfish-v0", *env_args, **env_kwargs)
    env = GymV21CompatibilityV0(env=env)
    env = ProcgenTaskWrapper(env, "bigfish")

    if type == "queue":
        env = MultiProcessingSyncWrapper(
            env, *args, task_space=env.task_space, **kwargs
        )
    elif type == "ray":
        env = RaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    return env


# Minigrid Tests
def create_minigrid_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    try:
        from gym_minigrid.envs import DoorKeyEnv  # noqa: F401
        from gym_minigrid.register import env_list
    except ImportError:
        warnings.warn("Unable to import gym_minigrid.")
    env = gym.make("MiniGrid-DoorKey-5x5-v0", **env_kwargs)

    def create_env(task):
        return gym.make(task)

    task_space = TaskSpace(gym.spaces.Discrete(len(env_list)), env_list)
    env = ReinitTaskWrapper(env, create_env, task_space=task_space)
    if type == "queue":
        env = MultiProcessingSyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    elif type == "ray":
        env = RaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    return env
