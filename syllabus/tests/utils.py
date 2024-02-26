import time
import warnings
from multiprocessing import Process

import gymnasium as gym
import ray
from pettingzoo.utils.env import ParallelEnv

from syllabus.core import MultiProcessingSyncWrapper, PettingZooMultiProcessingSyncWrapper, RaySyncWrapper, PettingZooRaySyncWrapper, ReinitTaskWrapper
from syllabus.examples.task_wrappers.cartpole_task_wrapper import CartPoleTaskWrapper
from syllabus.task_space import TaskSpace
from syllabus.tests import SyncTestEnv, PettingZooSyncTestEnv


def evaluate_random_policy(make_env, num_episodes=100, seeds=None):
    env = make_env()
    if isinstance(env, ParallelEnv):
        return evaluate_random_policy_pettingzoo(make_env, seeds=seeds, num_episodes=num_episodes)
    else:
        return evaluate_random_policy_gymnasium(make_env, seeds=seeds, num_episodes=num_episodes)


def evaluate_random_policy_gymnasium(make_env, num_episodes=100, seeds=None):
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


def evaluate_random_policy_pettingzoo(make_env, num_episodes=100, seeds=None):
    env = make_env(seed=seeds[0] if seeds else None)

    # Seed environment
    for agent in env.possible_agents:
        env.action_space(agent).seed(0)
        env.observation_space(agent).seed(0)

    episode_returns = []

    for i in range(num_episodes):
        episode_return = 0
        if seeds:
            _ = env.reset(new_task=seeds[i])
            for agent in env.possible_agents:
                env.action_space(agent).seed(0)
                env.observation_space(agent).seed(0)
        else:
            _ = env.reset()
        term = trunc = {agent: False for agent in env.agents}
        while not (all(term.values()) or all(trunc.values())):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _, rew, term, trunc, _ = env.step(actions)
            episode_return += sum(rew.values())
        episode_returns.append(episode_return)

    avg_return = sum(episode_returns) / len(episode_returns)
    # print(f"Average Episodic Return: {avg_return}")
    return avg_return, episode_returns


def run_pettingzoo_episode(env, new_task=None, curriculum=None):
    """Run a single episode of the environment."""
    if new_task:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    term = trunc = False
    ep_rew = 0
    while env.agents:
        action = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rew, term, trunc, info = env.step(action)
        if curriculum and curriculum.__class__.REQUIRES_STEP_UPDATES:
            curriculum.update_on_step(obs, sum(rew.values()), all(term.values()), all(trunc.values()), info)
        ep_rew += sum(rew.values())
    if curriculum and len(info.values()) > 0 and "task_completion" in list(info.values())[0]:
        task_progress = max([i["task_completion"] for i in info.values()])
        curriculum.update_task_progress(env.task, task_progress)
    return ep_rew


def run_gymnasium_episode(env, new_task=None, curriculum=None):
    """Run a single episode of the environment."""
    if new_task:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    term = trunc = False
    ep_rew = 0
    while not (term or trunc):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        if curriculum and curriculum.__class__.REQUIRES_STEP_UPDATES:
            curriculum.update_on_step(obs, rew, term, trunc, info)
        ep_rew += rew
    if curriculum and "task_completion" in info:
        curriculum.update_task_progress(env.task, info["task_completion"])
    return ep_rew


def run_episode(env, new_task=None, curriculum=None):
    if isinstance(env, ParallelEnv):
        return run_pettingzoo_episode(env, new_task, curriculum)
    else:
        return run_gymnasium_episode(env, new_task, curriculum)


def run_episodes(env_fn, env_args, env_kwargs, curriculum=None, num_episodes=10):
    """Run multiple episodes of the environment."""
    env = env_fn(env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        if curriculum:
            task = curriculum.sample()[0]
            rews = run_episode(env, new_task=task, curriculum=curriculum)
        else:
            rews = run_episode(env)
        ep_rews.append(rews)


def run_episodes_queue(env_fn, env_args, env_kwargs, task_queue, update_queue, sync=True, num_episodes=10, update_on_step=True):
    env = env_fn(task_queue, update_queue, env_args=env_args, env_kwargs=env_kwargs, type="queue", update_on_step=update_on_step) if sync else env_fn(env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env))


@ray.remote
def run_episodes_ray(env_fn, env_args, env_kwargs, sync=True, num_episodes=10, update_on_step=True):
    env = env_fn(env_args=env_args, env_kwargs=env_kwargs, type="ray", update_on_step=update_on_step) if sync else env_fn(env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env))


def test_single_process(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10):
    start = time.time()
    for _ in range(num_envs):
        run_episodes(env_fn, env_args, env_kwargs, curriculum=curriculum, num_episodes=num_episodes)
    end = time.time()
    native_speed = end - start
    return native_speed


def test_native_multiprocess(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10, update_on_step=True):
    start = time.time()

    # Choose multiprocessing and curriculum methods
    if curriculum:
        target = run_episodes_queue
        args = (env_fn, env_args, env_kwargs, curriculum.task_queue, curriculum.update_queue, True, num_episodes, update_on_step and curriculum.curriculum.__class__.REQUIRES_STEP_UPDATES)
    else:
        target = run_episodes
        args = (env_fn, env_args, env_kwargs, (), num_episodes)

    # Run episodes
    actors = []
    for _ in range(num_envs):
        actors.append(Process(target=target, args=args))
    for actor in actors:
        actor.start()
    for actor in actors:
        actor.join()

    end = time.time()
    native_speed = end - start
    time.sleep(3.0)
    return native_speed


def test_ray_multiprocess(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10, update_on_step=True):
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
    return ray_speed


# Sync Test Environments
def create_synctest_gymnasium_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    env = SyncTestEnv(*env_args, **env_kwargs)
    if type == "queue":
        env = MultiProcessingSyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    elif type == "ray":
        env = RaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    return env


def create_pettingzoo_synctest_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    env = PettingZooSyncTestEnv(*env_args, **env_kwargs)
    if type == "queue":
        env = PettingZooMultiProcessingSyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    elif type == "ray":
        env = PettingZooRaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
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
    from nle.env.tasks import NetHackScore
    from syllabus.examples.task_wrappers.nethack_wrappers import NethackTaskWrapper

    env = NetHackScore(*env_args, **env_kwargs)
    env = NethackTaskWrapper(env)

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


# Pistonball Tests
def create_pistonball_env(*args, type=None, env_args=(), env_kwargs={}, **kwargs):
    from pettingzoo.butterfly import pistonball_v6  # noqa: F401

    from syllabus.examples.task_wrappers import PistonballTaskWrapper
    env = pistonball_v6.parallel_env()

    def create_env(task):
        return pistonball_v6.parallel_env(n_pistons=task)

    env = PistonballTaskWrapper(env)
    if type == "queue":
        env = PettingZooMultiProcessingSyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    elif type == "ray":
        env = PettingZooRaySyncWrapper(env, *args, task_space=env.task_space, **kwargs)
    return env
