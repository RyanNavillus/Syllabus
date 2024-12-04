import time
import warnings
from multiprocessing import Process

import gym as openai_gym
import gymnasium as gym
import numpy as np
import ray
import torch
from pettingzoo.utils.env import ParallelEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from syllabus.core import (GymnasiumSyncWrapper,
                           PettingZooSyncWrapper,
                           RayPettingZooSyncWrapper,
                           PettingZooReinitTaskWrapper, RayGymnasiumSyncWrapper,
                           ReinitTaskWrapper)
from syllabus.examples.task_wrappers.cartpole_task_wrapper import CartPoleTaskWrapper
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace
from syllabus.tests import PettingZooSyncTestEnv, SyncTestEnv


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
            env.seed(seed=seeds[i])
            env.reset()
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
            env.seed(seed=seeds[i])
            env.reset()
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


def run_pettingzoo_episode(env, new_task=None, curriculum=None, env_id=0):
    """Run a single episode of the environment."""
    if new_task is not None:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    term = trunc = False
    ep_rew = {agent: 0 for agent in env.agents}
    steps = 0
    while env.agents:
        action = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rews, term, trunc, info = env.step(action)
        steps += 1
        if curriculum and curriculum.requires_step_updates:
            task_completion = max([i["task_completion"] for i in info.values()]) if len(
                env.agents) > 0 and "task_completion" in info[env.agents[0]] else 0.0
            curriculum.update_on_step(env.task_space.encode(env.task), obs, list(rews.values()),
                                      list(term.values()), list(trunc.values()), info, task_completion, env_id=env_id)
            curriculum.update_task_progress(env.task_space.encode(env.task), task_completion, env_id=env_id)
        for agent, rew in rews.items():
            ep_rew[agent] += rew
        time.sleep(0)
    if curriculum:
        curriculum.update_on_episode(ep_rew, steps, env.task_space.encode(env.task), 0.0, env_id=env_id)
    return ep_rew


def run_gymnasium_episode(env, new_task=None, curriculum=None, env_id=0):
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
            task_completion = info["task_completion"] if "task_completion" in info else 0.0
            curriculum.update_on_step(env.task_space.encode(env.task), obs, rew, term,
                                      trunc, info, task_completion, env_id=env_id)
            curriculum.update_task_progress(env.task_space.encode(env.task), task_completion, env_id=env_id)
        ep_rew += rew
        ep_len += 1
        time.sleep(0)
    if curriculum:
        curriculum.update_on_episode(ep_rew, ep_len, env.task_space.encode(env.task), 0.0, env_id=env_id)
    return ep_rew


def run_episode(env, new_task=None, curriculum=None, env_id=0):
    if isinstance(env, ParallelEnv):
        return run_pettingzoo_episode(env, new_task, curriculum, env_id=env_id)
    else:
        return run_gymnasium_episode(env, new_task, curriculum, env_id=env_id)


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
                curriculum.update_on_step(env.task_space.encode(env.task), obs, rew, term, trunc, info, env_id=env_id)
                curriculum.update_task_progress(env.task_space.encode(env.task), info["task_completion"], env_id=env_id)
            ep_rew += rew
            ep_len += 1
            n_steps += 1
        if (term or trunc) and curriculum:
            curriculum.update_on_episode(ep_rew, ep_len, env.task_space.encode(env.task), 0.0, env_id=env_id)
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
            rews = run_episode(env, new_task=task, curriculum=curriculum, env_id=env_id)
        else:
            rews = run_episode(env)
        ep_rews.append(rews)
    env.close()


def run_episodes_queue(env_fn, env_args, env_kwargs, curriculum_components, sync=True, num_episodes=10, buffer_size=1, env_id=0):
    env = env_fn(curriculum_components, env_args=env_args, env_kwargs=env_kwargs, sync_type="queue",
                 buffer_size=buffer_size, batch_size=2) if sync else env_fn(env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env, env_id=env_id))
    time.sleep(3)


@ray.remote
def run_episodes_ray(env_fn, env_args, env_kwargs, sync=True, num_episodes=10):
    env = env_fn(env_args=env_args, env_kwargs=env_kwargs, sync_type="ray") if sync else env_fn(
        env_args=env_args, env_kwargs=env_kwargs)
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env))
    env.close()


def run_single_process(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10):
    start = time.time()
    for _ in range(num_episodes):
        # Interleave episodes for each environment
        for env_idx in range(num_envs):
            run_episodes(env_fn, env_args, env_kwargs, curriculum=curriculum, num_episodes=1, env_id=env_idx)
    end = time.time()
    native_speed = end - start
    return native_speed


def run_native_multiprocess(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10, buffer_size=2):
    start = time.time()
    # Choose multiprocessing and curriculum methods
    if curriculum:
        target = run_episodes_queue
        args = (env_fn, env_args, env_kwargs, curriculum.components, True, num_episodes, buffer_size)
    else:
        target = run_episodes
        args = (env_fn, env_args, env_kwargs, (), num_episodes)
    if curriculum is not None:
        curriculum.start()
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


def run_ray_multiprocess(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10):
    if curriculum:
        target = run_episodes_ray
        args = (env_fn, env_args, env_kwargs, True, num_episodes)
    else:
        target = run_episodes_ray
        args = (env_fn, env_args, env_kwargs, False, num_episodes)

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


def run_native_vecenv(env_fn, env_args=(), env_kwargs={}, curriculum=None, num_envs=2, num_episodes=10):
    sample_env = env_fn(env_args=env_args, env_kwargs=env_kwargs)
    sample_env.reset()
    envs = gym.vector.AsyncVectorEnv(
        [env_fn(curriculum.components, env_args=env_args, sync_type="queue" if curriculum else None, env_kwargs=env_kwargs, wrap=True)
         for _ in range(num_envs)]
    )
    envs.reset()

    eval_episode_rewards = []
    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            eval_action = np.array([sample_env.action_space.sample() for _ in range(num_envs)])
        _, _, _, _, infos = envs.step(eval_action)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    eval_episode_rewards.append(info['episode']['r'])
    envs.close()


def get_test_values(x):
    return torch.unsqueeze(torch.Tensor(np.array([0] * len(x))), -1)


def get_test_actions(x):
    return torch.IntTensor(np.array([0] * len(x)))


class ExtractDictObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Extract space from Dict observation space by the key."""

    def __init__(self, env: gym.Env, filter_key: str = None):
        gym.utils.RecordConstructorArgs.__init__(self, filter_key=filter_key)
        gym.ObservationWrapper.__init__(self, env)

        wrapped_observation_space = env.observation_space
        if not isinstance(wrapped_observation_space, gym.spaces.Dict):
            raise ValueError(
                f"FilterObservationWrapper is only usable with dict observations, "
                f"environment observation space is {type(wrapped_observation_space)}"
            )

        self.observation_space = wrapped_observation_space.spaces[filter_key]
        self._env = env
        self.filter_key = filter_key

    def observation(self, observation):
        filter_observation = self._filter_observation(observation)
        return filter_observation

    def _filter_observation(self, observation):
        try:
            return observation[self.filter_key]
        except KeyError as e:
            raise KeyError(
                f"Key {self.filter_key} not found in observation. "
                f"Available keys are {observation.keys()}"
            ) from e

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self.observation(obs), info


# Sync Test Environments
def create_gymnasium_synctest_env(*args, sync_type=None, env_args=(), env_kwargs={}, **kwargs):
    env = SyncTestEnv(*env_args, **env_kwargs)
    if sync_type == "queue":
        env = GymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
    elif sync_type == "ray":
        env = RayGymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
    return env


def create_pettingzoo_synctest_env(*args, sync_type=None, env_args=(), env_kwargs={}, **kwargs):
    env = PettingZooSyncTestEnv(*env_args, **env_kwargs)
    if sync_type == "queue":
        env = PettingZooSyncWrapper(env, env.task_space, *args, **kwargs)
    elif sync_type == "ray":
        env = RayPettingZooSyncWrapper(env, env.task_space, *args, **kwargs)
    return env


# Cartpole Tests
def create_cartpole_env(*args, sync_type=None, env_args=(), env_kwargs={}, wrap=False, **kwargs):
    def thunk():
        env = gym.make("CartPole-v1", **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CartPoleTaskWrapper(env, discretize=False)

        if sync_type == "queue":
            env = GymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
        elif sync_type == "ray":
            env = RayGymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
        return env
    return thunk if wrap else thunk()


# Nethack Tests
def create_nethack_env(*args, sync_type=None, env_args=(), env_kwargs={}, wrap=False, **kwargs):
    from nle.env.tasks import NetHackScore

    from syllabus.examples.task_wrappers.nethack_wrappers import NethackSeedWrapper

    def thunk():
        env = NetHackScore(*env_args, **env_kwargs)
        env = ExtractDictObservation(env, filter_key="blstats")

        # env = GymV21CompatibilityV0(env=env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NethackSeedWrapper(env)

        if sync_type == "queue":
            env = GymnasiumSyncWrapper(
                env, env.task_space, *args, **kwargs
            )
        elif sync_type == "ray":
            env = RayGymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
        return env
    return thunk if wrap else thunk()


# Procgen Tests
def create_procgen_env(*args, sync_type=None, env_args=(), env_kwargs={}, wrap=False, **kwargs):
    try:
        import procgen

        from syllabus.examples.task_wrappers.procgen_task_wrapper import \
            ProcgenTaskWrapper
    except ImportError:
        warnings.warn("Unable to import procgen.", stacklevel=2)

    def thunk():
        env = openai_gym.make("procgen-bigfish-v0", *env_args, **env_kwargs)
        env = GymV21CompatibilityV0(env=env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = ProcgenTaskWrapper(env, "bigfish")

        if sync_type == "queue":
            env = GymnasiumSyncWrapper(
                env, env.task_space, *args, **kwargs
            )
        elif sync_type == "ray":
            env = RayGymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
        return env
    return thunk if wrap else thunk()


# Minigrid Tests
def create_minigrid_env(*args, sync_type=None, env_args=(), env_kwargs={}, **kwargs):
    try:
        from gym_minigrid.envs import DoorKeyEnv  # noqa: F401
        from gym_minigrid.register import env_list
    except ImportError:
        warnings.warn("Unable to import gym_minigrid.", stacklevel=2)
    env = gym.make("MiniGrid-DoorKey-5x5-v0", **env_kwargs)

    def create_env(task):
        return gym.make(task)

    task_space = DiscreteTaskSpace(gym.spaces.Discrete(len(env_list)), env_list)
    env = ReinitTaskWrapper(env, create_env, task_space=task_space)
    if sync_type == "queue":
        env = GymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
    elif sync_type == "ray":
        env = RayGymnasiumSyncWrapper(env, env.task_space, *args, **kwargs)
    return env


# Pistonball Tests
def create_pistonball_env(*args, sync_type=None, env_args=(), env_kwargs={}, **kwargs):
    try:
        from pettingzoo.butterfly import pistonball_v6  # noqa: F401

        from syllabus.examples.task_wrappers import PistonballTaskWrapper
    except ImportError:
        warnings.warn("Unable to import pistonball from pettingzoo.", stacklevel=2)

    env = pistonball_v6.parallel_env()
    env = PistonballTaskWrapper(env)
    if sync_type == "queue":
        env = PettingZooSyncWrapper(env, env.task_space, *args, **kwargs)
    elif sync_type == "ray":
        env = RayPettingZooSyncWrapper(env, env.task_space, *args, **kwargs)
    return env


# Simple Tag Tests
def create_simpletag_env(*args, sync_type=None, env_args=(), env_kwargs={}, **kwargs):
    try:
        from pettingzoo.mpe import simple_tag_v3  # noqa: F401

        # from syllabus.examples.task_wrappers import SimpleTagTaskWrapper
    except ImportError:
        warnings.warn("Unable to import simple tag from pettingzoo.", stacklevel=2)

    def create_env(task):
        good, adversary, obstacle = task
        return simple_tag_v3.parallel_env(num_good=good, num_adversaries=adversary, num_obstacles=obstacle, continuous_actions=False)

    task_space = MultiDiscreteTaskSpace(gym.spaces.MultiDiscrete([1, 1, 1]), [[4], [4], [4]])
    env = simple_tag_v3.parallel_env()
    env = PettingZooReinitTaskWrapper(env, create_env, task_space)

    # Set largest posiible task
    env.reset(new_task=(4, 4, 4))

    if sync_type == "queue":
        env = PettingZooSyncWrapper(env, env.task_space, *args, **kwargs)
    elif sync_type == "ray":
        env = RayPettingZooSyncWrapper(env, env.task_space, *args, **kwargs)
    return env
