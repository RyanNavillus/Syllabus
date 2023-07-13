import gym
import random
import ray
import time
from multiprocessing import Process
from syllabus.core import MultiProcessingSyncWrapper, RaySyncWrapper
from syllabus.task_space import TaskSpace
def run_episode(env, new_task=None, curriculum=None):
    """Run a single episode of the environment."""
    if new_task:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    done = False
    ep_rew = 0
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if curriculum and curriculum.__class__.REQUIRES_STEP_UPDATES:
            curriculum._on_step(obs, rew, done, info)
        ep_rew += rew
    return ep_rew


def run_episodes(env_fn, curriculum=None, num_episodes=10):
    """Run multiple episodes of the environment."""
    env = env_fn()
    ep_rews = []
    for _ in range(num_episodes):
        if curriculum:
            task = curriculum.sample()[0]
            ep_rews.append(run_episode(env, new_task=task, curriculum=curriculum))
            curriculum._complete_task(task, success_prob=random.random())
        else:
            ep_rews.append(run_episode(env))


def run_episodes_queue(env_fn, task_queue, update_queue, sync=True, num_episodes=10, update_on_step=True):
    env = env_fn(task_queue, update_queue, type="queue", update_on_step=update_on_step) if sync else env_fn()
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env))

@ray.remote
def run_episodes_ray(env_fn, sync=True, num_episodes=10, update_on_step=True):
    env = env_fn(type="ray", update_on_step=update_on_step) if sync else env_fn()
    ep_rews = []
    for _ in range(num_episodes):
        ep_rews.append(run_episode(env))


def test_single_process(env_fn, curriculum=None, num_envs=2, num_episodes=10):
    start = time.time()
    for _ in range(num_envs):
        run_episodes(env_fn, curriculum=curriculum, num_episodes=num_episodes)
    end = time.time()
    native_speed = end - start
    return native_speed


def test_native_multiprocess(env_fn, curriculum=None, num_envs=2, num_episodes=10, update_on_step=True):
    start = time.time()

    # Choose multiprocessing and curriculum methods
    if curriculum:
        target = run_episodes_queue
        args = (env_fn, curriculum.task_queue, curriculum.update_queue, True, num_episodes, update_on_step and curriculum.curriculum.__class__.REQUIRES_STEP_UPDATES)
    else:
        target = run_episodes
        args = (env_fn, None, num_episodes)

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
    return native_speed


def test_ray_multiprocess(env_fn, curriculum=None, num_envs=2, num_episodes=10, update_on_step=True):
    if curriculum:
        target = run_episodes_ray
        args = (env_fn, True, num_episodes, update_on_step and curriculum.curriculum.__class__.REQUIRES_STEP_UPDATES)
    else:
        target = run_episodes_ray
        args = (env_fn, False, num_episodes, update_on_step)

    start = time.time()
    remotes = []
    for _ in range(num_envs):
        remotes.append(target.remote(*args))
    ray.get(remotes)
    end = time.time()
    ray_speed = end - start
    return ray_speed


# Nethack Tests
from nle.env.tasks import NetHackScore
from syllabus.examples.task_wrappers.nethack_task_wrapper import NethackTaskWrapper

def create_nethack_env(*args, type=None, **kwargs):
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    if type == "queue":
        env = MultiProcessingSyncWrapper(env,
                                        *args,
                                        default_task=NetHackScore,
                                        task_space=env.task_space,
                                        **kwargs)
    elif type == "ray":
        env = RaySyncWrapper(env, *args, default_task=NetHackScore, task_space=env.task_space, **kwargs)
    return env


# Minigrid Tests
from gym_minigrid.envs import DoorKeyEnv
from syllabus.core import ReinitTaskWrapper
from gym_minigrid.register import env_list

def create_minigrid_env(*args, type=None, **kwargs):
    env = gym.make("MiniGrid-DoorKey-5x5-v0")

    def create_env(task):
        return gym.make(task)

    task_space = TaskSpace(gym.spaces.Discrete(len(env_list)), env_list)
    env = ReinitTaskWrapper(env, create_env, task_space=task_space)
    if type == "queue":
        env = MultiProcessingSyncWrapper(env,
                                        *args,
                                        default_task="MiniGrid-DoorKey-5x5-v0",
                                        task_space=env.task_space,
                                        **kwargs)
    elif type == "ray":
        env = RaySyncWrapper(env, *args, default_task="MiniGrid-DoorKey-5x5-v0", task_space=env.task_space, **kwargs)
    return env