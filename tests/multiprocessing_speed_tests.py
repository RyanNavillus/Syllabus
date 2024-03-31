""" Test curriculum synchronization across multiple processes. """
import ray
from nle.env.tasks import NetHackScore

from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.curricula import NoopCurriculum
from syllabus.tests import create_cartpole_env, create_nethack_env, create_procgen_env, run_native_multiprocess, run_ray_multiprocess
import pytest

N_ENVS = 128
N_EPISODES = 64
N_ROUNDS = 1
N_ITERATIONS = 1

ray.init()
env_fn = create_nethack_env
default_task = NetHackScore
# env_fn = create_cartpole_env
# default_task = (-0.1, 0.1)
# env_fn = create_procgen_env
# default_task = 0
env_args = ()
env_kwargs = {}
sample_env = env_fn(env_args=env_args, env_kwargs=env_kwargs)
# TODO: Test single process speed with Syllabus (with and without step updates)


@pytest.mark.benchmark(group="multiprocessing_speed")
def test_native_speed(benchmark):
    def wrapper():
        return run_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, num_envs=N_ENVS, num_episodes=N_EPISODES)
    
    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)


@pytest.mark.benchmark(group="multiprocessing_speed")
def test_ray_speed(benchmark):
    def wrapper():
        return run_ray_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, num_envs=N_ENVS, num_episodes=N_EPISODES)
    
    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)


@pytest.mark.benchmark(group="multiprocessing_speed")
def test_native_syllabus_speed(benchmark):
    def wrapper():
        curriculum = NoopCurriculum(default_task, sample_env.task_space)
        curriculum = make_multiprocessing_curriculum(curriculum)
        return run_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    
    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)

@pytest.mark.benchmark(group="multiprocessing_speed")
def test_ray_syllabus_speed(benchmark):
    def wrapper():
        curriculum = NoopCurriculum(default_task, sample_env.task_space)
        curriculum = make_ray_curriculum(curriculum)
        return run_ray_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    
    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)

@pytest.mark.benchmark(group="multiprocessing_speed")
def test_native_syllabus_speed_nostep(benchmark):
    def wrapper():
        curriculum = NoopCurriculum(default_task, sample_env.task_space)
        curriculum = make_multiprocessing_curriculum(curriculum)
        return run_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES, update_on_step=False)
    
    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)       

@pytest.mark.benchmark(group="multiprocessing_speed")
def test_ray_syllabus_speed_nostep(benchmark):
    def wrapper():
        curriculum = NoopCurriculum(default_task, sample_env.task_space, random_start_tasks=0)
        curriculum = make_ray_curriculum(curriculum)
        return run_ray_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES, update_on_step=False)
    
    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)
  