""" Test curriculum synchronization across multiple processes. """
from nle.env.tasks import NetHackScore

from syllabus.core import make_multiprocessing_curriculum
from syllabus.curricula import Constant
from syllabus.tests import create_cartpole_env, create_nethack_env, create_procgen_env, run_native_vecenv, run_native_multiprocess
import pytest

N_ENVS = 128
N_EPISODES = 64
N_ROUNDS = 1
N_ITERATIONS = 2

# env_fn = create_nethack_env
# default_task = NetHackScore
# env_fn = create_cartpole_env
# default_task = (-0.1, 0.1)
env_fn = create_procgen_env
default_task = 0

env_args = ()
env_kwargs = {}
sample_env = env_fn(env_args=env_args, env_kwargs=env_kwargs)
# TODO: Test single process speed with Syllabus (with and without step updates)


@pytest.mark.benchmark(group="multiprocessing_speed")
def test_native_syllabus_speed(benchmark):
    def wrapper():
        curriculum = Constant(default_task, sample_env.task_space)
        curriculum = make_multiprocessing_curriculum(curriculum)
        run_native_vecenv(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum,
                          num_envs=N_ENVS, num_episodes=N_EPISODES, update_on_step=True)
        curriculum.stop()

    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)


@pytest.mark.benchmark(group="multiprocessing_speed")
def test_native_syllabus_speed_nostep(benchmark):
    def wrapper():
        curriculum = Constant(default_task, sample_env.task_space)
        curriculum = make_multiprocessing_curriculum(curriculum)
        run_native_vecenv(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum,
                          num_envs=N_ENVS, num_episodes=N_EPISODES, update_on_step=False)
        curriculum.stop()

    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)
