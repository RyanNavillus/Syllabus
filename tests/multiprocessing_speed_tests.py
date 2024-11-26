""" Test curriculum synchronization across multiple processes. """
from nle.env.tasks import NetHackScore

from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.core.evaluator import DummyEvaluator
from syllabus.curricula import NoopCurriculum
from syllabus.curricula import PrioritizedLevelReplay
from syllabus.tests import create_cartpole_env, create_nethack_env, create_procgen_env, run_native_multiprocess, run_ray_multiprocess
import pytest

N_ENVS = 128
N_EPISODES = 64
N_ROUNDS = 1
N_ITERATIONS = 1

env_fn = create_nethack_env
default_task = NetHackScore
# env_fn = create_cartpole_env
# default_task = (-0.1, 0.1)
# env_fn = create_procgen_env
# default_task = 0

env_args = ()
env_kwargs = {}
sample_env = env_fn(env_args=env_args, env_kwargs=env_kwargs)


@pytest.mark.benchmark(group="multiprocessing_speed")
def test_native_speed(benchmark):
    def wrapper():
        return run_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, num_envs=N_ENVS, num_episodes=N_EPISODES)

    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)


@pytest.mark.benchmark(group="multiprocessing_speed")
def test_native_syllabus_speed(benchmark):
    def wrapper():
        curriculum = NoopCurriculum(default_task, sample_env.task_space)
        curriculum = make_multiprocessing_curriculum(curriculum)
        return run_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)

    benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)


# @pytest.mark.benchmark(group="multiprocessing_plr_speed")
# def test_plr_speed(benchmark):
#     def wrapper():
#         evaluator = DummyEvaluator(sample_env.action_space)
#         curriculum = PrioritizedLevelReplay(sample_env.task_space,
#                                             sample_env.observation_space,
#                                             evaluator=evaluator,
#                                             device="cpu",
#                                             num_processes=N_ENVS,
#                                             num_steps=256)

#         curriculum = make_multiprocessing_curriculum(curriculum)
#         return run_native_multiprocess(env_fn, env_args=env_args, env_kwargs=env_kwargs, curriculum=curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
#     benchmark.pedantic(wrapper, iterations=N_ITERATIONS, rounds=N_ROUNDS)
