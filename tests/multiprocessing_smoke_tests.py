""" Test curriculum synchronization across multiple processes. """
import ray
from nle.env.tasks import NetHackScore
import pytest

from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.curricula import (CentralizedPrioritizedLevelReplay,
                                DomainRandomization,
                                LearningProgressCurriculum,
                                NoopCurriculum,
                                PrioritizedLevelReplay,
                                SimpleBoxCurriculum,
                                AnnealingBoxCurriculum)
from syllabus.tests import (create_cartpole_env,
                            create_nethack_env,
                            get_test_values)

from syllabus.tests import test_single_process as run_single_process
from syllabus.tests import test_native_multiprocess as run_native_multiprocess
from syllabus.tests import test_ray_multiprocess as run_ray_multiprocess

N_ENVS = 2
N_EPISODES = 2


ray.init()
nethack_env = create_nethack_env()
cartpole_env = create_cartpole_env()

curricula = [
        (NoopCurriculum, create_nethack_env, (NetHackScore, nethack_env.task_space), {}),
        (DomainRandomization, create_nethack_env, (nethack_env.task_space,), {}),
        # (LearningProgressCurriculum, create_nethack_env, (nethack_env.task_space,), {}),
        (CentralizedPrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space,), {"device": "cpu", "suppress_usage_warnings": True, "num_processes": N_ENVS}),
        #(PrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space, nethack_env.observation_space), {
        #    "get_value": get_test_values,
        #    "device": "cpu",
        #    "num_processes": N_ENVS,
        #    "num_steps": 2048
        #}),
        (SimpleBoxCurriculum, create_cartpole_env, (cartpole_env.task_space,), {}),
        (AnnealingBoxCurriculum, create_cartpole_env, (cartpole_env.task_space,), {
            'start_values': [-0.02, 0.02],
            'end_values': [-0.3, 0.3],
            'total_steps': [10]
        }),
    ]

@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula)
def test_multiprocessing_sync_single_process(curriculum, env_fn, args, kwargs):
    # Test single process speed
    print("RUNNING: Python single process test (1 env)...")
    single_kwargs = kwargs.copy()
    if "num_processes" in single_kwargs:
        single_kwargs["num_processes"] = 1
    test_curriculum = curriculum(*args, **single_kwargs)
    native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=1, num_episodes=N_EPISODES)
    print(f"PASSED: single process test (1 env) passed: {native_speed:.2f}s")


@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula)
def test_multiprocessing_sync_queue_multi_process(curriculum, env_fn, args, kwargs):
    # Test Queue multiprocess speed with Syllabus
    test_curriculum = curriculum(*args, **kwargs)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = run_native_multiprocess(env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula)
def test_multiprocessing_sync_ray_multi_process(curriculum, env_fn, args, kwargs):
    # Test Ray multiprocess speed with Syllabus
    test_curriculum = curriculum(*args, **kwargs)
    test_curriculum = make_ray_curriculum(test_curriculum)
    print("\nRUNNING: Ray multiprocess test with Syllabus...")
    ray_syllabus_speed = run_ray_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Ray multiprocess test with Syllabus: {ray_syllabus_speed:.2f}s")
