""" Test curriculum synchronization across multiple processes. """
import pytest

from syllabus.core import MultiagentSharedCurriculumWrapper, make_multiprocessing_curriculum
from syllabus.core.evaluator import DummyEvaluator
from syllabus.curricula import (DomainRandomization,
                                LearningProgressCurriculum, NoopCurriculum,
                                PrioritizedLevelReplay)
from syllabus.tests import create_pistonball_env, run_native_multiprocess, run_single_process

N_ENVS = 2
N_EPISODES = 2

pistonball_env = create_pistonball_env()
default_task = pistonball_env.task_space.encode(1)
evaluator = DummyEvaluator(pistonball_env.action_space("piston_0"))

curricula = [
    (NoopCurriculum, create_pistonball_env, (default_task, pistonball_env.task_space), {}),
    (DomainRandomization, create_pistonball_env, (pistonball_env.task_space,), {}),
    # (LearningProgressCurriculum, create_pistonball_env, (pistonball_env.task_space,), {}),
    (PrioritizedLevelReplay, create_pistonball_env, (pistonball_env.task_space, pistonball_env.observation_space), {
        "evaluator": evaluator, "device": "cpu", "num_processes": N_ENVS*len(pistonball_env.possible_agents), "num_steps": 2048}),
]


test_names = [curriculum_args[0].__name__ for curriculum_args in curricula]


@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula, ids=test_names)
def test_multiprocessing_sync_single_process(curriculum, env_fn, args, kwargs):
    # Test single process speed
    print("RUNNING: Python single process test (1 env)...")
    sample_env = env_fn()
    single_kwargs = kwargs.copy()
    if "num_processes" in single_kwargs:
        single_kwargs["num_processes"] = len(sample_env.possible_agents)
    test_curriculum = curriculum(*args, **single_kwargs)
    test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
    native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=1, num_episodes=N_EPISODES)
    print(f"PASSED: single process test (1 env) passed: {native_speed:.2f}s")


@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula, ids=test_names)
def test_multiprocessing_sync_queue_multi_process_joint(curriculum, env_fn, args, kwargs):
    # Test Queue multiprocess speed with Syllabus
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    sample_env = env_fn()
    test_curriculum = curriculum(*args, **kwargs)
    test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents, joint_policy=True)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum)
    native_syllabus_speed = run_native_multiprocess(
        env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")


@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula, ids=test_names)
def test_multiprocessing_sync_queue_multi_process(curriculum, env_fn, args, kwargs):
    # Test Queue multiprocess speed with Syllabus
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    sample_env = env_fn()
    test_curriculum = curriculum(*args, **kwargs)
    test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents, joint_policy=True)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum)
    native_syllabus_speed = run_native_multiprocess(
        env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")


# if __name__ == "__main__":

#     for curriculum, env_fn, args, kwargs in curricula:
#         print("")
#         print("*" * 80)
#         print("Testing curriculum:", curriculum.__name__)
#         print("*" * 80)
#         print("")

#         sample_env = env_fn()

#         # Test single process speed
#         print("RUNNING: Single process test (2 envs)...")
#         test_curriculum = curriculum(*args, **kwargs)
#         test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
#         native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=2, num_episodes=N_EPISODES)
#         print(f"PASSED: Single process test (2 envs) passed: {native_speed:.2f}s")

#         # Test single process speed
#         print("\nRUNNING: Single process test (2 envs)...")
#         test_curriculum = curriculum(*args, **kwargs)
#         test_curriculum = MultiagentSharedCurriculumWrapper(
#             test_curriculum, sample_env.possible_agents, joint_policy=True)
#         native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=2, num_episodes=N_EPISODES)
#         print(f"PASSED: Single process test (2 envs) passed: {native_speed:.2f}s")

#         # Test multiprocess process speed without Syllabus
#         print("\nRUNNING: Python native multiprocess test (2 envs)...")
#         test_curriculum = curriculum(*args, **kwargs)
#         test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
#         native_speed = run_native_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
#         print(f"PASSED: Python native multiprocess test (2 envs) passed: {native_speed:.2f}s")

#         # Test multiprocess process speed without Syllabus
#         print("\nRUNNING: Python native multiprocess test (2 envs)...")
#         test_curriculum = curriculum(*args, **kwargs)
#         test_curriculum = MultiagentSharedCurriculumWrapper(
#             test_curriculum, sample_env.possible_agents, joint_policy=True)
#         native_speed = run_native_multiprocess(env_fn, num_envs=N_ENVS, num_episodes=N_EPISODES)
#         print(f"PASSED: Python native multiprocess test (2 envs) passed: {native_speed:.2f}s")

#         # Test Queue multiprocess speed with Syllabus
#         test_curriculum = curriculum(*args, **kwargs)
#         test_curriculum = MultiagentSharedCurriculumWrapper(test_curriculum, sample_env.possible_agents)
#         test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
#         print("\nRUNNING: Python native multiprocess test with Syllabus...")
#         native_syllabus_speed = run_native_multiprocess(
#             env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
#         print(f"PASSED: Python native multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")

#         # Test Queue multiprocess speed with Syllabus
#         test_curriculum = curriculum(*args, **kwargs)
#         test_curriculum = MultiagentSharedCurriculumWrapper(
#             test_curriculum, sample_env.possible_agents, joint_policy=True)
#         test_curriculum = make_multiprocessing_curriculum(test_curriculum, sequential_start=False)
#         print("\nRUNNING: Python native multiprocess test with Syllabus...")
#         native_syllabus_speed = run_native_multiprocess(
#             env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
#         print(f"PASSED: Python native multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")
