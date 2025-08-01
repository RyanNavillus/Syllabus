""" Test curriculum synchronization across multiple processes. """
import gym
import gymnasium as gym
import pytest

from syllabus.core import make_multiprocessing_curriculum
from syllabus.core.evaluator import DummyEvaluator
from syllabus.curricula import (SimulatedAnnealing,
                                CentralPrioritizedLevelReplay,
                                DomainRandomization,
                                LearningProgress, Constant,
                                PrioritizedLevelReplay, SequentialCurriculum,
                                ExpandingBox)
from syllabus.tests import create_cartpole_env, create_nethack_env, run_native_multiprocess, run_single_process

N_ENVS = 2
N_EPISODES = 2

nethack_env = create_nethack_env()
cartpole_env = create_cartpole_env()
eval_envs = gym.vector.SyncVectorEnv(
    [create_nethack_env(wrap=True, eval=True) for _ in range(8)]
)
evaluator = DummyEvaluator(nethack_env.action_space)

curricula = [
    (Constant, create_nethack_env, (0, nethack_env.task_space), {}),
    (DomainRandomization, create_nethack_env, (nethack_env.task_space,), {}),
    (LearningProgress, create_nethack_env, (nethack_env.task_space, ),
     {"eval_envs": eval_envs, "evaluator": evaluator, "eval_eps": 100}),
    (CentralPrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space,),
     {"device": "cpu", "suppress_usage_warnings": True, "num_processes": N_ENVS}),
    (CentralPrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space,), {
        "evaluator": evaluator,
        "device": "cpu",
        "num_processes": N_ENVS,
        "num_steps": 2048,
        "robust_plr": True,
        "eval_envs": eval_envs,
        "task_sampler_kwargs_dict": {
            "strategy": "grounded_signed_value_loss",
            "replay_schedule": "proportionate",
            "rho": 0.5,
            "replay_prob": 0.5,
        }
    }),
    (PrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space, nethack_env.observation_space), {
        "evaluator": evaluator,
        "device": "cpu",
        "num_processes": N_ENVS,
        "num_steps": 2048
    }),
    (ExpandingBox, create_cartpole_env, (cartpole_env.task_space,), {}),
    (SimulatedAnnealing, create_cartpole_env, (cartpole_env.task_space,), {
        'start_values': [-0.02, 0.02],
        'end_values': [-0.3, 0.3],
        'total_steps': [10]
    }),
    (SequentialCurriculum, create_nethack_env, ([CentralPrioritizedLevelReplay(nethack_env.task_space, device="cpu", suppress_usage_warnings=True, num_processes=N_ENVS), PrioritizedLevelReplay(
        nethack_env.task_space, nethack_env.observation_space, evaluator=evaluator, device="cpu", num_processes=N_ENVS, num_steps=2048), 0, [1, 2]], ["steps>1000", "episodes>=50", "tasks>20"], nethack_env.task_space), {}),
]

test_names = [curriculum_args[0].__name__ for curriculum_args in curricula]


@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula, ids=test_names)
def test_multiprocessing_sync_single_process(curriculum, env_fn, args, kwargs):
    # Test single process speed
    print("RUNNING: Python single process test (1 env)...")
    single_kwargs = kwargs.copy()
    if "num_processes" in single_kwargs:
        single_kwargs["num_processes"] = 1
    test_curriculum = curriculum(*args, **single_kwargs)
    native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=1, num_episodes=N_EPISODES)
    print(f"PASSED: single process test (1 env) passed: {native_speed:.2f}s")


@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula, ids=test_names)
def test_multiprocessing_sync_queue_multi_process(curriculum, env_fn, args, kwargs):
    # Test Queue multiprocess speed with Syllabus
    test_curriculum = curriculum(*args, **kwargs)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum)
    print("\nRUNNING: Python multiprocess test with Syllabus...")
    native_syllabus_speed = run_native_multiprocess(
        env_fn, curriculum=test_curriculum, num_envs=N_ENVS, num_episodes=N_EPISODES)
    print(f"PASSED: Python multiprocess test with Syllabus: {native_syllabus_speed:.2f}s")


if __name__ == "__main__":
    test_multiprocessing_sync_single_process(*curricula[2])
    test_multiprocessing_sync_queue_multi_process(*curricula[2])
