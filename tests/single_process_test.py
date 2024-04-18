""" Test curriculum synchronization across multiple processes. """
import pytest
from nle.env.tasks import NetHackScore, NetHackScout, NetHackStaircase

from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.curricula import (AnnealingBoxCurriculum,
                                CentralizedPrioritizedLevelReplay,
                                DomainRandomization,
                                LearningProgressCurriculum, NoopCurriculum,
                                PrioritizedLevelReplay, SequentialCurriculum,
                                SimpleBoxCurriculum)
from syllabus.tests import (create_cartpole_env, create_nethack_env,
                            get_test_values, run_native_multiprocess,
                            run_ray_multiprocess, run_single_process, run_episode)

N_ENVS = 1
N_EPISODES = 34


nethack_env = create_nethack_env()
cartpole_env = create_cartpole_env()

curricula = [
    (NoopCurriculum, create_nethack_env, (NetHackScore, nethack_env.task_space), {}),
    (DomainRandomization, create_nethack_env, (nethack_env.task_space,), {}),
    # (LearningProgressCurriculum, create_nethack_env, (nethack_env.task_space,), {}),
    (CentralizedPrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space,), {
        "device": "cpu", "suppress_usage_warnings": True, "num_processes": N_ENVS
    }),
    (PrioritizedLevelReplay, create_nethack_env, (nethack_env.task_space, nethack_env.observation_space), {
        "get_value": get_test_values,
        "device": "cpu",
        "num_processes": N_ENVS,
        "num_steps": 2048
    }),
    (SimpleBoxCurriculum, create_cartpole_env, (cartpole_env.task_space,), {}),
    (AnnealingBoxCurriculum, create_cartpole_env, (cartpole_env.task_space,), {
        'start_values': [-0.02, 0.02],
        'end_values': [-0.3, 0.3],
        'total_steps': [10]
    }),
    (SequentialCurriculum, create_nethack_env, ([
        CentralizedPrioritizedLevelReplay(nethack_env.task_space, device="cpu", suppress_usage_warnings=True, num_processes=N_ENVS, warmup_strategy = 'random', warmup_samples = 1),
        PrioritizedLevelReplay(nethack_env.task_space, nethack_env.observation_space, get_value=get_test_values, device="cpu", num_processes=N_ENVS, num_steps=2048, warmup_strategy = 'fix', warmup_samples = 1),
        NetHackScore,
        [NetHackScout, NetHackStaircase]
    ], ["steps>1000", "episodes>=50", "tasks>20"], nethack_env.task_space), {}),
]

test_names = [curriculum_args[0].__name__ for curriculum_args in curricula]
@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula, ids=test_names)
def test_multiprocessing_sync_single_process_fix(curriculum, env_fn, args, kwargs):
    # Test single process speed
    print("RUNNING: Python single process test (1 env)...")
    single_kwargs = kwargs.copy()
    if "num_processes" in single_kwargs:
        single_kwargs["num_processes"] = 1
    test_curriculum = curriculum(*args, warmup_strategy="fix", warmup_samples=26, **kwargs)
    env = env_fn(env_args=(), env_kwargs={})
    ep_rews = []
    for i in range(N_EPISODES):
        if test_curriculum:
            if i == N_EPISODES - 9 and (not isinstance(test_curriculum, NoopCurriculum)):
                assert test_curriculum._should_use_startup_sampling()
            if i == N_EPISODES - 8 and (not isinstance(test_curriculum, NoopCurriculum)):
                assert not test_curriculum._should_use_startup_sampling()
            task = env.task_space.decode(test_curriculum.sample()[0])
            ep_rews.append(run_episode(env, new_task=task, curriculum=test_curriculum, env_id=0))
        else:
            ep_rews.append(run_episode(env))
    env.close()
    print("PASSED: single process test on fix sampling (1 env) passed")

test_names = [curriculum_args[0].__name__ for curriculum_args in curricula]
@pytest.mark.parametrize("curriculum, env_fn, args, kwargs", curricula, ids=test_names)
def test_multiprocessing_sync_single_process_random(curriculum, env_fn, args, kwargs):
    # Test single process speed
    print("RUNNING: Python single process test (1 env)...")
    single_kwargs = kwargs.copy()
    if "num_processes" in single_kwargs:
        single_kwargs["num_processes"] = 1
    test_curriculum = curriculum(*args, warmup_strategy="random", warmup_samples=26, **kwargs)
    env = env_fn(env_args=(), env_kwargs={})
    ep_rews = []
    for i in range(N_EPISODES):
        if test_curriculum:
            if i == N_EPISODES - 9 and (not isinstance(test_curriculum, NoopCurriculum)):
                assert test_curriculum._should_use_startup_sampling()
            if i == N_EPISODES - 8 and (not isinstance(test_curriculum, NoopCurriculum)):
                assert not test_curriculum._should_use_startup_sampling()
            task = env.task_space.decode(test_curriculum.sample()[0])
            ep_rews.append(run_episode(env, new_task=task, curriculum=test_curriculum, env_id=0))
        else:
            ep_rews.append(run_episode(env))
    env.close()
    print("PASSED: single process test on random sampling (1 env) passed")
