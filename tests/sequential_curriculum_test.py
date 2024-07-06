import pytest
from nle.env.tasks import NetHackScore, NetHackScout, NetHackStaircase

from syllabus.core import make_multiprocessing_curriculum, make_ray_curriculum
from syllabus.curricula import (AnnealingBoxCurriculum,
                                CentralizedPrioritizedLevelReplay,
                                DomainRandomization,
                                LearningProgressCurriculum, NoopCurriculum,
                                PrioritizedLevelReplay, SequentialCurriculum, Condition,
                                SimpleBoxCurriculum)
from syllabus.tests import (create_cartpole_env, create_nethack_env,
                            get_test_values, run_native_multiprocess,
                            run_ray_multiprocess, run_single_process, run_episode)

N_ENVS = 1
N_EPISODES = 34

nethack_env = create_nethack_env()
cartpole_env = create_cartpole_env()

def create_sequential_curriculum_1(task_space):

    curricula = []
    stopping = []

    # Stage 1 - Survival
    stage1 = [0, 1, 2, 3]
    stopping.append(
        Condition(
            metric_name="episode_return", comparator='>=', value=0.9
        ) & Condition(
            metric_name="episodes", comparator='>=', value=5000
        )
    )

    # Stage 2 - Harvest Equipment
    stage2 = [4, 5]
    stopping.append(
        Condition(
            metric_name="episode_return", comparator='>=', value=0.9
        ) & Condition(
            metric_name="episodes", comparator='>=', value=5000
        )
    )

    # Stage 3 - Equip Weapons
    stage3 = [6, 7]

    curricula = [stage1, stage2, stage3]
    curriculum = SequentialCurriculum(curricula, stopping, task_space)

    return curriculum

def foo(obs, info):
    return

def create_sequential_curriculum_2(task_space):
    curricula = []
    stopping = []

    # Stage 1 - Survival
    stage1 = [0, 1, 2, 3]
    stopping.append(
        Condition(
            metric_name="episode_return", comparator='>=', value=0.9
        ) & Condition(
            metric_name="episodes", comparator='>=', value=5000
        ) & Condition(
            metric_name="my_custom_function", comparator='>=', value=1.0, custom_metrics={"my_custom_function": foo}
        )
    )

    # Stage 2 - Harvest Equipment
    stage2 = [4, 5]
    stopping.append(
        Condition(
            metric_name="episode_return", comparator='>=', value=0.9
        ) & Condition(
            metric_name="episodes", comparator='>=', value=5000
        ) & Condition(
            metric_name="my_custom_function", comparator='>=', value=2.0, custom_metrics={"my_custom_function": foo}
        )
    )

    # Stage 3 - Equip Weapons
    stage3 = [6, 7]

    curricula = [stage1, stage2, stage3]
    curriculum = SequentialCurriculum(curricula, stopping, task_space)

    return curriculum


def test_custom_sequential_curriculum():
    task_space = nethack_env.task_space

    curriculum = create_sequential_curriculum_2(task_space)

    native_speed = run_single_process(create_nethack_env, curriculum=curriculum, num_envs=1, num_episodes=N_EPISODES)
    print(f"PASSED: single process test (1 env) passed: {native_speed:.2f}s")

test_custom_sequential_curriculum()