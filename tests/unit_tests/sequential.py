
from copy import deepcopy

import pytest
from nle.env.tasks import NetHackEat, NetHackScore

from syllabus.core import make_multiprocessing_curriculum
from syllabus.curricula import SequentialMetaCurriculum
from syllabus.tests.utils import create_nethack_env, run_native_multiprocess, run_single_process


@pytest.fixture(scope="module")
def create_env():
    return create_nethack_env


def run_curriculum(curriculum, env_fn):
    # Test single process speed
    test_curriculum = deepcopy(curriculum)
    native_speed = run_single_process(env_fn, curriculum=test_curriculum, num_envs=1, num_episodes=4)

    # Test Queue multiprocess speed with Syllabus
    test_curriculum = deepcopy(curriculum)
    test_curriculum = make_multiprocessing_curriculum(test_curriculum)
    native_syllabus_speed = run_native_multiprocess(env_fn, curriculum=test_curriculum, num_envs=4, num_episodes=4)


def test_parsing_condition_operators(create_env):
    env = create_env()
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps<100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps<=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps>=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps>100"], env.task_space)
    run_curriculum(curriculum, create_env)


def test_parsing_compount_conditions(create_env):
    env = create_env()
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["episodes>5&steps=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps=100|episode_return<=5"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps=100|steps>200"], env.task_space)
    run_curriculum(curriculum, create_env)


if __name__ == "__main__":
    test_parsing_condition_operators()
    test_parsing_compount_conditions()
    print("All tests passed!")
