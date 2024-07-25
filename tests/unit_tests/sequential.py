
from copy import deepcopy

import pytest
from nle.env.tasks import NetHackEat, NetHackScore

from syllabus.core import make_multiprocessing_curriculum
from syllabus.curricula import SequentialCurriculum, NoopCurriculum, DomainRandomization
from syllabus.task_space import TaskSpace
from syllabus.tests.utils import create_nethack_env, run_native_multiprocess, run_single_process, run_set_length


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
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["steps<100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["steps<=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["steps=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["steps>=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["steps>100"], env.task_space)
    run_curriculum(curriculum, create_env)


def test_parsing_compount_conditions(create_env):
    env = create_env()
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["episodes>5&steps=100"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["steps=100|episode_return<=5"], env.task_space)
    run_curriculum(curriculum, create_env)
    curriculum = SequentialCurriculum([NetHackScore, NetHackEat], ["steps=100|steps>200"], env.task_space)
    run_curriculum(curriculum, create_env)


def test_curriculum_sequence_2step(create_env):
    env = create_env()
    curriculum = SequentialCurriculum([NetHackScore, TaskSpace(3, env.task_space.list_tasks()[1:4])], ["steps>100"], env.task_space)
    assert isinstance(curriculum.current_curriculum, NoopCurriculum)
    env_outputs = run_set_length(env, curriculum, steps=50)
    assert isinstance(curriculum.current_curriculum, NoopCurriculum)
    run_set_length(env, curriculum, episodes=1, env_outputs=env_outputs)
    assert isinstance(curriculum.current_curriculum, DomainRandomization)


def test_curriculum_sequence_3step(create_env):
    env = create_env()
    curriculum = SequentialCurriculum([NetHackScore, TaskSpace(3, env.task_space.list_tasks()[1:4]), NetHackEat], ["steps>100", "episodes>=5"], env.task_space)
    assert isinstance(curriculum.current_curriculum, NoopCurriculum)
    env_outputs = run_set_length(env, curriculum, steps=50)
    assert isinstance(curriculum.current_curriculum, NoopCurriculum)
    run_set_length(env, curriculum, episodes=1, env_outputs=env_outputs)
    assert isinstance(curriculum.current_curriculum, DomainRandomization)
    run_set_length(env, curriculum, episodes=5, env_outputs=env_outputs)
    assert isinstance(curriculum.current_curriculum, NoopCurriculum)


if __name__ == "__main__":
    test_parsing_condition_operators(create_nethack_env)
    test_parsing_compount_conditions(create_nethack_env)
    # test_curriculum_sequence_2step(create_nethack_env)
    # test_curriculum_sequence_3step(create_nethack_env)
    print("All tests passed!")
