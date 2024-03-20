
from nle.env.tasks import NetHackScore, NetHackEat
from syllabus.tests.utils import create_nethack_env
from syllabus.curricula import NoopCurriculum, SequentialMetaCurriculum


def test_initialization():
    nethack_env = create_nethack_env()
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps=100", "episodes<=5&episode_return>10"], nethack_env.task_space)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps=100", "episodes<=5|episode_return>10"], nethack_env.task_space)
    curriculum = SequentialMetaCurriculum([NetHackScore, NetHackEat], ["steps=100", "episodes<=5&episode_return>10|steps=1000"], nethack_env.task_space)


test_initialization()
