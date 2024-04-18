# flake8: noqa: F401
import sys

from .domain_randomization import DomainRandomization
from .learning_progress import LearningProgressCurriculum
from .noop import NoopCurriculum
from .plr.plr_wrapper import PrioritizedLevelReplay
from .plr.task_sampler import TaskSampler
from .selfplay import FictitiousSelfPlay, PrioritizedFictitiousSelfPlay, SelfPlay
from .sequential import SequentialCurriculum
from .simple_box import SimpleBoxCurriculum
