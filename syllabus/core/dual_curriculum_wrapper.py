from typing import Tuple, TypeVar

from syllabus.core.curriculum_base import Curriculum
from syllabus.core.task_interface import TaskWrapper

AgentID = TypeVar("AgentID")
Agent = TypeVar("Agent")
EnvTask = TypeVar("EnvTask")
AgentTask = TypeVar("AgentTask")


class DualCurriculumWrapper:
    """Curriculum wrapper containing both an agent and environment-based curriculum."""

    def __init__(
        self,
        env: TaskWrapper,
        env_curriculum: Curriculum,
        agent_curriculum: Curriculum,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.env = env
        self.agent_curriculum = agent_curriculum
        self.env_curriculum = env_curriculum
        self.task_space = (env_curriculum.task_space, agent_curriculum.task_space)
        # self.env_mp_curriculum, self.env_task_queue, self.env_update_queue = (
        #     make_multiprocessing_curriculum(env_curriculum)
        # )
        # self.agent_mp_curriculum, self.agent_task_queue, self.agent_update_queue = (
        #     make_multiprocessing_curriculum(agent_curriculum)
        # )

    def sample(self, k=1) -> Tuple[EnvTask, AgentTask]:
        """Sets new tasks for the environment and agent curricula."""
        self.env_task = self.env_curriculum.sample()
        self.agent_task = self.agent_curriculum.sample()
        return self.env_task, self.agent_task

    def get_opponent(self, agent_task: AgentTask) -> Agent:
        return self.agent_curriculum.get_opponent(agent_task)

    def update_agent(self, agent: Agent) -> Agent:
        return self.agent_curriculum.update_agent(agent)

    def __getattr__(self, name):
        """Delegate attribute lookup to the curricula if not found."""
        if hasattr(self.env_curriculum, name):
            return getattr(self.env_curriculum, name)
        elif hasattr(self.agent_curriculum, name):
            return getattr(self.agent_curriculum, name)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
