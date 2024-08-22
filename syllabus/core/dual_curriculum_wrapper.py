from typing import Tuple, TypeVar

from gymnasium import spaces

from syllabus.core.curriculum_base import Curriculum, TaskSpace

AgentID = TypeVar("AgentID")
Agent = TypeVar("Agent")
EnvTask = TypeVar("EnvTask")
AgentTask = TypeVar("AgentTask")


class DualCurriculumWrapper(Curriculum):
    """Curriculum wrapper containing both an agent and environment-based curriculum."""

    def __init__(
        self,
        env_curriculum: Curriculum,
        agent_curriculum: Curriculum,
        batch_agent_tasks: bool = False,
        batch_size: int = 32,
        *args,
        **kwargs,
    ) -> None:
        self.agent_curriculum = agent_curriculum
        self.env_curriculum = env_curriculum
        self.task_space = TaskSpace(
            spaces.Dict(
                {
                    "env_space": env_curriculum.task_space.gym_space,
                    "agent_space": agent_curriculum.task_space.gym_space,
                }
            )
        )
        self.batch_agent_tasks = batch_agent_tasks
        self.batch_size = batch_size
        self.batched_tasks = []
        self.agent_task = None
        super().__init__(task_space=self.task_space, *args, **kwargs)

    def sample(self, k=1) -> Tuple[EnvTask, AgentTask]:
        """Sets new tasks for the environment and agent curricula."""
        env_task = self.env_curriculum.sample(k=k)
        if len(self.batched_tasks) < k:
            self.batched_tasks = self.agent_curriculum.sample(k=1) * self.batch_size
        agent_task = [self.batched_tasks.pop() for _ in range(k)]
        return list(zip(env_task, agent_task))

    def get_opponent(self, agent_task: AgentTask) -> Agent:
        return self.agent_curriculum.get_opponent(agent_task)

    def update_agent(self, agent: Agent) -> Agent:
        return self.agent_curriculum.update_agent(agent)

    def update_winrate(self, opponent_id: int, opponent_reward: int) -> None:
        return self.agent_curriculum.update_winrate(opponent_id, opponent_reward)

    def update_on_step(self, task, obs, reward, term, trunc, info, env_id=None):
        if self.env_curriculum.requires_step_updates:
            self.env_curriculum.update_on_step(
                task, obs, reward, term, trunc, info, env_id=env_id
            )
        if self.agent_curriculum.requires_step_updates:
            self.agent_curriculum.update_on_step(
                task, obs, reward, term, trunc, info, env_id=env_id
            )

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
