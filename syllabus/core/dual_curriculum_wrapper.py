from typing import Tuple, TypeVar

from syllabus.core import Agent, Curriculum, CurriculumWrapper
from syllabus.task_space import TupleTaskSpace

EnvTask = TypeVar("EnvTask")
AgentTask = TypeVar("AgentTask")


class DualCurriculumWrapper(CurriculumWrapper):
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
        self.task_space = TupleTaskSpace(
            env_curriculum.task_space.gym_space,
            agent_curriculum.task_space.gym_space,
        )
        self.batch_agent_tasks = batch_agent_tasks
        self.batch_size = batch_size
        self.batched_tasks = []
        self.agent_task = None
        super().__init__(self.task_space, *args, **kwargs)

    def sample(self, k=1) -> Tuple[EnvTask, AgentTask]:
        """Sets new tasks for the environment and agent curricula."""
        env_task = self.env_curriculum.sample(k=k)
        if len(self.batched_tasks) < k:
            self.batched_tasks = self.agent_curriculum.sample(k=1) * self.batch_size
        agent_task = [self.batched_tasks.pop() for _ in range(k)]
        return list(zip(env_task, agent_task))

    def get_agent(self, agent: AgentTask) -> Agent:
        return self.agent_curriculum.get_opponent(agent)

    def update_agent(self, agent: Agent) -> int:
        return self.agent_curriculum.update_agent(agent)

    def update_on_episode(self, episode_return, length, task, progress, env_id=None):
        self.env_curriculum.update_on_episode(episode_return, length, task[0], progress, env_id)
        self.agent_curriculum.update_on_episode(episode_return, length, task[1], progress, env_id)

    def update_on_step(self, task, obs, reward, term, trunc, info, env_id=None):
        if self.env_curriculum.requires_step_updates:
            self.env_curriculum.update_on_step(
                task[0], obs, reward, term, trunc, info, env_id=env_id
            )
        if self.agent_curriculum.requires_step_updates:
            self.agent_curriculum.update_on_step(
                task[1], obs, reward, term, trunc, info, env_id=env_id
            )

    def update_on_step_batch(self, step_results, env_id=None):
        tasks, o, r, t, tr, i, p = step_results
        env_step_results = ([task[0] for task in tasks], o, r, t, tr, i, p)
        agent_step_results = ([task[1] for task in tasks], o, r, t, tr, i, p)
        if self.env_curriculum.requires_step_updates:
            self.env_curriculum.update_on_step_batch(env_step_results, env_id=env_id)
        if self.agent_curriculum.requires_step_updates:
            self.agent_curriculum.update_on_step_batch(agent_step_results, env_id=env_id)

    def update_task_progress(self, task, progress):
        self.env_curriculum.update_task_progress(task[0], progress)
        self.agent_curriculum.update_task_progress(task[1], progress)

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
