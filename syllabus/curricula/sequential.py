import re
import warnings
from typing import Any, Callable, List, Union

from syllabus.core import Curriculum
from syllabus.curricula import NoopCurriculum, DomainRandomization
from syllabus.task_space import TaskSpace


class SequentialCurriculum(Curriculum):
    REQUIRES_STEP_UPDATES = False
    REQUIRES_EPISODE_UPDATES = True
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, curriculum_list: List[Curriculum], stopping_conditions: List[Any], *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        assert len(curriculum_list) > 0, "Must provide at least one curriculum"
        assert len(stopping_conditions) == len(curriculum_list) - 1, f"Stopping conditions must be one less than the number of curricula. Final curriculum is used for the remainder of training. Expected {len(curriculum_list) - 1}, got {len(stopping_conditions)}."
        if len(curriculum_list) == 1:
            warnings.warn("Your sequential curriculum only containes one element. Consider using that element directly instead.")

        self.curriculum_list = self._parse_curriculum_list(curriculum_list)
        self.stopping_conditions = self._parse_stopping_conditions(stopping_conditions)
        self._curriculum_index = 0

        # Stopping metrics
        self.n_steps = 0
        self.total_steps = 0
        self.n_episodes = 0
        self.total_episodes = 0
        self.n_tasks = 0
        self.total_tasks = 0
        self.episode_returns = []

    def _parse_curriculum_list(self, curriculum_list: List[Curriculum]) -> List[Curriculum]:
        """ Parse the curriculum list to ensure that all items are curricula. 
        Adds Curriculum objects directly. Wraps task space items in NoopCurriculum objects.
        """
        parsed_list = []
        for item in curriculum_list:
            if isinstance(item, Curriculum):
                parsed_list.append(item)
            elif isinstance(item, TaskSpace):
                parsed_list.append(DomainRandomization(item))
            elif isinstance(item, list):
                task_space = TaskSpace(len(item), item)
                parsed_list.append(DomainRandomization(task_space))
            elif self.task_space.contains(item):
                parsed_list.append(NoopCurriculum(item, self.task_space))
            else:
                raise ValueError(f"Invalid curriculum item: {item}")

        return parsed_list

    def _parse_stopping_conditions(self, stopping_conditions: List[Any]) -> List[Any]:
        """ Parse the stopping conditions to ensure that all items are integers. """
        parsed_list = []
        for item in stopping_conditions:
            if isinstance(item, Callable):
                parsed_list.append(item)
            elif isinstance(item, str):
                parsed_list.append(self._parse_condition_string(item))
            else:
                raise ValueError(f"Invalid stopping condition: {item}")

        return parsed_list

    def _parse_condition_string(self, condition: str) -> Callable:
        """ Parse a string condition to a callable function. """

        # Parse composite conditions
        if '|' in condition:
            conditions = re.split(re.escape('|'), condition)
            return lambda: any(self._parse_condition_string(cond)() for cond in conditions)
        elif '&' in condition:
            conditions = re.split(re.escape('&'), condition)
            return lambda: all(self._parse_condition_string(cond)() for cond in conditions)

        clauses = re.split('(<=|>=|=|<|>)', condition)

        try:
            metric, comparator, value = clauses

            if metric == "steps":
                metric_fn = self._get_steps
            elif metric == "total_steps":
                metric_fn = self._get_total_steps
            elif metric == "episodes":
                metric_fn = self._get_episodes
            elif metric == "total_episodes":
                metric_fn = self._get_total_episodes
            elif metric == "tasks":
                metric_fn = self._get_tasks
            elif metric == "total_tasks":
                metric_fn = self._get_total_tasks
            elif metric == "episode_return":
                metric_fn = self._get_episode_return
            else:
                raise ValueError(f"Invalid metric name: {metric}")

            if comparator == '<':
                return lambda: metric_fn() < float(value)
            elif comparator == '>':
                return lambda: metric_fn() > float(value)
            elif comparator == '<=':
                return lambda: metric_fn() <= float(value)
            elif comparator == '>=':
                return lambda: metric_fn() >= float(value)
            elif comparator == '=':
                return lambda: metric_fn() == float(value)
            else:
                raise ValueError(f"Invalid comparator: {comparator}")
        except ValueError as e:
            raise ValueError(f"Invalid condition string: {condition}") from e

    def _get_steps(self):
        return self.n_steps

    def _get_total_steps(self):
        return self.total_steps

    def _get_episodes(self):
        return self.n_episodes

    def _get_total_episodes(self):
        return self.total_episodes

    def _get_tasks(self):
        return self.n_tasks

    def _get_total_tasks(self):
        return self.total_tasks

    def _get_episode_return(self):
        return sum(self.episode_returns) / len(self.episode_returns) if len(self.episode_returns) > 0 else 0

    @property
    def current_curriculum(self):
        return self.curriculum_list[self._curriculum_index]

    @property
    def requires_step_updates(self):
        return any(map(lambda c: c.requires_step_updates, self.curriculum_list))

    def _sample_distribution(self) -> List[float]:
        """
        Return None to indicate that tasks are not drawn from a distribution.
        """
        return None

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Choose the next k tasks from the list.
        """
        curriculum = self.current_curriculum
        tasks = curriculum.sample(k)

        # Recode tasks into environment task space
        decoded_tasks = [curriculum.task_space.decode(task) for task in tasks]
        recoded_tasks = [self.task_space.encode(task) for task in decoded_tasks]

        self.n_tasks += k
        self.total_tasks += k

        # Check if we should move on to the next phase of the curriculum
        self.check_stopping_conditions()
        return recoded_tasks

    def update_on_episode(self, episode_return, episode_len, episode_task, env_id=None):
        self.n_episodes += 1
        self.total_episodes += 1
        self.n_steps += episode_len
        self.total_steps += episode_len
        self.episode_returns.append(episode_return)

        # Update current curriculum
        if self.current_curriculum.requires_episode_updates:
            self.current_curriculum.update_on_episode(episode_return, episode_len, episode_task, env_id)

    def update_on_step(self, task, obs, rew, term, trunc, info, env_id=None):
        if self.current_curriculum.requires_step_updates:
            self.current_curriculum.update_on_step(task, obs, rew, term, trunc, info, env_id)

    def update_on_step_batch(self, step_results, env_id=None):
        if self.current_curriculum.requires_step_updates:
            self.current_curriculum.update_on_step_batch(step_results, env_id)

    def update_on_demand(self, metrics):
        self.current_curriculum.update_on_demand(metrics)

    def update_task_progress(self, task, progress, env_id=None):
        self.current_curriculum.update_task_progress(task, progress, env_id)

    def check_stopping_conditions(self):
        if self._curriculum_index < len(self.stopping_conditions) and self.stopping_conditions[self._curriculum_index]():
            self._curriculum_index += 1
            self.n_episodes = 0
            self.n_steps = 0
            self.episode_returns = []
            self.n_tasks = 0

    def log_metrics(self, writer, step=None, log_full_dist=False):
        # super().log_metrics(writer, step, log_full_dist)
        writer.add_scalar("curriculum/current_stage", self._curriculum_index, step)
        writer.add_scalar("curriculum/steps", self.n_steps, step)
        writer.add_scalar("curriculum/episodes", self.n_episodes, step)
        writer.add_scalar("curriculum/episode_returns", self._get_episode_return(), step)
