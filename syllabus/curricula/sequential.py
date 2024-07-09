import re
import warnings
from typing import Any, Callable, List, Union

from syllabus.core import Curriculum
from syllabus.curricula import NoopCurriculum, DomainRandomization
from syllabus.task_space import TaskSpace


class Condition:

    def __init__(self, metric_name: str, comparator: str, value: float, custom_metrics=None):
        self.metric_name = metric_name
        self.comparator = comparator
        self.value = value
        self.custom_metrics = custom_metrics or {}

    def __call__(self, curriculum):
        predefined_metrics = {
        "steps": lambda curriculum: curriculum._get_steps(),
        "total_steps": lambda curriculum: curriculum._get_total_steps(),
        "episodes": lambda curriculum: curriculum._get_episodes(),
        "total_episodes": lambda curriculum: curriculum._get_total_episodes(),
        "tasks": lambda curriculum: curriculum._get_tasks(),
        "total_tasks": lambda curriculum: curriculum._get_total_tasks(),
        "episode_return": lambda curriculum: curriculum._get_episode_return()
        }
        if self.metric_name in predefined_metrics:
            metric_fn = predefined_metrics[self.metric_name]
        elif self.metric_name in self.custom_metrics:
            metric_fn = self.custom_metrics[self.metric_name]
        else:
            raise ValueError(f"Invalid metric name: {self.metric_name}")

        metric_value = metric_fn(curriculum)
        if self.comparator == '<':
            return metric_value < self.value
        elif self.comparator == '>':
            return metric_value > self.value
        elif self.comparator == '<=':
            return metric_value <= self.value
        elif self.comparator == '>=':
            return metric_value >= self.value
        elif self.comparator == '==':
            return metric_value == self.value
        else:
            raise ValueError(f"Invalid comparator: {self.comparator}")

    def __and__(self, other):
        if isinstance(other, Condition):
            return CompositeCondition([self, other], all)
        elif isinstance(other, CompositeCondition):
            return CompositeCondition([self] + other.conditions, all)
        else:
            raise ValueError("Can only combine Condition with Condition or CompositeCondition")

    def __or__(self, other):
        if isinstance(other, Condition):
            return CompositeCondition([self, other], any)
        elif isinstance(other, CompositeCondition):
            return CompositeCondition([self] + other.conditions, any)
        else:
            raise ValueError("Can only combine Condition with Condition or CompositeCondition")


class CompositeCondition:
    def __init__(self, conditions: List[Callable], operation: Callable):
        self.conditions = conditions
        self.operation = operation

    def __call__(self, curriculum):
        return self.operation(cond(curriculum) for cond in self.conditions)

    def __and__(self, other):
        if isinstance(other, Condition):
            return CompositeCondition(self.conditions + [other], all)
        elif isinstance(other, CompositeCondition):
            return CompositeCondition(self.conditions + other.conditions, all)
        else:
            raise ValueError("Can only combine CompositeCondition with Condition or CompositeCondition")

    def __or__(self, other):
        if isinstance(other, Condition):
            return CompositeCondition(self.conditions + [other], any)
        elif isinstance(other, CompositeCondition):
            return CompositeCondition(self.conditions + other.conditions, any)
        else:
            raise ValueError("Can only combine CompositeCondition with Condition or CompositeCondition")


class SequentialCurriculum(Curriculum):
    REQUIRES_STEP_UPDATES = False
    REQUIRES_EPISODE_UPDATES = True
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, curriculum_list: List[Curriculum], stopping_conditions: List[Any], *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        assert len(curriculum_list) > 0, "Must provide at least one curriculum"
        assert len(stopping_conditions) == len(curriculum_list) - 1, f"Stopping conditions must be one less than the number of curricula. Final curriculum is used for the remainder of training. Expected {len(curriculum_list) - 1}, got {len(stopping_conditions)}."
        if len(curriculum_list) == 1:
            warnings.warn("Your sequential curriculum only contains one element. Consider using that element directly instead.")

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
        """ Parse the stopping conditions to ensure that all items are callable conditions. """
        parsed_list = []
        for item in stopping_conditions:
            if isinstance(item, Callable):
                parsed_list.append(item)
            elif isinstance(item, Condition):
                parsed_list.append(self._parse_condition(item))
            else:
                raise ValueError(f"Invalid stopping condition: {item}")

        return parsed_list

    def _parse_condition(self, condition: Condition) -> Callable:
        return condition

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
        
        if self._should_use_startup_sampling():
            self.startup_sampled_tasks += curriculum.startup_sampled_tasks
        
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

    def update_on_step(self, obs, rew, term, trunc, info, env_id=None):
        if self.current_curriculum.requires_step_updates:
            self.current_curriculum.update_on_step(obs, rew, term, trunc, info, env_id)

    def update_on_step_batch(self, step_results, env_id=None):
        if self.current_curriculum.requires_step_updates:
            self.current_curriculum.update_on_step_batch(step_results, env_id)

    def update_on_demand(self, metrics):
        self.current_curriculum.update_on_demand(metrics)

    def update_task_progress(self, task, progress, env_id=None):
        self.current_curriculum.update_task_progress(task, progress, env_id)

    def check_stopping_conditions(self):
        if self._curriculum_index < len(self.stopping_conditions) and self.stopping_conditions[self._curriculum_index](self):
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
