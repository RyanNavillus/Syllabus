from syllabus.curricula import LevelSampler
from syllabus.core import Curriculum
from typing import Any, Callable, Dict, List, Union, Tuple


class PrioritizedLevelReplay(Curriculum):
    def __init__(self, level_sampler_args, *curriculum_args, level_sampler_kwargs, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.level_sampler = LevelSampler(*level_sampler_args, **level_sampler_kwargs)

    def _complete_task(self, task: Any, success_prob: float) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        self.completed_tasks += 1

    def _on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        raise NotImplementedError("Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def _on_step_batch(self, step_results: List[Tuple[int, int, int, int]]) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        for step_result in step_results:
            self.on_step(*step_result)

    def _on_episode(self, episode_return: float, trajectory: List = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        raise NotImplementedError("Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def _on_demand(self, metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        self.level_sampler.update_with_rollouts(metrics["rollouts"])

    def update_curriculum(self, update_data: Dict):
        """
        Update the curriculum with the specified update type.
        """
        update_type = update_data["update_type"]
        args = update_data["metrics"]

        if update_type == "step":
            self._on_step(*args)
        elif update_type == "step_batch":
            self._on_step_batch(*args)
        elif update_type == "episode":
            self._on_episode(*args)
        elif update_type == "demand":
            self._on_demand(*args)
        elif update_type == "complete":
            self._complete_task(*args)
        elif update_type == "noop":
            # Used to request tasks from the synchronization layer
            pass
        else:
            raise NotImplementedError(f"Update type {update_type} not implemented.")

    def batch_update_curriculum(self, update_data: List[Dict]):
        """
        Update the curriculum with the specified update type.
        """
        for update in update_data:
            self.update_curriculum(update)

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self.level_sampler.sample_weights()

    def sample(self, k: int = 1) -> Union[List, Any]:
        return [self.level_sampler.sample() for k in range(k)]
