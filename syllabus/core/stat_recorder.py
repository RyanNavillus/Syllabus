import os
import json
import warnings
import numpy as np
from syllabus.task_space import TaskSpace
from typing import Callable
from gymnasium.spaces import Discrete #, MultiDiscrete?
from collections import deque, defaultdict

class StatRecorder:
    """
    Individual stat tracking for each task.
    """

    def __init__(self, task_space: TaskSpace, keep_last_N=10, calc_past_N=None):
        """Initialize the StatRecorder"""

        self.task_space = task_space
        self.calc_past_N = calc_past_N

        assert isinstance(self.task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        assert isinstance(self.task_space.gym_space, Discrete), f"Only Discrete task spaces are supported. Got {type(task_space.gym_space)}"
        if calc_past_N and calc_past_N > keep_last_N:
            warnings.warn("The number of data points requested to calculate statistics exceeds the number of data points kept in memory. Will only use data points available in memory to calculate statistics instead.")

        self.tasks = self.task_space.get_tasks()
        self.num_tasks = self.task_space.num_tasks

        self.episode_returns = {task: deque(maxlen=keep_last_N) for task in self.tasks}
        self.episode_lengths = {task: deque(maxlen=keep_last_N) for task in self.tasks}
        self.env_ids = {task: deque(maxlen=keep_last_N) for task in self.tasks}
        self.stats = {task: defaultdict(float) for task in self.tasks}

    def record(self, episode_return: float, episode_length: int, episode_task, env_id=None):
        """
        Record the length and return of an episode for a given task.

        :param episode_task: Identifier for the task
        :param episode_length: Length of the episode, i.e. the total number of steps taken during the episode
        :param episodic_return: Total return for the episode
        """

        if episode_task in self.tasks:
            if self.calc_past_N:
                self.episode_returns[episode_task].append(episode_return)
                self.episode_lengths[episode_task].append(episode_length)
                self.env_ids[episode_task].append(env_id)

                self.stats[episode_task]['mean_r'] = np.mean(list(self.episode_returns[episode_task])[-self.calc_past_N:]) # I am not sure whether there is a more efficient way to slice to deque. I temperorily convert it to a list then slice it, which should cost O(n)
                self.stats[episode_task]['var_r'] = np.var(list(self.episode_returns[episode_task])[-self.calc_past_N:])
                self.stats[episode_task]['mean_l'] = np.mean(list(self.episode_lengths[episode_task])[-self.calc_past_N:])
                self.stats[episode_task]['var_l'] = np.var(list(self.episode_lengths[episode_task])[-self.calc_past_N:])
            else:
                # save the mean/variance of all the episodes
                N_past = len(self.episode_returns[episode_task])
                
                self.stats[episode_task]['mean_r'] = (self.stats[episode_task]['mean_r'] * N_past + episode_return) / (N_past + 1)
                self.stats[episode_task]['mean_r_squared'] = (self.stats[episode_task]['mean_r_squared'] * N_past + episode_return ** 2) / (N_past + 1)
                self.stats[episode_task]['var_r'] = self.stats[episode_task]['mean_r_squared'] - self.stats[episode_task]['mean_r'] ** 2
                
                self.stats[episode_task]['mean_l'] = (self.stats[episode_task]['mean_l'] * N_past + episode_length) / (N_past + 1)
                self.stats[episode_task]['mean_l_squared'] = (self.stats[episode_task]['mean_l_squared'] * N_past + episode_length ** 2) / (N_past + 1)
                self.stats[episode_task]['var_l'] = self.stats[episode_task]['mean_l_squared'] - self.stats[episode_task]['mean_l'] ** 2
        else:
            raise ValueError("Unknown task")
    
    def log_metrics(self, writer, step=None, log_full_dist=False):
        """Log the statistics of the first 5 tasks to the provided tensorboard writer.

        :param writer: Tensorboard summary writer.
        """
        try:
            import wandb
            tasks_to_log = self.tasks
            if len(self.tasks) > 5 and not log_full_dist:
                warnings.warn("Only logging stats for 5 tasks.")
                tasks_to_log = self.tasks[:5]
            for idx in tasks_to_log:
                if self.stats[idx]:
                    writer.add_scalar(f"stats_per_task/task_{idx}_episode_return_mean", self.stats[idx]['mean_r'], step)
                    writer.add_scalar(f"stats_per_task/task_{idx}_episode_return_var", self.stats[idx]['var_r'], step)
                    writer.add_scalar(f"stats_per_task/task_{idx}_episode_length_mean", self.stats[idx]['mean_l'], step)
                    writer.add_scalar(f"stats_per_task/task_{idx}_episode_length_var", self.stats[idx]['var_l'], step)
        except ImportError:
            warnings.warn("Wandb is not installed. Skipping logging.")
        except wandb.errors.Error:
            # No need to crash over logging :)
            warnings.warn("Failed to log curriculum stats to wandb.")
    
    def save_statistics(self, output_path):
        """
        Write task-specific statistics to file.
        """
        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()  # Use .item() to convert numpy types to native Python types
            raise TypeError
        stats = json.dumps(self.stats, default=convert_numpy)
        with open(os.path.join(output_path, 'task_specific_stats.json'), "w") as file:
            file.write(stats)