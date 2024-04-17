import warnings
import numpy as np
from syllabus.task_space import TaskSpace
from typing import Callable
from gymnasium.spaces import Discrete #, MultiDiscrete?
import json
import os

class StatRecorder:
    """
    Individual stat tracking for each task.
    """

    def __init__(self, task_space: TaskSpace, task_names: Callable = None, calc_past_N=None):
        """Initialize the StatRecorder"""

        self.task_space = task_space
        self.task_names = task_names
        self.calc_past_N = calc_past_N

        assert isinstance(self.task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        assert isinstance(self.task_space.gym_space, Discrete), f"Only Discrete task spaces are supported. Got {type(task_space.gym_space)}"

        self.tasks = self.task_space.get_tasks()
        self.num_tasks = self.task_space.num_tasks

        self.records = {task: [] for task in self.tasks}
        self.stats = {task: {} for task in self.tasks}

    def record(self, episode_return: float, episode_length: int, episode_task, env_id=None):
        """
        Records the length and return of an episode for a given task.

        :param episode_task: Identifier for the task
        :param episode_length: Length of the episode, i.e. the total number of steps taken during the episode
        :param episodic_return: Total return for the episode
        """

        if episode_task in self.tasks:
            if self.calc_past_N:
                self.records[episode_task].append({
                    "r": episode_return,
                    "l": episode_length,
                    "env_id": env_id
                })
                self.records[episode_task] = self.records[episode_task][-self.calc_past_N:]
                self.stats[episode_task]['mean_r'] = np.mean([record["r"] for record in self.records[episode_task]])
                self.stats[episode_task]['var_r'] = np.var([record["r"] for record in self.records[episode_task]])
                self.stats[episode_task]['mean_l'] = np.mean([record["l"] for record in self.records[episode_task]])
                self.stats[episode_task]['var_l'] = np.var([record["l"] for record in self.records[episode_task]])
                # only save mean/variance the past N episodes
            else:
                # save the mean/variance of all the episodes
                if 'mean_r' not in self.stats[episode_task].keys():
                    # the first episode for a task
                    self.stats[episode_task]['mean_r'] = episode_return
                    self.stats[episode_task]['mean_r_squared'] = episode_return ** 2
                    self.stats[episode_task]['var_r'] = 0
                    self.stats[episode_task]['mean_l'] = episode_length
                    self.stats[episode_task]['mean_l_squared'] = episode_length ** 2
                    self.stats[episode_task]['var_l'] = 0
                else:
                    N_past = len(self.records[episode_task])
                    
                    self.stats[episode_task]['mean_r'] =round((self.stats[episode_task]['mean_r'] * N_past + episode_return) / (N_past + 1), 4)
                    self.stats[episode_task]['mean_r_squared'] = round((self.stats[episode_task]['mean_r_squared'] * N_past + episode_return ** 2) / (N_past + 1), 4)
                    self.stats[episode_task]['var_r'] = round(self.stats[episode_task]['mean_r_squared'] - self.stats[episode_task]['mean_r'] ** 2, 4)
                    
                    self.stats[episode_task]['mean_l'] = round((self.stats[episode_task]['mean_l'] * N_past + episode_length) / (N_past + 1), 4)
                    self.stats[episode_task]['mean_l_squared'] = round((self.stats[episode_task]['mean_l_squared'] * N_past + episode_length ** 2) / (N_past + 1), 4)
                    self.stats[episode_task]['var_l'] = round(self.stats[episode_task]['mean_l_squared'] - self.stats[episode_task]['mean_l'] ** 2, 4)
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
            if self.task_names:
                for idx in tasks_to_log:
                    if self.stats[idx]:
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_return_mean", self.stats[idx]['mean_r'], step)
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_return_var", self.stats[idx]['var_r'], step)
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_length_mean", self.stats[idx]['mean_l'], step)
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_length_var", self.stats[idx]['var_l'], step)
                    else:
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_return_mean", 0, step)
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_return_var", 0, step)
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_length_mean", 0, step)
                        writer.add_scalar(f"stats_per_task/task_{self.task_space.task_name(idx)}_episode_length_var", 0, step)
            else:
                for idx in tasks_to_log:
                    if self.stats[idx]:
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_return_mean", self.stats[idx]['mean_r'], step)
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_return_var", self.stats[idx]['var_r'], step)
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_length_mean", self.stats[idx]['mean_l'], step)
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_length_var", self.stats[idx]['var_l'], step)
                    else:
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_return_mean", 0, step)
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_return_var", 0, step)
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_length_mean", 0, step)
                        writer.add_scalar(f"stats_per_task/task_{idx}_episode_length_var", 0, step)
        except ImportError:
            warnings.warn("Wandb is not installed. Skipping logging.")
        except wandb.errors.Error:
            # No need to crash over logging :)
            warnings.warn("Failed to log curriculum stats to wandb.")
    
    def output_results(self, output_path):
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