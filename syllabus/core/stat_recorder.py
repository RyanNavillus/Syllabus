import os
import json
import warnings
import numpy as np
from syllabus.task_space import TaskSpace
from gymnasium.spaces import Discrete
from collections import deque, defaultdict


class StatRecorder:
    """
    Individual statistics tracking for each task.
    """

    def __init__(self, task_space: TaskSpace, calc_past_n=None, task_names=None):
        """Initialize the StatRecorder"""

        self.task_space = task_space
        self.calc_past_n = calc_past_n
        self.task_names = task_names if task_names is not None else lambda task, idx: idx

        assert isinstance(self.task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        assert isinstance(self.task_space.gym_space, Discrete), f"Only Discrete task spaces are supported. Got {type(task_space.gym_space)}"

        self.tasks = self.task_space.get_tasks()
        self.num_tasks = self.task_space.num_tasks

        if self.calc_past_n is not None:
            self.episode_returns = {task: deque(maxlen=calc_past_n) for task in self.tasks}
            self.episode_lengths = {task: deque(maxlen=calc_past_n) for task in self.tasks}
            self.env_ids = {task: deque(maxlen=calc_past_n) for task in self.tasks}
        else:
            self.num_past_episodes = {task: 0 for task in self.tasks}

        self.stats = {task: defaultdict(float) for task in self.tasks}

    def record(self, episode_return: float, episode_length: int, episode_task, env_id=None):
        """
        Record the length and return of an episode for a given task.

        :param episode_length: Length of the episode, i.e. the total number of steps taken during the episode
        :param episodic_return: Total return for the episode
        :param episode_task: Identifier for the task
        """
        if episode_task in self.tasks:
            if self.calc_past_n is not None:
                self.episode_returns[episode_task].append(episode_return)
                self.episode_lengths[episode_task].append(episode_length)
                self.env_ids[episode_task].append(env_id)

                self.stats[episode_task]['mean_r'] = np.mean(self.episode_returns[episode_task])
                self.stats[episode_task]['var_r'] = np.var(self.episode_returns[episode_task])
                self.stats[episode_task]['mean_l'] = np.mean(self.episode_lengths[episode_task])
                self.stats[episode_task]['var_l'] = np.var(self.episode_lengths[episode_task])
            else:
                n_past = self.num_past_episodes[episode_task]
                self.num_past_episodes[episode_task] += 1

                self.stats[episode_task]['mean_r'] = (self.stats[episode_task]['mean_r'] * n_past + episode_return) / (n_past + 1)
                self.stats[episode_task]['mean_r_squared'] = (self.stats[episode_task]['mean_r_squared'] * n_past + episode_return ** 2) / (n_past + 1)
                self.stats[episode_task]['var_r'] = self.stats[episode_task]['mean_r_squared'] - self.stats[episode_task]['mean_r'] ** 2

                self.stats[episode_task]['mean_l'] = (self.stats[episode_task]['mean_l'] * n_past + episode_length) / (n_past + 1)
                self.stats[episode_task]['mean_l_squared'] = (self.stats[episode_task]['mean_l_squared'] * n_past + episode_length ** 2) / (n_past + 1)
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
            if len(self.tasks) > 10 and not log_full_dist:
                warnings.warn("Only logging stats for 5 tasks.")
                tasks_to_log = self.tasks[:10]
            log_data = []
            for idx in tasks_to_log:
                if len(self.stats[idx]) > 0:
                    name = self.task_names(list(self.task_space.tasks)[idx], idx)
                    log_data.append((f"stats_per_task/{name}_episode_return_mean", self.stats[idx]['mean_r'], step))
                    log_data.append((f"stats_per_task/{name}_episode_return_var", self.stats[idx]['var_r'], step))
                    log_data.append((f"stats_per_task/{name}_episode_length_mean", self.stats[idx]['mean_l'], step))
                    log_data.append((f"stats_per_task/{name}_episode_length_var", self.stats[idx]['var_l'], step))
            for name, prob, step in log_data:
                if writer == wandb:
                    writer.log({name: prob}, step=step)
                else:
                    writer.add_scalar(name, prob, step)
        except ImportError:
            warnings.warn("Wandb is not installed. Skipping logging.")
        except wandb.errors.Error:
            # No need to crash over logging :)
            warnings.warn("Failed to log curriculum stats to wandb.")

    def normalize(self, reward, task):
        """
        Normalize reward by task.
        """
        task_stats = self.stats[task]
        reward_mean = task_stats['mean_r']
        reward_std = np.sqrt(task_stats['var_r'])
        normalized_reward = deque(maxlen=reward.maxlen)
        for r in reward:
            normalized_reward.append((r - reward_mean) / max(0.01, reward_std))
        return normalized_reward

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
