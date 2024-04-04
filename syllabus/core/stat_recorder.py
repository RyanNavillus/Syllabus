import numpy as np
from syllabus.task_space import TaskSpace
from gymnasium.spaces import Discrete #, MultiDiscrete?
import json
import os

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Use .item() to convert numpy types to native Python types
    raise TypeError

class StatRecorder:
    """
    Individual stat tracking for each task.
    """

    def __init__(self, task_space: TaskSpace):
        """Initialize the StatRecorder"""

        self.write_path = '/Users/allisonyang/Downloads'

        self.task_space = task_space

        assert isinstance(self.task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        assert isinstance(self.task_space.gym_space, Discrete), f"Only Discrete task spaces are supported. Got {type(task_space.gym_space)}"

        self.tasks = self.task_space.get_tasks()
        self.num_tasks = self.task_space.num_tasks

        self.records = {task: [] for task in self.tasks}
        self.stats = {task: {} for task in self.tasks}

    def record(self, episode_return: float, episode_length: int, episode_task, env_id=None):
        """
        Records the length and return of an episode for a given task.

        :param task: Identifier for the task
        :param episode_length: Length of the episode, i.e. the total number of steps taken during the episode
        :param episodic_return: Total return for the episode
        """

        if episode_task in self.tasks:
            self.records[episode_task].append({
                "r": episode_return,
                "l": episode_length,
                "env_id": env_id
            })
            self.stats[episode_task]['mean_r'] = np.mean([record["r"] for record in self.records[episode_task]])
            self.stats[episode_task]['var_r'] = np.var([record["r"] for record in self.records[episode_task]])
            self.stats[episode_task]['mean_l'] = np.mean([record["l"] for record in self.records[episode_task]])
            self.stats[episode_task]['var_l'] = np.var([record["l"] for record in self.records[episode_task]])
        else:
            raise ValueError("Unknown task")
        
        """
        records = json.dumps(self.records, default=convert_numpy)
        with open(os.path.join(self.write_path, 'records.json'), "w") as file:
            file.write(records)
        stats = json.dumps(self.stats, default=convert_numpy)
        with open(os.path.join(self.write_path, 'stats.json'), "w") as file:
            file.write(stats)
        """

    def get_task_return_avg(self, task):
        """Returns the average episode length for a given task."""
        return np.mean([record["r"] for record in self.records[task]])

    def get_task_return_sum(self, task):
        """Returns the total return for a given task."""
        return sum([record["r"] for record in self.records[task]])

    def get_task_return_variance(self, task):
        """Returns the variance of returns for a given task."""
        return np.var([record["r"] for record in self.records[task]])

    def get_task_return_std(self, task):
        """Returns the standard deviation of returns for a given task."""
        return np.std([record["r"] for record in self.records[task]])