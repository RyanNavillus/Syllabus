import gymnasium as gym
import pettingzoo


class TaskEnv(gym.Env):
    # TODO: Update to new TaskSpace API
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_completion = 0.0
        self.task_space = None
        self.task = None

    def reset(self, *args, **kwargs):
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            self.change_task(new_task)
            self.task = new_task

        obs, info = super().reset(*args, **kwargs)
        return self.observation(obs), info

    def change_task(self, new_task):
        """
        Changes the task of the existing environment to the new_task.

        Each environment will implement tasks differently. The easiest system would be to call a
        function or set an instance variable to change the task.

        Some environments may need to be reset or even reinitialized to change the task.
        If you need to reset or re-init the environment here, make sure to check
        that it is not in the middle of an episode to avoid unexpected behavior.
        """
        raise NotImplementedError

    def _task_completion(self, obs, rew, term, trunc, info) -> float:
        """
        Implement this function to indicate whether the selected task has been completed.
        This can be determined using the observation, rewards, term, trunc, info or internal values
        from the environment. Intended to be used for automatic curricula.
        Returns a boolean or float value indicating binary completion or scalar degree of completion.
        """
        return 1.0 if term or trunc else 0.0

    def _encode_goal(self):
        """
        Implement this method to indicate which task is selected to the agent.
        Returns: Numpy array encoding the goal.
        """
        return None

    def observation(self, observation):
        """
        Adds the goal encoding to the observation.
        Override to add additional task-specific observations.
        Returns a modified observation.
        TODO: Complete this implementation and find way to support centralized encodings
        """
        # Add goal to observation
        goal_encoding = self._encode_goal()
        if goal_encoding is not None:
            observation['goal'] = goal_encoding

        return observation

    def step(self, action):
        """
        Steps the environment with the given action.
        Unlike the typical Gym environment, this method should also add the
        {"task_completion": self.task_completion()} key to the info dictionary
        to support curricula that rely on this metric.
        """
        raise NotImplementedError


class PettingZooTaskEnv(pettingzoo.ParallelEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_completion = 0.0
        self.task_space = None
        self.task = None

    def get_current_task(self):
        return self.task

    def reset(self, *args, **kwargs):
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            self.change_task(new_task)
            self.task = new_task

        obs, info = super().reset(*args, **kwargs)
        return self.observation(obs), info

    def change_task(self, new_task):
        """
        Changes the task of the existing environment to the new_task.

        Each environment will implement tasks differently. The easiest system would be to call a
        function or set an instance variable to change the task.

        Some environments may need to be reset or even reinitialized to change the task.
        If you need to reset or re-init the environment here, make sure to check
        that it is not in the middle of an episode to avoid unexpected behavior.
        """
        raise NotImplementedError

    def _task_completion(self, obs, rew, term, trunc, info) -> float:
        """
        Implement this function to indicate whether the selected task has been completed.
        This can be determined using the observation, rewards, term, trunc, info or internal values
        from the environment. Intended to be used for automatic curricula.
        Returns a boolean or float value indicating binary completion or scalar degree of completion.
        """
        return 1.0 if term or trunc else 0.0

    def _encode_goal(self):
        """
        Implement this method to indicate which task is selected to the agent.
        Returns: Numpy array encoding the goal.
        """
        return None

    def observation(self, observation):
        """
        Adds the goal encoding to the observation.
        Override to add additional task-specific observations.
        Returns a modified observation.
        TODO: Complete this implementation and find way to support centralized encodings
        """
        # Add goal to observation
        goal_encoding = self._encode_goal()
        if goal_encoding is not None:
            observation['goal'] = goal_encoding

        return observation

    def step(self, action):
        """
        Steps the environment with the given action.
        Unlike the typical Gym environment, this method should also add the
        {"task_completion": self.task_completion()} key to the info dictionary
        to support curricula that rely on this metric.
        """
        raise NotImplementedError
