import gym


class TaskWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_completion = 0.0
        self.task_space = None
        self.task = None

    def reset(self, *args, **kwargs):
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            self.change_task(new_task)
            # TODO: Handle failure case for change task
            self.task = new_task
        return self.observation(super().reset(*args, **kwargs))

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

    def _task_completion(self, obs, rew, done, info) -> float:
        """
        Implement this function to indicate whether the selected task has been completed.
        This can be determined using the observation, rewards, done, info or internal values
        from the environment. Intended to be used for automatic curricula.
        Returns a boolean or float value indicating binary completion or scalar degree of completion.
        """
        return 1.0 if done else 0.0

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
        """
        # Add goal to observation
        goal_encoding = self._encode_goal()
        if goal_encoding:
            observation['goal'] = goal_encoding

        return observation

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # Determine completion status of the current task
        self.task_completion = self._task_completion(obs, rew, done, info)
        info["task_completion"] = self.task_completion

        return self.observation(obs), rew, done, info
