import gymnasium as gym
import pettingzoo
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper


class TaskWrapper(gym.Wrapper):
    # TODO: Update to new TaskSpace API
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_completion = 0.0
        self.task_space = None
        self.task = None    # TODO: Would making this a property protect from accidental overriding?

    def reset(self, **kwargs):
        new_task = kwargs.pop("new_task", None)
        if new_task is not None:
            self.change_task(new_task)

        obs, info = self.env.reset(**kwargs)
        info["task_completion"] = 0.0
        info["task"] = self.task
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
        self.task = new_task

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
        obs, rew, term, trunc, info = self.env.step(action)

        # Determine completion status of the current task
        self.task_completion = self._task_completion(obs, rew, term, trunc, info)
        info["task_completion"] = self.task_completion
        info["task"] = self.task

        return self.observation(obs), rew, term, trunc, info

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr)
        if env_attr:
            return env_attr
        else:
            raise AttributeError(f"TaskWrapper and env do not have attribute {attr}")


class PettingZooTaskWrapper(BaseParallelWrapper):
    def __init__(self, env: pettingzoo.ParallelEnv):
        super().__init__(env)
        self.task = None

    @property
    def agents(self):
        return self.env.agents

    def __getattr__(self, attr):
        env_attr = getattr(self.env, attr, None)
        if env_attr:
            return env_attr

    def get_current_task(self):
        return self.current_task

    def reset(self, **kwargs):
        new_task = kwargs.pop("new_task", None)
        if new_task is not None:
            self.change_task(new_task)
            self.task = new_task
        obs, info = self.env.reset(**kwargs)
        for agent in info.keys():
            info[agent]["task_completion"] = 0.0
            info[agent]["task_id"] = self.task
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

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        # Determine completion status of the current task
        self.task_completion = self._task_completion(obs, rew, term, trunc, info)
        for agent in self.env.possible_agents:
            if agent not in info:
                info[agent] = {}
            info[agent]["task_completion"] = self.task_completion
            info[agent]["task_id"] = self.task
        return obs, rew, term, trunc, info

    def observation(self, observation):
        """
        Adds the goal encoding to the observation.
        Override to add additional task-specific observations.
        Returns a modified observation.
        TODO: Complete this implementation and find way to support centralized encodings
        TODO: Support PettingZoo environments
        TODO: Use TaskSpace for encodings?
        """
        # Add goal to observation
        goal_encoding = self._encode_goal()
        if goal_encoding is not None:
            observation['goal'] = goal_encoding

        return observation

    def _encode_goal(self):
        """
        Implement this method to indicate which task is selected to the agent.
        Returns: Numpy array encoding the goal.
        """
        return None

    def _task_completion(self, obs, rew, term, trunc, info) -> float:
        """
        Implement this function to indicate whether the selected task has been completed.
        This can be determined using the observation, rewards, term, trunc, info or internal values
        from the environment. Intended to be used for automatic curricula.
        Returns a boolean or float value indicating binary completion or scalar degree of completion.
        # TODO: Support PettingZoo environments
        """
        return 0.0
