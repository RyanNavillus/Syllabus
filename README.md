# Syllabus

Syllabus is an API for designing curricula for reinforcement learning agents, as well as a framework for synchronizing those curricula across environments running in multiple processes. it currently has support for environments run with Ray actors and Python multiprocessing, which should include RL libraries such as RLLib, CleanRL, Stable Baselines 3, and TorchBeast. We currently have working examples with **CleanRL**, **RLLib**, and **Torchbeast**.



## How it works

Syllabus uses a bidirectional sender-receiver model where the curricula sends tasks and receives environment outputs, while the environment receives tasks and sends outputs. The environment can use the provided task and the curriculum can use the outputs to update its task distribution. Adding this functionality to existing RL training code requires only a few additions.

To use syllabus for your curriculum learning project, you need:

* A curriculum that subclasses `Curriculum` or follows its API
* An environment that supports multiple tasks
* A wrapper that subclasses `TaskWrapper` allowing you to set a new task on `reset()`
* Learning code that uses ray actors or python multiprocessing to parallelize environments

All of the global coordination is handled automatically by Syllabus's synchronization wrappers.

## Example

This is a simple example of using Syllabus to synchronize a curriculum for CartPole using RLLib. CartPole doesn't normally support multiple tasks so we make a slight modification, allowing us to change the initialization range for the cart (the range from which the cart's initial location is selected). We also implement a `SimpleBoxCurriculum` which increases the initialization range whenever a specific reward threshold is met. We can use the `TaskWrapper` class to implement this new functionality for CartPole and to allow us to change the task on `reset()`.

```python
from syllabus import TaskWrapper


class CartPoleTaskWrapper(TaskWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.task = (-0.02, 0.02)
        self.total_reward = 0

    def reset(self, *args, **kwargs):
        self.env.reset()
        self.total_reward = 0
        if "new_task" in kwargs:
            new_task = kwargs.pop("new_task")
            self.change_task(new_task)
        return np.array(self.env.state, dtype=np.float32)

    def change_task(self, new_task):
        low, high = new_task
        self.env.state = self.env.np_random.uniform(low=low, high=high, size=(4,))
        self.task = new_task

    def _task_completion(self, obs, rew, done, info) -> float:
        # Return percent of optimal reward
        self.total_reward += rew
        return self.total_reward / 500.0
```



We can train an agent for this environment with the following code:

```python
import gym
import ray
import numpy as np
from ray.tune.registry import register_env
from ray import tune


def env_creator(config):
    env = gym.make("CartPole-v1")
    return CartPoleTaskWrapper(env)


ray.init()
register_env("task_cartpole", env_creator)

config = {
        "env": "task_cartpole",
        "num_gpus": 1,
        "num_workers": 16,
        "framework": "torch",
}

tuner = tune.Tuner("APEX", param_space=config)
results = tuner.fit()
```



With a few modifications, we can train this agent with a curriculum that's globally synchronized across multiple parallel environments.

```python
import gym
import ray
import numpy as np
from ray.tune.registry import register_env
from ray import tune

# Additional imports
from gym.spaces import Box
from syllabus import RaySyncWrapper, RayCurriculumWrapper
from curricula import SimpleBoxCurriculum


def env_creator(config):
    env = gym.make("CartPole-v1")
    env = CartPoleTaskWrapper(env)
    # Here we wrap the environment in a wrapper to receive new tasks
    # and send updates to the curriculum
    return RaySyncWrapper(env,
                          default_task=(-0.02, 0.02),
                          task_space=Box(-0.3, 0.3, shape=(2,)),
                          update_on_step=False)


ray.init()
register_env("task_cartpole", env_creator)

# We create a curriculum to server new tasks to the environment
curriculum = RayCurriculumWrapper(SimpleBoxCurriculum, task_space=Box(-0.3, 0.3, shape=(2,)))

config = {
        "env": "task_cartpole",
        "num_gpus": 1,
        "num_workers": 8,
        "framework": "torch",
}

tuner = tune.Tuner("APEX", param_space=config)
results = tuner.fit()
```

As you can see, we just wrap the task-enabled CartPole environment with a `RaySyncWrapper`, and create a curriculum with the `RayCurriculumWrapper`. They automatically communicate with each other to sample tasks from your curriculum and use them in the environments. That's it! Now you can implement as many curricula as you want, and as long as they follow the `Curriculum` API, you can hot-swap them in this code.


## Task Spaces
Syllabus uses task spaces to define valid ranges for tasks and simplify some logic. These are simply [Gym spaces](https://gymnasium.farama.org/api/spaces/) which support a majority of existing curriculum methods. For now, the code thoroughly supports Discrete and MultiDiscrete spaces with preliminary support for Box spaces. The task space is typically determined by the environment and limits the type of curriculum that you can use. Most curricula support either a discrete set of tasks or a continuous space of tasks.  


## Curriculum API

```python
class Curriculum:
    """
    Base class and API for defining curricula to interface with Gym environments.
    """
    def __init__(self, task_space: gym.Space, random_start_tasks: int = 0, use_wandb: bool = False) -> None:

    @property
    def _n_tasks(self, task_space: gym.Space = None) -> int:
        """
        Return the number of discrete tasks in the task_space.
        Returns None for continuous spaces.
        """

    @property
    def _tasks(self, task_space: gym.Space = None, sample_interval: float = None) -> List[tuple]:
        """
        Return the full list of discrete tasks in the task_space.
        Return a sample of the tasks for continuous spaces if sample_interval is specified.
        Can be overridden to exclude invalid tasks within the space.
        """

    def complete_task(self, task: typing.Any, success_prob: float) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """

    def on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """

    def on_step_batch(self, step_results: List[typing.Tuple[int, int, int, int]]) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """

    def log_task_dist(self, task_dist: List[float], check_dist=True) -> None:
        """
        Log the task distribution to wandb.

        Paramaters:
            task_dist: List of task probabilities. Must be a valid probability distribution.
        """
```



## Task-Enabled Environment API

```python
class TaskWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):

    def reset(self, *args, **kwargs):
    	"""
    	Changes the current task and resets the environment.
    	Calls the change_task function whenever "new_task" is passed in kwargs.
    	Also uses self.observation to add a goal encoding.
    	"""

    def change_task(self, new_task):
        """
        Changes the task of the existing environment to the new_task.

        Each environment will implement tasks differently. The easiest system would be to call a
        function or set an instance variable to change the task.

        Some environments may need to be reset or even reinitialized to change the task.
        If you need to reset or re-init the environment here, make sure to check
        that it is not in the middle of an episode to avoid unexpected behavior.
        """

    def _task_completion(self, obs, rew, done, info) -> float:
        """
        Implement this function to indicate whether the selected task has been completed.
        This can be determined using the observation, rewards, done, info or internal values
        from the environment. Intended to be used for automatic curricula.
        Returns a boolean or float value indicating binary completion or scalar degree of completion.
        """

    def _encode_goal(self):
        """
        Implement this method to indicate which task is selected to the agent.
        Returns: Numpy array encoding the goal.
        """

    def observation(self, observation):
        """
        Adds the goal encoding to the observation.
        Override to add additional task-specific observations.
        Returns a modified observation.
        """

    def step(self, action):
    	"""
    	Forwards the step action to the inner env.
    	Also adds the result of self.task_completion to info["task_completion"]
    	which is sent to the curriculum.
        """
```

