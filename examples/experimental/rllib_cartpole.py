import gymnasium as gym
from gymnasium.spaces import Box
from ray import tune
from ray.tune.registry import register_env
from syllabus.core import RaySyncWrapper, make_ray_curriculum
from syllabus.curricula import SimpleBoxCurriculum
from syllabus.task_space import TaskSpace

from .task_wrappers import CartPoleTaskWrapper

# Define a task space
if __name__ == "__main__":
    task_space = TaskSpace(Box(-0.3, 0.3, shape=(2,)))

    def env_creator(config):
        env = gym.make("CartPole-v1")
        # Wrap the environment to change tasks on reset()
        env = CartPoleTaskWrapper(env)
        # Add environment sync wrapper
        env = RaySyncWrapper(
            env, task_space=task_space, update_on_step=False
        )
        return env

    register_env("task_cartpole", env_creator)

    # Create the curriculum
    curriculum = SimpleBoxCurriculum(task_space)
    # Add the curriculum sync wrapper
    curriculum = make_ray_curriculum(curriculum)

    config = {
        "env": "task_cartpole",
        "num_gpus": 1,
        "num_workers": 8,
        "framework": "torch",
    }

    tuner = tune.Tuner("APEX", param_space=config)
    results = tuner.fit()
