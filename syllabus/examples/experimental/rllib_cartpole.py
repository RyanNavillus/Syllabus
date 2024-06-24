import wandb
import ray
import gymnasium as gym
from gymnasium.spaces import Box
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb, _clean_log, _QueueItem
from ray.tune.registry import register_env
from syllabus.core import RaySyncWrapper, make_ray_curriculum
from syllabus.curricula import SimpleBoxCurriculum, NoopCurriculum
from syllabus.task_space import TaskSpace

from syllabus.examples.task_wrappers import CartPoleTaskWrapper
from ray.tune.logger import LoggerCallback


class CustomWandbLoggerCallback(WandbLoggerCallback):
    def __init__(self, curriculum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum = curriculum

    def log_trial_result(self, iteration: int, trial: "Trial", result):
        if trial not in self._trial_logging_actors:
            self.log_trial_start(trial)

        result = _clean_log(result)
        self._trial_queues[trial].put((_QueueItem.RESULT, result))
        # full_range = ray.get(self.curriculum.curriculum.get_remote_attr.remote("max_range"))
        # self._trial_queues[trial].put((_QueueItem.RESULT, {"range_min": full_range[0]},))
        # self._trial_queues[trial].put((_QueueItem.RESULT, {"range_max": full_range[1]},))
        # self._trial_queues[trial].put((_QueueItem.RESULT, {"range_size": full_range[1] - full_range[0]},))

        self._trial_queues[trial].put((_QueueItem.RESULT, {"range_min": -0.3},))
        self._trial_queues[trial].put((_QueueItem.RESULT, {"range_max": 0.3},))
        self._trial_queues[trial].put((_QueueItem.RESULT, {"range_size": 0.6},))


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
    # curriculum = SimpleBoxCurriculum(task_space)
    curriculum = NoopCurriculum((-0.3, 0.3), task_space)

    # Add the curriculum sync wrapper
    curriculum = make_ray_curriculum(curriculum)

    config = {
        "env": "task_cartpole",
        "num_gpus": 1,
        "num_workers": 8,
        "framework": "torch",
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            callbacks=[CustomWandbLoggerCallback(curriculum, project="syllabus", group="Noop Curriculum")],
            stop={"timesteps_total": 200000},
            name="Box Curriculum",
        ),
    )
    results = tuner.fit()
