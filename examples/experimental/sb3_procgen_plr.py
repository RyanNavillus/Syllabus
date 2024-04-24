from typing import Callable

import gym
import procgen  # noqa: F401
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (DummyVecEnv, VecMonitor,
                                              VecNormalize)
from syllabus.core import (MultiProcessingSyncWrapper,
                           make_multiprocessing_curriculum)
from syllabus.curricula import CentralizedPrioritizedLevelReplay
from syllabus.examples.task_wrappers import ProcgenTaskWrapper
from wandb.integration.sb3 import WandbCallback


def make_env(task_queue, update_queue, start_level=0, num_levels=1):
    def thunk():
        env = gym.make("procgen-bigfish-v0", distribution_mode="easy", start_level=start_level, num_levels=num_levels)
        env = ProcgenTaskWrapper(env)
        env = MultiProcessingSyncWrapper(
            env,
            task_queue,
            update_queue,
            update_on_step=False,
            task_space=env.task_space,
        )
        return env
    return thunk


def wrap_vecenv(vecenv):
    vecenv.is_vector_env = True
    vecenv = VecMonitor(venv=vecenv, filename=None)
    vecenv = VecNormalize(venv=vecenv, norm_obs=False, norm_reward=True)
    return vecenv


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, curriculum, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum

    def _on_step(self) -> bool:
        tasks = self.training_env.venv.venv.venv.get_attr("task")

        update = {
            "update_type": "on_demand",
            "metrics": {
                "value": self.locals["values"],
                "next_value": self.locals["values"],
                "rew": self.locals["rewards"],
                "dones": self.locals["dones"],
                "tasks": tasks,
            },
        }
        self.curriculum.update_curriculum(update)
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


run = wandb.init(
    project="sb3",
    entity="ryansullivan",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)


sample_env = gym.make("procgen-bigfish-v0")
sample_env = ProcgenTaskWrapper(sample_env)
curriculum = CentralizedPrioritizedLevelReplay(sample_env.task_space, num_processes=64, num_steps=256)
curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)
venv = DummyVecEnv(
    [
        make_env(task_queue, update_queue, num_levels=0)
        for i in range(64)
    ]
)
venv = wrap_vecenv(venv)

model = PPO(
    "CnnPolicy",
    venv,
    verbose=1,
    n_steps=256,
    learning_rate=linear_schedule(0.0005),
    gamma=0.999,
    gae_lambda=0.95,
    n_epochs=3,
    clip_range_vf=0.2,
    ent_coef=0.01,
    batch_size=256 * 64,
    tensorboard_log="runs/testing"
)

wandb_callback = WandbCallback(
    model_save_path=f"models/{run.id}",
    verbose=2,
)
plr_callback = CustomCallback(curriculum)
callback = CallbackList([wandb_callback, plr_callback])
model.learn(
    25000000,
    callback=callback,
)
