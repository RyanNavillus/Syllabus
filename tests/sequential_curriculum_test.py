from copy import deepcopy
import pytest
from nle.env.tasks import NetHackEat, NetHackScore
from syllabus.core import make_multiprocessing_curriculum
from syllabus.curricula import SequentialCurriculum, NoopCurriculum, DomainRandomization
from syllabus.task_space import TaskSpace
from syllabus.tests.utils import create_nethack_env, run_native_multiprocess, run_single_process, run_set_length
from unittest.mock import patch
import logging

# Set up logging
logging.basicConfig(filename='test_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def create_env():
    return create_nethack_env

def run_curriculum(curriculum, env_fn):
    # Test single process speed
    native_speed = run_single_process(env_fn, curriculum=curriculum, num_envs=1, num_episodes=4)

def run_gymnasium_episode(env, new_task=None, curriculum=None, env_id=0):
    """Run a single episode of the environment."""
    if new_task is not None:
        obs = env.reset(new_task=new_task)
    else:
        obs = env.reset()
    
    term = trunc = False
    ep_rew = 0
    ep_len = 0

    task_completion = 0
    while not (term or trunc):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)

        task_completion += 1
        info['task_completion'] += task_completion
        if curriculum.custom_metrics:
            for name, func in curriculum.custom_metrics.items():
                curriculum.metric_values[name] = func(obs, info)
        
        logger.info(info)
        
        if curriculum and curriculum.requires_step_updates:
            curriculum.update_on_step(env.task_space.encode(env.task), obs, rew, term, trunc, info, env_id=env_id)
            curriculum.update_task_progress(env.task_space.encode(env.task), info["task_completion"], env_id=env_id)
        ep_rew += rew
        ep_len += 1

    if curriculum and curriculum.requires_episode_updates:
        curriculum.update_on_episode(ep_rew, ep_len, env.task_space.encode(env.task), env_id=env_id)
    return ep_rew

def test_custom_sequential_curriculum(create_env):
    env = create_env()
    curricula = []
    stopping = []

    # Custom metrics definition
    custom_metrics = {
        "my_custom_function_1": lambda obs, info: sum(sum(obs['glyphs'])) + info['task_completion'],
         "my_custom_function_2": lambda obs, info: info['task_completion'],
    }

    # Stage 1 - Survival
    stage1 = [0, 1, 2, 3]
    stopping.append("steps<100")

    # Stage 2 - Harvest Equipment
    stage2 = [4, 5]
    stopping.append("my_custom_function_1<0")

    # Stage 3 - Equip Weapons
    stage3 = [6, 7]

    curricula = [stage1, stage2, stage3]
    curriculum = SequentialCurriculum(curricula, stopping, env.task_space, custom_metrics=custom_metrics)

    with patch('syllabus.tests.utils.run_gymnasium_episode', side_effect=lambda env, new_task, curriculum, env_id: run_gymnasium_episode(env, new_task, curriculum, env_id)):
        run_curriculum(curriculum, create_env)

if __name__ == "__main__":
    test_custom_sequential_curriculum(create_nethack_env)
    logger.info("All tests passed!")
