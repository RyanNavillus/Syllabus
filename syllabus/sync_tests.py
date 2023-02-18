
import time
import random
from multiprocessing import SimpleQueue

import ray
from nle.env.tasks import NetHackScore

import curriculum
from curriculum import (MultiProcessingSyncWrapper,
                        RaySyncWrapper,
                        LearningProgressCurriculum,
                        RayCurriculumWrapper,
                        NestedRayCurriculumWrapper,
                        MultiProcessingCurriculumWrapper)
from nethack_le import NethackTaskWrapper

N_ENVS = 2
N_EPISODES = 50


def create_nethack_env(sample_queue, complete_queue):
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = MultiProcessingSyncWrapper(env, sample_queue, complete_queue, default_task=0, task_space=env.task_space)
    return env


def run_episode(env):
    obs = env.reset()
    done = False
    ep_rew = 0
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        ep_rew += rew
    return ep_rew


def run_episodes(sample_queue, complete_queue):
    env = create_nethack_env(sample_queue, complete_queue)
    ep_rews = []
    for i in range(N_EPISODES):
        # if i % 10 == 0:
        #     distribution, task_lps_standardized = curriculum.unwrapped._sample_distribution()
        #     print(distribution, task_lps_standardized)
        #     x_axis = np.linspace(-3, 3, num=len(task_lps_standardized))
        #     plt.plot(x_axis, task_lps_standardized, color="blue", label="Standardized LP")
        #     plt.plot(x_axis, distribution, color="orange", label="Sampling distribution weight")
        #     plt.xlabel('Z-scored distributed learning progress')
        #     plt.legend()
        #     plt.show()
        ep_rews.append(run_episode(env))


def create_nethack_env_ray():
    env = NetHackScore()
    env = NethackTaskWrapper(env)
    env = RaySyncWrapper(env, update_on_step=False, default_task=0, task_space=env.task_space)
    return env


@ray.remote
def run_episodes_ray():
    env = create_nethack_env_ray()
    ep_rews = []
    for _ in range(N_EPISODES):
        ep_rews.append(run_episode(env))


if __name__ == "__main__":
    # Test single process
    sample_queue = SimpleQueue()
    complete_queue = SimpleQueue()
    sample_env = create_nethack_env(sample_queue, complete_queue)
    curriculum = LearningProgressCurriculum(sample_env.task_space, random_start_tasks=10)
    curriculum = MultiProcessingCurriculumWrapper(curriculum,
                                                  sample_queue=sample_queue,
                                                  complete_queue=complete_queue,
                                                  task_space=sample_env.task_space)
    curriculum.start()
    del sample_env

    print("\nRunning single process test...")
    start = time.time()
    for _ in range(N_ENVS):
        run_episodes(sample_queue, complete_queue)
    end = time.time()
    print(f"Single process test passed: {end - start:.2f}s")

    # Test Queue multi process
    # print("\nRunning Python multi process test...")
    # start = time.time()
    # actors = []
    # for _ in range(N_ENVS):
    #     actors.append(Process(target=run_episodes, args=(sample_queue, complete_queue)))

    # for actor in actors:
    #     actor.start()
    # for actor in actors:
    #     actor.join()
    # end = time.time()
    # print(f"Multi process test passed: {end - start:.2f}s")


    # Test Ray multi process
    sample_env = NethackTaskWrapper(NetHackScore())
    #curriculum = LearningProgressCurriculum(sample_env.task_space, random_start_episodes=10)
    curriculum = RayCurriculumWrapper(LearningProgressCurriculum, sample_env.task_space, random_start_tasks=10)
    #curriculum = RandomStartWrapper(curriculum, start_steps=100)
    #curriculum = LearningProgressCurriculum(sample_env.task_space)
    #curriculum = RayCurriculumWrapper(curriculum)
    del sample_env

    print("\nRunning Ray multi process test...")
    start = time.time()
    remotes = []
    for _ in range(N_ENVS):
        remotes.append(run_episodes_ray.remote())
    ray.get(remotes)
    end = time.time()
    print(f"Multi process test passed: {end - start:.2f}s")
