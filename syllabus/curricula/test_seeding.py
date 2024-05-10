from syllabus.core import Curriculum
import gymnasium as gym

from syllabus.task_space import TaskSpace
from domain_randomization import DomainRandomization
from learning_progress import LearningProgressCurriculum
# from simple_box import SimpleBoxCurriculum
# from annealing_box import AnnealingBoxCurriculum
# from syllabus.curricula.plr import CentralizedPrioritizedLevelReplay
# from syllabus.curricula.plr import PrioritizedLevelReplay
# from syllabus.curricula.plr import TaskSampler

#1: DomainRandomization with seed
task_space = TaskSpace(200)
seed = 3
c = DomainRandomization(task_space = task_space, seed = seed)
sample = c.sample()
for i in range(5):
    next_sample = c.sample()
    assert sample == next_sample, f"Expected all samples to be the same, got {sample} and {next_sample}"
    sample = next_sample

print("DomainRandomization with seed! SUCCESSFUL")

#2: DomainRandomization without seed
task_space = TaskSpace(200)
c = DomainRandomization(task_space = task_space)
sample = c.sample()
for i in range(5):
    next_sample = c.sample()
    assert sample != next_sample, f"Expected all samples to be different, got {sample} and {next_sample}"
    sample = next_sample

print("DomainRandomization without seed! SUCCESSFUL")


#3: LearningProgressCurriculum with seed
task_space = TaskSpace(200)
seed = 5
c = LearningProgressCurriculum(task_space = task_space, seed = seed)
sample = c.sample()
for i in range(5):
    next_sample = c.sample()
    assert sample == next_sample, f"Expected all samples to be the same, got {sample} and {next_sample}"
    sample = next_sample

print("LearningProgressCurriculum with seed! SUCCESSFUL")

#4: LearningProgressCurriculum without seed
task_space = TaskSpace(200)
c = LearningProgressCurriculum(task_space = task_space)
sample = c.sample()
for i in range(5):
    next_sample = c.sample()
    assert sample != next_sample, f"Expected all samples to be different, got {sample} and {next_sample}"
    sample = next_sample

print("LearningProgressCurriculum without seed! SUCCESSFUL")

#4: SequentialCurriculum with seed
# task_space = TaskSpace(200)
# c = SequentialCurriculum(task_space = task_space, seed = seed, curriculum_list = list, stopping_conditions = [])
# sample = c.sample()
# for i in range(5):
#     next_sample = c.sample()
#     assert sample == next_sample, f"Expected all samples to be same, got {sample} and {next_sample}"
#     sample = next_sample

# print("SequentialCurriculum with seed! SUCCESSFUL")

#5 SimpleBoxCurriculum and AnnealingBoxCurriculum with seed
# task_space = TaskSpace(gym.spaces.Box(low=0, high=1, shape=(2,)), [(0, 0), (0, 1), (1, 0), (1, 1)])
# seed = 3 

# listb = [SimpleBoxCurriculum(task_space = task_space, seed = seed),
# AnnealingBoxCurriculum(task_space = task_space, seed = seed, start_values = [1,2], end_values = [1,5], total_steps = 1),
# ]  

# sample_list = [listb[0].sample(), listb[1].sample()]
# for i in range(5):
#     next_sample_0 = listb[0].sample()
#     next_sample_1 = listb[1].sample()

#     assert sample_list[0] == next_sample_0 , f'Expected all samples to be same, got {str(sample_list[0])} and {next_sample_0}'
#     assert sample_list[1] == next_sample_1 , f'Expected all samples to be same, got {sample_list[1]} and {next_sample_1}'
#     sample_list[0] = next_sample_0
#     sample_list[1] = next_sample_1

# print("Its interesting to see that given these arbitrary values the sample returns the same value over 5 iterations")
# Its interesting to see that given these arbitrary values the sample returns the same value over 5 iterations
