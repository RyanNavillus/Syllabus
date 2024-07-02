from syllabus.core import Curriculum
import gymnasium as gym

from syllabus.task_space import TaskSpace
from domain_randomization import DomainRandomization
from learning_progress import LearningProgressCurriculum
from sequential import SequentialCurriculum
from syllabus.curricula.plr import CentralizedPrioritizedLevelReplay
from syllabus.curricula.plr import PrioritizedLevelReplay
from syllabus.curricula.plr import TaskSampler
import numpy as np

def seed_test(c: Curriculum):
    sample = c.sample()
    for i in range(5):
        next_sample = c.sample()
        print(next_sample)
        assert sample == next_sample, f"Expected all samples to be the same, got {sample} and {next_sample}"
        sample = next_sample

    return True

def no_seed_test(c: Curriculum):
    sample = c.sample()
    list = [int(sample[0])]
    for i in range(5):
        next_sample = c.sample()
        list.append(int(next_sample[0]))
        print(next_sample)
        sample = next_sample
    
    if(len(set(list))<=1) :
        raise Exception(f"Expected samples to variable, only one sample value {sample}")
    else :
        return True
        
#Seed Tests
task_space = TaskSpace(200)
# task_space = TaskSpace(gym.spaces.Box(low=0, high=1, shape=(2,)), [(0, 0), (0, 1), (1, 0), (1, 1)])
seed = 3

#1: DomainRandomization with seed
c = DomainRandomization(task_space = task_space, seed = seed)
if seed_test(c = c) :
    print("DomainRandomization with seed! SUCCESSFUL")

#2: DomainRandomization without seed
c = DomainRandomization(task_space = task_space)
if no_seed_test(c = c) :
    print("DomainRandomization without seed! SUCCESSFUL")


#3: LearningProgressCurriculum with seed
c = LearningProgressCurriculum(task_space = task_space, seed = seed)
if seed_test(c = c) :
    print("LearningProgressCurriculum with seed! SUCCESSFUL")

#4: LearningProgressCurriculum without seed
c = LearningProgressCurriculum(task_space = task_space)
if no_seed_test(c = c) :
    print("LearningProgressCurriculum without seed! SUCCESSFUL")

#5: SequentialCurriculum with seed
list = [LearningProgressCurriculum(task_space = task_space),DomainRandomization(task_space = task_space) ]
c = SequentialCurriculum(task_space = task_space, curriculum_list = list, stopping_conditions =  ["steps>1"], seed = seed) 
if seed_test(c = c) :
    print("SequentialCurriculum with seed! SUCCESSFUL")

#6: SequentialCurriculum without seed
list = [LearningProgressCurriculum(task_space = task_space),DomainRandomization(task_space = task_space) ]
c = SequentialCurriculum(task_space = task_space, curriculum_list = list, stopping_conditions =  ["steps>1"]) 
if no_seed_test(c = c) :
    print("SequentialCurriculum without seed! SUCCESSFUL")

#7 CentralizedPrioritizedLevelReplay with seed
c = CentralizedPrioritizedLevelReplay(task_space = task_space, seed = seed)
if seed_test(c = c) :
    print("CentralizedPrioritizedLevelReplay with seed! SUCCESSFUL")

#8 CentralizedPrioritizedLevelReplay without seed
c = CentralizedPrioritizedLevelReplay(task_space = task_space)
if no_seed_test(c = c) :
    print("CentralizedPrioritizedLevelReplay without seed! SUCCESSFUL")

#9 PrioritizedLevelReplay with seed
c = PrioritizedLevelReplay(task_space = task_space, observation_space = gym.spaces.Discrete(3), seed = seed)
if seed_test(c = c) :
    print("PrioritizedLevelReplay with seed! SUCCESSFUL")

#10 PrioritizedLevelReplay without seed
c = PrioritizedLevelReplay(task_space = task_space, observation_space = gym.spaces.Discrete(3))
if no_seed_test(c = c) :
    print("PrioritizedLevelReplay without seed! SUCCESSFUL")

#11 DomainRandomization with seed with sample_weights
space = TaskSpace(gym.spaces.Discrete(4), ["a", "b", "c","d"])
c = DomainRandomization(task_space = space, seed = seed, sample_weights = [0.6,0.2,0.1,0.1])
if seed_test(c = c) :
    print("DomainRandomization with seed with sample weights! SUCCESSFUL")

#2: DomainRandomization without seed
c = DomainRandomization(task_space = space, sample_weights = [0.3,0.2,0.4,0.1])
if no_seed_test(c = c) :
    print("DomainRandomization without seed with sample weights! SUCCESSFUL")


