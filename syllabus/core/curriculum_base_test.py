from curriculum_base import Curriculum
import gymnasium as gym
from multivariate_curriculum_wrapper import MultitaskWrapper
from task_space import TaskSpace
from domain_randomization import DomainRandomization
from learning_progress import LearningProgressCurriculum
from noop import NoopCurriculum
from sequential import SequentialCurriculum
from simple_box import SimpleBoxCurriculum
from annealing_box import AnnealingBoxCurriculum
from central_plr_wrapper import CentralizedPrioritizedLevelReplay
from plr_wrapper import PrioritizedLevelReplay
from task_sampler import TaskSampler
from curriculum_base import Curriculum
from curriculum_sync_wrapper import CurriculumWrapper



#1: Simple Test with DomainRandomization and LearningProgressCurriculum with and without seed
task_space = TaskSpace(200)
seeds = [3, None] #default

for s in seeds:
    print ("-------seed value--------")
    print (s)
    list = [DomainRandomization(task_space = task_space, seed = s),
    LearningProgressCurriculum(task_space = task_space, seed = s),
    ]
    for i in range(5):
        print("----", i)
        for c in list:
            print(c.sample())


#2: Simple Test with SequentialCurriculum
# Ask Ryan : Find out a valid stopping condition
# c = SequentialCurriculum(task_space = task_space, seed = seed, curriculum_list = list, stopping_conditions = [])
# for i in range(5):
#     print(c.sample())



#3: Simple Test with Curriculum that take box spaces
task_space = TaskSpace(gym.spaces.Box(low=0, high=1, shape=(2,)), [(0, 0), (0, 1), (1, 0), (1, 1)])
seed = None #default

listb = [SimpleBoxCurriculum(task_space = task_space, seed = seed),
AnnealingBoxCurriculum(task_space = task_space, seed = seed, start_values = [1,2], end_values = [1,5], total_steps = 1),
]  
print("--------")
for i in range(5):
    print("----", i)
    for c in listb:
        print(c.sample())

print("Its interesting to see that given these arbitrary values the sample returns the same value over 5 iterations")
# Its interesting to see that given these arbitrary values the sample returns the same value over 5 iterations




#3: Simple Test for plr functions
task_space = TaskSpace(200)
# seeds = [3, None] #default
s = 3


list1 = [CentralizedPrioritizedLevelReplay(task_space = task_space, seed = s),
    PrioritizedLevelReplay(task_space = task_space, seed = s, observation_space = TaskSpace(gym.spaces.Discrete(3), ["a", "b", "c"])),
    TaskSampler([TaskSpace(gym.spaces.Discrete(3), ["a", "b", "c"]),TaskSpace(200) ], seed = s)
    ]

print("--------")
for i in range(5):
    print("----", i)
    for c in list1:
        print(c.sample())


print("Same Problem here, seed is not impacting the randomness")
# Its interesting to see that given these arbitrary values the sample returns the same value over 5 iterations


#Testing MultitaskWrapper
# task_spaces = (gym.spaces.MultiDiscrete([3, 2]), gym.spaces.Discrete(3))
# task_names = ((("a", "b", "c"), (1, 0)), ("X", "Y", "Z"))
# task_space = TaskSpace(gym.spaces.Tuple(task_spaces), task_names)

# wrapper =  CurriculumWrapper(curriculum = DomainRandomization(task_space = task_space, seed = s), task_space = task_space, unwrapped = LearningProgressCurriculum(task_space = task_space, seed = s))
# c = MultitaskWrapper(wrapper)
# print(c.sample())


