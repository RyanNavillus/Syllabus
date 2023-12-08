==================
Curriculum Methods
==================

Syllabus has a small collection of curriculum learning methods implemented.These include simple techniques that are often used in practice
but rarely highlighted in the literature,such as simulated annealing of difficulty, or sequential curricula of easy to hard tasks. We also
have several popular curriculum learning baselines; Domain Randomization, Prioritized Level Replay (Jiang et al. 2021), and the learning
progress curriculum introduced in Kanitscheider et al. 2021.

-----------------------------------------------------------------------------------------
:mod:`Domain Randomization <syllabus.curricula.domain_randomization.DomainRandomization>`
-----------------------------------------------------------------------------------------

A simple but strong baseline for curriculum learning that uniformly samples a task from the task space.

---------------------------------------------------------------------------------
:mod:`Sequential Curriculum <syllabus.curricula.sequential.SequentialCurriculum>`
---------------------------------------------------------------------------------

Plays a provided list of tasks in order for a prespecified number of episodes.
It can be used to manually design curricula by providing tasks in an order that you feel will result in the best final performance.
*Coming Soon*: functional stopping criteria instead of a fixed number of episodes.

--------------------------------------------------------------------------------
:mod:`Simple Box Curriculum <syllabus.curricula.simple_box.SimpleBoxCurriculum>`
--------------------------------------------------------------------------------

A simple curriculum that expands a zero-centered range from an initial range to a final range over a number of discrete steps.
The curriculum increases the range to the next stage when a provided reward threshold is met.

------------------------------------------------------------------------------------------
:mod:`Learning Progress <syllabus.curricula.learning_progress.LearningProgressCurriculum>`
------------------------------------------------------------------------------------------

Uses a heuristic to estimate the learning progress of a task. It maintains a fast and slow exponential moving average (EMA) of the task
completion rates for a set of discrete tasks.
By measuring the difference between the fast and slow EMAs and reweighting it to adjust for the time delay created by the EMA, this method can
estimate the learning progress of a task.
The curriculum then assigns a higher probability to tasks with a very high or very low learning progress, indicating that the agent
is either learning or forgetting the task. For more information you can read the original paper
`Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft (Kanitscheider et al. 2021) <https://arxiv.org/pdf/2106.14876.pdf>`_.

-------------------------------------------------------------------------------------------
:mod:`Prioritized Level Replay <syllabus.curricula.plr.plr_wrapper.PrioritizedLevelReplay>`
-------------------------------------------------------------------------------------------

A curriculum learning method that estimates an agent's regret on particular environment instantiations and uses a prioritized replay buffer to
replay levels for which the agent has high regret. This implementation is based on the open-source original implementation at
https://github.com/facebookresearch/level-replay, but has been modified to support Syllabus task spaces instead of just environment seeds.
PLR has been used in multiple prominent RL works. For more information you can read the original paper
`Prioritized Level Replay (Jiang et al. 2021) <https://arxiv.org/pdf/2010.03934.pdf>`_.
