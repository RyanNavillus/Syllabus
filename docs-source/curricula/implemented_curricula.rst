.. _Implemented Curricula:

==================
Curriculum Methods
==================

Syllabus has a small collection of curriculum learning methods implemented.These include simple techniques that are often used in practice but rarely highlighted in the literature,such as simulated annealing of difficulty, or sequential curricula of easy to hard tasks. We also have several popular automatic curriculum learning baselines; Domain Randomization, Prioritized Level Replay (Jiang et al. 2021), and the learning progress curriculum introduced in Kanitscheider et al. 2021. We plan to implement many more curriculum learning methods, but we have so far focused on methods which have had demonstrable empirical success in challenging RL environments.

-----------------------------------------------------------------------------------------
:mod:`Domain Randomization <syllabus.curricula.domain_randomization.DomainRandomization>`
-----------------------------------------------------------------------------------------

A simple but strong baseline for curriculum learning that uniformly samples a task from the task space.

---------------------------------------------------------------------------------
:mod:`Sequential Curriculum <syllabus.curricula.sequential.SequentialCurriculum>`
---------------------------------------------------------------------------------

Plays a sequence of curricula in order, either for a fixed time or after certain stopping criteria are met. It can be used to manually design curricula using existing domain knowledge by providing curricula or tasks in an order that will result in the best final performance.

--------------------------------------------------------------------------------
:mod:`Simple Box Curriculum <syllabus.curricula.simple_box.SimpleBoxCurriculum>`
--------------------------------------------------------------------------------

A simple curriculum that expands a zero-centered range from an initial range to a final range over a number of discrete steps. The curriculum increases the range to the next stage when a provided reward threshold is met.

------------------------------------------------------------------------------------------
:mod:`Learning Progress <syllabus.curricula.learning_progress.LearningProgressCurriculum>`
------------------------------------------------------------------------------------------

Uses a heuristic to estimate the learning progress of a task. It maintains a fast and slow exponential moving average (EMA) of the task completion rates for a set of discrete tasks. By measuring the difference between the fast and slow EMAs and reweighting it to adjust for the time delay created by the EMA, this method can estimate the learning progress of a task. The curriculum then assigns a higher probability to tasks with a very high or very low learning progress, indicating that the agent is either learning or forgetting the task. For more information you can read the original paper `Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft (Kanitscheider et al. 2021) <https://arxiv.org/abs/2106.14876.pdf>`_.

-------------------------------------------------------------------------------------------
:mod:`Prioritized Level Replay <syllabus.curricula.plr.plr_wrapper.PrioritizedLevelReplay>`
-------------------------------------------------------------------------------------------

A curriculum learning method that estimates an agent's regret on particular environment seed and uses a prioritized replay buffer to replay levels for which the agent has high regret. This implementation is based on the open-source original implementation at https://github.com/facebookresearch/level-replay, but has been modified to support Syllabus task spaces instead of just environment seeds. PLR has been used in multiple prominent RL works, such as `Human-Timescale Adaptation in an Open-Ended Task Space <https://arxiv.org/abs/2301.07608>`_. For more information you can read the original paper `Prioritized Level Replay (Jiang et al. 2021) <https://arxiv.org/abs/2010.03934.pdf>`_.

---------------------------------------------------------------------
:mod:`Self Play <syllabus.curricula.selfplay.SelfPlay>`
---------------------------------------------------------------------

A simple method for 2-player competitive games where the protagonist plays against a copy of itself. This produces an implicit curriculum of increasingly challenging opponents as the agent becomes more proficient at the game. However, because the opponent is always equally skilled at the game, it does not always produce the most useful reward signal. In addition, in transative games where it is not possible to strictly improve over a given strategy, Self Play can lead to oscillations in performance as the agent learns cyclical strategies to exploit its current behavior. The classic example of this is Rock Paper Scissors, where the agent will rotate between choosing rock, paper, and scissors over the course of training.

----------------------------------------------------------------------------------------
:mod:`Fictitious Self Play <syllabus.curricula.selfplay.FictitiousSelfPlay>`
----------------------------------------------------------------------------------------

An extension of Self Play that samples the opponent from previous iterations of the protagonist agent. The deep learning version is also sometimes called `Neural Ficitious Self Play <https://arxiv.org/abs/1603.01121>`_. This allows the agent to play against a variety of strategies that it has previously learned, and can be used to prevent oscillations in performance. This allows the agent to converge to a policy that is robust against all strategies it has previously learned. However, it can be less sample-efficient than Self Play because the agent must spend a disproportionate amount of time playing against older strategies that it has already learned to beat. Despite this, Fictitious Self Play has been used in several high profile successes in reinforcement learning including `AlphaGo <https://www.nature.com/articles/nature16961>`_ and `OpenAI Five <https://arxiv.org/pdf/1912.06680>`_, and agent trained to play Dota 2. The method was originally introduced in `"Iterative Solutions of Games by Fictitious Play" In Activity Analysis of Production and Allocation <https://cowles.yale.edu/sites/default/files/2022-09/m13-all.pdf>`_ by Brown, G. W. in 1951.

---------------------------------------------------------------------------------------------------------------------------
:mod:`Prioritized Fictitious Self Play <syllabus.curricula.selfplay.PrioritizedFictitiousSelfPlay>`
---------------------------------------------------------------------------------------------------------------------------

This method addresses some of the limitations of Fictitious Self Play by prioritizing agents which have a high winrate against the current agent. That way, the protagonist agent is trained against a variety of strategies but does not spend a disproportionate amount of time playing against weak opponents. This method in combination with many other curricula, was used to train `AlphaStar <https://www.nature.com/articles/s41586-019-1724-z>`_, the agent which learned to play Starcraft 2 at a high professional level.