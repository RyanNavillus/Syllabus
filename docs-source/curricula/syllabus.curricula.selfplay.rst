.. _SelfPlay:

Self Play Curricula
===================

These curricula are designed to be used for 2-player competitive games. They save a history of previous agents and sample one to play against the current agent at the start of each episode. For more information on the co-player interface, see :ref:`Co-player`. For all of these methods, the user is required to have training code which supports self play already. These curricula provide the additional logic to store and sample from a history of agents.

Self Play
---------

A simple method for 2-player competitive games where the protagonist plays against a copy of itself. This produces an implicit curriculum of increasingly challenging opponents as the agent becomes more proficient at the game. However, because the opponent is always equally skilled at the game, it does not always produce the most useful reward signal. In addition, in transative games where it is not possible to strictly improve over a given strategy, Self Play can lead to oscillations in performance as the agent learns cyclical strategies to exploit its current behavior. The classic example of this is Rock Paper Scissors, where the agent will rotate between choosing rock, paper, and scissors over the course of training.

Note that this curriculum always returns the current agent's identifier 0, so it does not add anything to existing self play code. It is included for completeness and to allow comparisons between the other self play algorithms with only a single change to the training code.

.. autoclass:: syllabus.curricula.selfplay.SelfPlay
   :members:
   :undoc-members:
   :show-inheritance:

Fictitious Self Play
--------------------

An extension of Self Play that samples the opponent from previous iterations of the protagonist agent. The deep learning version is also sometimes called `Neural Ficitious Self Play <https://arxiv.org/abs/1603.01121>`_. This allows the agent to play against a variety of strategies that it has previously learned, and can be used to prevent oscillations in performance. This allows the agent to converge to a policy that is robust against all strategies it has previously learned. However, it can be less sample-efficient than Self Play because the agent must spend a disproportionate amount of time playing against older strategies that it has already learned to beat. Despite this, Fictitious Self Play has been used in several high profile successes in reinforcement learning including `AlphaGo <https://www.nature.com/articles/nature16961>`_ and `OpenAI Five <https://arxiv.org/pdf/1912.06680>`_, and agent trained to play Dota 2. The method was originally introduced in `"Iterative Solutions of Games by Fictitious Play" In Activity Analysis of Production and Allocation <https://cowles.yale.edu/sites/default/files/2022-09/m13-all.pdf>`_ by Brown, G. W. in 1951.

This curriculum stores a history of previous agents to the disk, and maintains a cache of recently used agents in memory. You can control the size of the history with the ``max_agents`` argument and the size of the cache with the ``max_loaded_agents`` argument. You can also control the storage path for the agent history with ``storage_path``, and the device that they will be loaded onto with ``device``.

.. autoclass:: syllabus.curricula.selfplay.FictitiousSelfPlay
   :members:
   :undoc-members:
   :show-inheritance:

Prioritized Fictitious Self Play
--------------------------------

This method addresses some of the limitations of Fictitious Self Play by prioritizing agents which have a high winrate against the current agent. That way, the protagonist agent is trained against a variety of strategies but does not spend a disproportionate amount of time playing against weak opponents. This method in combination with many other curricula, was used to train `AlphaStar <https://www.nature.com/articles/s41586-019-1724-z>`_, the agent which learned to play Starcraft 2 at a high professional level.

This curriculum stores a history of previous agents to the disk, and maintains a cache of recently used agents in memory. You can control the size of the history with the ``max_agents`` argument and the size of the cache with the ``max_loaded_agents`` argument. You can also control the storage path for the agent history with ``storage_path``, and the device that they will be loaded onto with ``device``.

.. autoclass:: syllabus.curricula.selfplay.PrioritizedFictitiousSelfPlay
   :members:
   :undoc-members:
   :show-inheritance: