=======================================
Implemented Curriculum Learning Methods
=======================================

Syllabus has a small collection of curriculum learning methods implemented.These include simple techniques that are often used in practice
but rarely highlighted in the literature,such as simulated annealing of difficulty, or sequential curricula of easy to hard tasks. We also
have several popular curriculum learning baselines; Domain Randomization, Prioritized Level Replay (Jiang et al. 2021), and the learning progress curriculum
introduced in Kanitscheider et al. 2021.

--------------------
Domain Randomization
--------------------

Domain Randomization is a simple but strong baseline for curriculum learning. It just uniformly samples a task from the task space.