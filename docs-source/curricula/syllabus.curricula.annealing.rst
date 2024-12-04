.. _SimulatedAnnealing:

Simulated Annealing
===================

Simulated annealing is an algorithm which slowly "anneals" continuous values towards a target value. This method allows you to set a starting value, a target value, and number of steps to anneal over for every parameter in a `BoxTaskSpace <syllabus.task_space.BoxTaskSpace>`. The number of steps can be specified as a single value, or individual values for each element of the box space.

**Note:** This implementation only updates the number of steps after each episode. For environments with extremely long episodes, this may not work as expected. However for most environments this should only result in a very small annealing delay.

.. automodule:: syllabus.curricula.annealing_box
   :members:
   :undoc-members:
   :show-inheritance: