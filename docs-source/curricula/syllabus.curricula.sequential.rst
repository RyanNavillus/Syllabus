.. _Sequential:

Sequential Curriculum
=====================

The ``Sequential`` curriculum allows you to manually design a sequence of tasks and curricula to train on in order. It provides flexible stopping conditions for transitioning to the next curriculum, including episode count, step count, and mean return. This curriculum passes all update data to the current curriculum in the sequence and sample from it when generating tasks. Note that updates are not passed to inactive curricula in the sequence, so automatic methods will not have a headstart on tracking relevant metrics.

The items in the sequence can be any `<Curriculum>` object, including `<constant>` for individual tasks. ``Sequential`` uses syntactic sugar to make it easier to define a sequence of curricula. It takes in a list of curricula ``curriculum_list`` and a list of stopping conditions ``stopping_conditions``. The ``curriculum_list`` can contain any of the following objects:

* A ``Curriculum`` object - will be directly added to the sequence.

* A single task - will be wrapped in a ``Constant`` curriculum.

* A list of tasks - will be wrapped in a ``DomainRandomization`` curriculum.

* A ``TaskSpace`` object - will be wrapped in a ``DomainRandomization`` curriculum.

Similarly, stopping conditions can be defined with a simple string format. These conditions are composed of metrics, comparison operators, the stopping value, and optional boolean operators to create composite conditions. The format supports the ``>``, ``>=``, ``=``, ``<=``, ``<`` comparison operators and the ``&`` and ``|`` boolean operators. The currently implemented metrics are:

* *"steps"* - the number of steps taken in the environment during this stage of the sequential curriculum.

* *"total_steps"* - the total number of steps taken in the environment during the entire sequential curriculum.

* *"episodes"* - the number of episodes completed in the environment during this stage of the sequential curriculum.

* *"total_episodes"* - the total number of episodes completed in the environment during the entire sequential curriculum.

* *"tasks"* - the number of tasks completed in the environment during this stage of the sequential curriculum.

* *"total_tasks"* - the total number of tasks completed in the environment during the entire sequential curriculum.

* *"episode_return"* - the mean return of the environment during this stage of the sequential curriculum.

During logging, the ``Sequential`` curriculum will accurately set the probability of tasks outside the current stage's task space to 0.

Sequential
----------

.. automodule:: syllabus.curricula.sequential
   :members:
   :undoc-members:
   :show-inheritance:
